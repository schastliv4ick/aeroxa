"""API contract tests.

The pipeline is stubbed out: these assert the shape of the contract the website codes
against and the behaviour of the abstention and auth paths, without loading 2.3 GB of
model weights or requiring Qdrant.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.config import get_settings
from app.main import create_app
from app.rag.pipeline import AnswerResult, Fragment

PAYLOAD = {
    "chunk_id": "102051516:57:14:0",
    "act_title": "Об обществах с ограниченной ответственностью",
    "act_number": "14-ФЗ",
    "article_number": "14",
    "article_title": "Уставный капитал общества",
    "chapter": "Глава II",
    "revision_index": 57,
    "revision_date": "2026-08-04",
    "url": "http://example.invalid/doc",
    "text": "Размер уставного капитала общества должен быть не менее чем десять тысяч рублей.",
    "breadcrumbs": "Акт · редакция 57\nСтатья 14",
}


class StubPipeline:
    def __init__(self, result: AnswerResult, fragments: list[Fragment] | None = None) -> None:
        self._result = result
        self._fragments = fragments or []

    async def answer(self, question, **kwargs):  # noqa: ANN001, ANN003
        return self._result

    async def retrieve(self, query, **kwargs):  # noqa: ANN001, ANN003
        return self._fragments


@pytest.fixture(autouse=True)
def _no_query_log(monkeypatch):
    async def noop(**_):
        return None

    monkeypatch.setattr(routes.query_log, "record", noop)


@pytest.fixture
def client():
    return TestClient(create_app())


def _install(monkeypatch, pipeline):
    monkeypatch.setattr(routes, "get_pipeline", lambda: pipeline)


def test_ask_returns_answer_with_numbered_citations(client, monkeypatch):
    _install(
        monkeypatch,
        StubPipeline(
            AnswerResult(
                answer="Минимальный уставный капитал — 10 000 рублей (Статья 14) [1].",
                language="ru",
                abstained=False,
                escalate=False,
                fragments=[Fragment(payload=PAYLOAD, score=0.94, retrieval_score=0.71)],
                model="qwen3",
                latency_ms=1234,
            )
        ),
    )
    response = client.post("/ask", json={"question": "Каков минимальный уставный капитал ООО?"})
    assert response.status_code == 200

    body = response.json()
    assert body["abstained"] is False
    assert body["escalate"] is False
    assert body["language"] == "ru"
    assert body["model"] == "qwen3"

    citation = body["citations"][0]
    assert citation["n"] == 1
    assert citation["article_number"] == "14"
    # The edition must reach the client: an answer is only as current as its source.
    assert citation["revision_index"] == 57
    assert citation["revision_date"] == "2026-08-04"


def test_abstention_escalates_and_returns_no_invented_answer(client, monkeypatch):
    from app.rag.prompts import ABSTENTION_MESSAGES

    _install(
        monkeypatch,
        StubPipeline(
            AnswerResult(
                answer=ABSTENTION_MESSAGES["zh"],
                language="zh",
                abstained=True,
                escalate=True,
                latency_ms=800,
            )
        ),
    )
    response = client.post("/ask", json={"question": "俄罗斯的遗产税是多少？"})
    body = response.json()
    assert body["abstained"] is True
    assert body["escalate"] is True
    assert body["citations"] == []
    assert body["answer"] == ABSTENTION_MESSAGES["zh"]


def test_search_returns_retrieval_without_generation(client, monkeypatch):
    _install(
        monkeypatch,
        StubPipeline(
            AnswerResult(answer="", language="ru", abstained=False, escalate=False),
            fragments=[Fragment(payload=PAYLOAD, score=0.81, retrieval_score=0.6)],
        ),
    )
    response = client.post("/search", json={"query": "уставный капитал"})
    hit = response.json()["hits"][0]
    assert hit["chunk_id"] == "102051516:57:14:0"
    assert hit["score"] == 0.81
    assert hit["retrieval_score"] == 0.6


def test_question_is_required(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_history_role_is_validated(client):
    response = client.post(
        "/ask",
        json={"question": "и что дальше?", "history": [{"role": "system", "content": "x"}]},
    )
    assert response.status_code == 422


def test_api_key_is_enforced_when_configured(monkeypatch, client):
    get_settings.cache_clear()
    monkeypatch.setenv("API_KEY", "secret")
    try:
        assert client.post("/ask", json={"question": "тест"}).status_code == 401
    finally:
        get_settings.cache_clear()


def test_api_key_is_not_required_by_default(client, monkeypatch):
    _install(
        monkeypatch,
        StubPipeline(AnswerResult(answer="ok", language="ru", abstained=False, escalate=False)),
    )
    assert client.post("/ask", json={"question": "тест"}).status_code == 200


def test_generation_outage_does_not_claim_the_law_is_silent(client, monkeypatch):
    """Regression: an LLM failure used to return the "no provision found" message.

    Retrieval had succeeded — the correct article was in hand — so telling the client that
    Russian law contains nothing on the point was simply false, and it put a misleading
    "nothing found" row in the lawyer's queue. The outage must be reported as an outage.
    """
    from app.rag.prompts import ABSTENTION_MESSAGES, GENERATION_UNAVAILABLE_MESSAGES

    _install(
        monkeypatch,
        StubPipeline(
            AnswerResult(
                answer=GENERATION_UNAVAILABLE_MESSAGES["ru"],
                language="ru",
                abstained=True,
                escalate=True,
                reason="generation_unavailable",
                fragments=[Fragment(payload=PAYLOAD, score=0.99, retrieval_score=0.8)],
                latency_ms=900,
            )
        ),
    )
    body = client.post("/ask", json={"question": "Каков минимальный уставный капитал?"}).json()

    assert body["reason"] == "generation_unavailable"
    assert body["escalate"] is True
    # The citations survive: we found the norm, we just could not phrase the answer.
    assert body["citations"][0]["article_number"] == "14"
    # And we must not have said the corpus is silent.
    assert body["answer"] != ABSTENTION_MESSAGES["ru"]
    assert "не нашлось нормы" not in body["answer"]


def test_low_relevance_abstention_is_labelled_distinctly(client, monkeypatch):
    from app.rag.prompts import ABSTENTION_MESSAGES

    _install(
        monkeypatch,
        StubPipeline(
            AnswerResult(
                answer=ABSTENTION_MESSAGES["ru"],
                language="ru",
                abstained=True,
                escalate=True,
                reason="low_relevance",
                latency_ms=700,
            )
        ),
    )
    body = client.post("/ask", json={"question": "Какая ставка НДС?"}).json()
    assert body["reason"] == "low_relevance"
    assert body["citations"] == []


def test_successful_answer_has_no_reason(client, monkeypatch):
    _install(
        monkeypatch,
        StubPipeline(
            AnswerResult(
                answer="Ответ со ссылкой на Статью 14 [1].",
                language="ru",
                abstained=False,
                escalate=False,
                fragments=[Fragment(payload=PAYLOAD, score=0.94, retrieval_score=0.7)],
                model="qwen3",
                latency_ms=1200,
            )
        ),
    )
    body = client.post("/ask", json={"question": "тест"}).json()
    assert body["reason"] is None
    assert body["abstained"] is False


def test_every_supported_language_has_an_outage_message():
    from app.rag.prompts import ABSTENTION_MESSAGES, GENERATION_UNAVAILABLE_MESSAGES

    assert set(GENERATION_UNAVAILABLE_MESSAGES) == set(ABSTENTION_MESSAGES)
    for language, text in GENERATION_UNAVAILABLE_MESSAGES.items():
        assert text != ABSTENTION_MESSAGES[language]
