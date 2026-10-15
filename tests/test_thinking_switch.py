"""Qwen3's reasoning switch.

On a CPU-hosted Qwen3 the reasoning pass is most of the latency, so "/no_think" is appended
to every prompt the pipeline sends, unless LLM_DISABLE_THINKING is turned off.
"""

from __future__ import annotations

import asyncio

import pytest

from app.config import get_settings


@pytest.fixture
def make_pipeline(monkeypatch):
    """Build a pipeline with retrieval stubbed and every LLM call recorded."""
    from app.rag import pipeline as pipeline_module

    calls: list[list[dict]] = []

    class RecordingLLM:
        model = "qwen3"

        async def complete(self, messages, *args, **kwargs):
            calls.append(messages)
            return "Ответ [1]"

    async def fake_retrieve(*args, **kwargs):
        payload = {
            "act_id": "fz-14-ooo",
            "article_number": "14",
            "act_title": "Об ООО",
            "breadcrumbs": "Об ООО · Статья 14",
            "text": "Размер уставного капитала не менее десяти тысяч рублей.",
        }
        return [pipeline_module.Fragment(payload=payload, score=0.99, retrieval_score=0.99)]

    def _make(disable_thinking: bool):
        monkeypatch.setenv("LLM_DISABLE_THINKING", str(disable_thinking).lower())
        get_settings.cache_clear()
        p = pipeline_module.Pipeline()
        monkeypatch.setattr(p, "retrieve", fake_retrieve)
        monkeypatch.setattr(pipeline_module, "get_llm", lambda: RecordingLLM())
        return p, calls

    yield _make
    get_settings.cache_clear()


HISTORY = [
    {"role": "user", "content": "Каков минимальный уставный капитал ООО?"},
    {"role": "assistant", "content": "10 000 рублей [1]."},
]


def test_no_think_is_appended_to_the_answer_and_condense_prompts(make_pipeline):
    p, calls = make_pipeline(disable_thinking=True)

    asyncio.run(p.answer("А для АО?", history=HISTORY))

    condense, answer = calls
    assert condense[-1]["content"].endswith("/no_think")
    assert answer[0]["role"] == "system"
    assert answer[0]["content"].endswith("/no_think")


def test_prompts_are_left_alone_when_thinking_is_allowed(make_pipeline):
    p, calls = make_pipeline(disable_thinking=False)

    asyncio.run(p.answer("А для АО?", history=HISTORY))

    assert calls, "the LLM was never called"
    assert not any("/no_think" in m["content"] for messages in calls for m in messages)


def test_extra_body_is_forwarded_to_the_backend(monkeypatch):
    """Ollama only drops Qwen3's reasoning when the request body says so."""
    from app.rag import llm as llm_module

    monkeypatch.setenv("LLM_EXTRA_BODY", '{"reasoning_effort": "none"}')
    get_settings.cache_clear()
    client = llm_module.LLMClient()
    sent = {}

    class _Message:
        content = "Ответ"

    class _Response:
        choices = [type("Choice", (), {"message": _Message()})()]

    async def fake_create(**kwargs):
        sent.update(kwargs)
        return _Response()

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)

    asyncio.run(client.complete([{"role": "user", "content": "?"}]))
    get_settings.cache_clear()

    assert sent["extra_body"] == {"reasoning_effort": "none"}
