"""The RAG pipeline: condense -> retrieve -> filter -> rerank -> generate, or abstain.

Ordering and stage responsibilities follow ТЗ slides 8-10. The abstention gate sits
*before* generation on purpose: if the corpus does not support an answer, the cheapest and
safest thing to do is not to call the model at all.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from app.config import get_settings
from app.rag import prompts
from app.rag.embedder import get_embedder
from app.rag.llm import LLMError, get_llm
from app.rag.reranker import get_reranker
from app.storage.qdrant_store import Hit, get_store

log = logging.getLogger(__name__)


@dataclass
class Fragment:
    """A retrieved chunk together with the score that put it in front of the model."""

    payload: dict
    score: float
    retrieval_score: float

    def as_context(self) -> dict:
        return {"breadcrumbs": self.payload.get("breadcrumbs", ""), "text": self.payload.get("text", "")}


# Why no generated answer was produced. The website and the lawyer queue must be able to
# tell "the corpus does not cover this" from "our generator was down" -- they are different
# facts and only one of them is about the law.
NO_ANSWER_LOW_RELEVANCE = "low_relevance"
NO_ANSWER_GENERATION_UNAVAILABLE = "generation_unavailable"


@dataclass
class AnswerResult:
    answer: str
    language: str
    abstained: bool
    escalate: bool
    reason: str | None = None
    fragments: list[Fragment] = field(default_factory=list)
    standalone_question: str | None = None
    model: str | None = None
    latency_ms: int = 0


class Pipeline:
    def __init__(self) -> None:
        self._settings = get_settings()

    # -- retrieval ---------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        act_ids: list[str] | None = None,
        rerank: bool | None = None,
    ) -> list[Fragment]:
        settings = self._settings
        limit = top_k or settings.retrieve_top_k
        use_rerank = settings.rerank_enabled if rerank is None else rerank

        # Model inference is CPU-bound and blocking; keep the event loop free.
        embedding = await asyncio.to_thread(get_embedder().encode_one, query)
        hits: list[Hit] = await asyncio.to_thread(
            get_store().hybrid_search, embedding, limit=limit, act_ids=act_ids
        )
        if not hits:
            return []

        if not use_rerank:
            candidates = [
                Fragment(payload=hit.payload, score=hit.score, retrieval_score=hit.score)
                for hit in hits
            ]
        else:
            documents = [
                f"{hit.payload.get('breadcrumbs', '')}\n{hit.payload.get('text', '')}"
                for hit in hits
            ]
            scores = await asyncio.to_thread(get_reranker().score, query, documents)
            candidates = sorted(
                (
                    Fragment(payload=hit.payload, score=score, retrieval_score=hit.score)
                    for hit, score in zip(hits, scores)
                ),
                key=lambda fragment: fragment.score,
                reverse=True,
            )

        return _diversify(candidates, settings.rerank_top_n, settings.max_parts_per_article)

    # -- generation --------------------------------------------------------

    async def answer(
        self,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
        language: str | None = None,
        top_k: int | None = None,
        act_ids: list[str] | None = None,
    ) -> AnswerResult:
        started = time.perf_counter()
        answer_language = language or prompts.detect_language(question)

        standalone = question
        if history and self._settings.condense_enabled:
            standalone = await self._condense(question, history)

        fragments = await self.retrieve(standalone, top_k=top_k, act_ids=act_ids)

        if self._should_abstain(fragments, answer_language):
            log.info(
                "abstaining (%s): best score %.3f < %.3f",
                answer_language,
                fragments[0].score if fragments else 0.0,
                self.abstain_threshold_for(answer_language),
            )
            return AnswerResult(
                answer=prompts.ABSTENTION_MESSAGES.get(
                    answer_language, prompts.ABSTENTION_MESSAGES["en"]
                ),
                language=answer_language,
                abstained=True,
                escalate=True,
                reason=NO_ANSWER_LOW_RELEVANCE,
                fragments=fragments,
                standalone_question=standalone if standalone != question else None,
                latency_ms=_elapsed_ms(started),
            )

        messages = [
            {
                "role": "system",
                "content": prompts.SYSTEM_PROMPT.format(
                    language=prompts.LANGUAGE_NAMES.get(answer_language, "the user's language")
                ),
            },
            {
                "role": "user",
                "content": prompts.USER_PROMPT.format(
                    context=prompts.build_context([f.as_context() for f in fragments]),
                    question=question,
                ),
            },
        ]

        llm = get_llm()
        try:
            answer = await llm.complete(messages)
        except LLMError:
            # Generation failing must not produce a fabricated answer. Escalate instead --
            # but never by claiming the law is silent: retrieval succeeded, so the
            # fragments are returned and the message says the generator is down.
            log.exception("generation unavailable; escalating with retrieved fragments")
            return AnswerResult(
                answer=prompts.GENERATION_UNAVAILABLE_MESSAGES.get(
                    answer_language, prompts.GENERATION_UNAVAILABLE_MESSAGES["en"]
                ),
                language=answer_language,
                abstained=True,
                escalate=True,
                reason=NO_ANSWER_GENERATION_UNAVAILABLE,
                fragments=fragments,
                standalone_question=standalone if standalone != question else None,
                latency_ms=_elapsed_ms(started),
            )

        return AnswerResult(
            answer=answer,
            language=answer_language,
            abstained=False,
            escalate=False,
            fragments=fragments,
            standalone_question=standalone if standalone != question else None,
            model=llm.model,
            latency_ms=_elapsed_ms(started),
        )

    # -- internals ---------------------------------------------------------

    def abstain_threshold_for(self, language: str) -> float:
        """The cross-encoder's score scale depends on the language pair, so the threshold
        does too. See ``abstain_threshold_by_language`` in the config for why."""
        return self._settings.abstain_threshold_by_language.get(
            language, self._settings.abstain_threshold
        )

    def _should_abstain(self, fragments: list[Fragment], language: str = "ru") -> bool:
        if not fragments:
            return True
        if not self._settings.rerank_enabled:
            # RRF scores are not calibrated, so the threshold is meaningless without the
            # cross-encoder. Having any hit at all is the only signal available.
            return False
        return fragments[0].score < self.abstain_threshold_for(language)

    async def _condense(self, question: str, history: list[dict[str, str]]) -> str:
        turns = history[-self._settings.condense_max_history_turns :]
        rendered = "\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)
        try:
            rewritten = await get_llm().complete(
                [
                    {
                        "role": "user",
                        "content": prompts.CONDENSE_PROMPT.format(
                            history=rendered, question=question
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=200,
            )
        except LLMError:
            # Retrieval on the raw question is degraded but functional; do not fail here.
            log.warning("condensing failed; retrieving on the raw question")
            return question
        return rewritten or question


def _diversify(candidates: list[Fragment], limit: int, max_per_article: int) -> list[Fragment]:
    """Take the best ``limit`` fragments, allowing at most ``max_per_article`` per article.

    Parts of one long article score almost identically on a query about that article, so
    without a cap a single Статья fills every context slot and the model never sees the
    related norms it needs. Order is preserved, so the best fragment always survives; if
    the cap leaves the context short, the skipped fragments are added back rather than
    returning less context than asked for.
    """
    if max_per_article <= 0:
        return candidates[:limit]

    selected: list[Fragment] = []
    overflow: list[Fragment] = []
    seen: dict[tuple[str, str], int] = {}

    for fragment in candidates:
        payload = fragment.payload
        key = (payload.get("act_id", ""), payload.get("article_number", ""))
        if seen.get(key, 0) < max_per_article:
            seen[key] = seen.get(key, 0) + 1
            selected.append(fragment)
            if len(selected) == limit:
                return selected
        else:
            overflow.append(fragment)

    selected.extend(overflow[: limit - len(selected)])
    return selected[:limit]


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline
