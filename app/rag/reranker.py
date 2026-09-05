"""Cross-encoder reranking with bge-reranker-v2-m3 (ТЗ slide 9).

Hybrid search is cheap and approximate: it compares a question vector to an article vector
computed independently. The cross-encoder reads the ``(question, article)`` pair together,
so it separates близкие формулировки that the bi-encoder collapses. It is also the most
expensive step on CPU, which is why ``RERANK_TOP_K`` is capped and the stage can be
disabled outright.

Its raw output is an unbounded logit. We squash it with a sigmoid so that
``ABSTAIN_THRESHOLD`` is a stable, interpretable number rather than a model-specific
magic constant.
"""

from __future__ import annotations

import logging
import math
import threading

from app.config import get_settings

log = logging.getLogger(__name__)


class Reranker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._model = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from FlagEmbedding import FlagReranker

            log.info("loading reranker %s", self._settings.reranker_model)
            self._model = FlagReranker(
                self._settings.reranker_model, use_fp16=self._settings.use_fp16
            )
            log.info("reranker ready")

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Relevance in [0, 1] for each document, aligned with ``documents``."""
        if not documents:
            return []
        self.load()
        assert self._model is not None
        raw = self._model.compute_score([[query, document] for document in documents])
        if isinstance(raw, (int, float)):
            raw = [raw]
        return [_sigmoid(float(value)) for value in raw]


def _sigmoid(value: float) -> float:
    # Guard against overflow on large-magnitude logits.
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
