"""BGE-M3 embeddings: one multilingual space, dense and sparse from one forward pass.

Chosen per ТЗ slide 8. A Chinese question and a Russian article land in the same vector
space, so cross-lingual retrieval works without translating the corpus. The same pass
yields lexical (sparse) weights, which carry exact article numbers and legal terms that
dense vectors blur -- the hybrid requirement without a second model.

The model is ~2.3 GB and loads lazily on first use so that importing the app (for tests,
for ``--help``) stays cheap.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from app.config import get_settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Embedding:
    dense: list[float]
    sparse: dict[int, float]  # token id -> lexical weight


class Embedder:
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
            from FlagEmbedding import BGEM3FlagModel

            log.info("loading embedding model %s", self._settings.embedding_model)
            self._model = BGEM3FlagModel(
                self._settings.embedding_model,
                use_fp16=self._settings.use_fp16,
            )
            log.info("embedding model ready")

    def encode(self, texts: list[str], *, batch_size: int | None = None) -> list[Embedding]:
        if not texts:
            return []
        self.load()
        assert self._model is not None
        output = self._model.encode(
            texts,
            batch_size=batch_size or self._settings.embed_batch_size,
            max_length=self._settings.embedding_max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        embeddings = []
        for dense, sparse in zip(output["dense_vecs"], output["lexical_weights"]):
            embeddings.append(
                Embedding(
                    dense=[float(value) for value in dense],
                    # FlagEmbedding returns token ids as strings keyed to float32 weights.
                    sparse={int(token): float(weight) for token, weight in sparse.items() if float(weight) > 0},
                )
            )
        return embeddings

    def encode_one(self, text: str) -> Embedding:
        return self.encode([text])[0]


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
