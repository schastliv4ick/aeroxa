"""Qdrant collection: named dense + sparse vectors, fused server-side with RRF.

One collection holds every act. The in-force redaction is selected by a payload filter
rather than by separate collections, so promoting a new edition is an update of two
payload fields instead of a reindex of everything.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from app.config import get_settings
from app.rag.embedder import Embedding

log = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


@dataclass
class Hit:
    chunk_id: str
    score: float
    payload: dict


class QdrantStore:
    def __init__(self, client: QdrantClient | None = None) -> None:
        settings = get_settings()
        self._settings = settings
        self._collection = settings.qdrant_collection
        self._client = client or QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=60,
        )

    @property
    def client(self) -> QdrantClient:
        return self._client

    @property
    def collection(self) -> str:
        return self._collection

    def ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection):
            return
        log.info("creating collection %s", self._collection)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=self._settings.embedding_dim,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams()},
        )
        # Indexed payload fields are the ones we filter on. Without these, filtering
        # degrades to a full scan as the corpus grows.
        for field, schema in (
            ("in_force", models.PayloadSchemaType.BOOL),
            ("jurisdiction", models.PayloadSchemaType.KEYWORD),
            ("act_id", models.PayloadSchemaType.KEYWORD),
            ("nd", models.PayloadSchemaType.KEYWORD),
            ("revision_index", models.PayloadSchemaType.INTEGER),
            ("article_number", models.PayloadSchemaType.KEYWORD),
        ):
            self._client.create_payload_index(
                collection_name=self._collection, field_name=field, field_schema=schema
            )

    def count(self) -> int:
        return self._client.count(self._collection, exact=True).count

    def retire_act(self, nd: str, keep_revision: int) -> bool:
        """Mark every chunk of ``nd`` from an older redaction as no longer in force.

        The chunks are kept rather than deleted so a past answer stays auditable, but they
        can no longer be retrieved -- the ТЗ's "источник ошибок №1" guard.
        """
        result = self._client.set_payload(
            collection_name=self._collection,
            payload={"in_force": False},
            points=models.Filter(
                must=[models.FieldCondition(key="nd", match=models.MatchValue(value=nd))],
                must_not=[
                    models.FieldCondition(
                        key="revision_index", match=models.MatchValue(value=keep_revision)
                    )
                ],
            ),
            wait=True,
        )
        return result.status == models.UpdateStatus.COMPLETED

    def upsert(self, chunk_ids: list[str], embeddings: list[Embedding], payloads: list[dict]) -> None:
        points = [
            models.PointStruct(
                # Deterministic id from chunk_id: re-running ingestion updates in place
                # instead of duplicating points.
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
                vector={
                    DENSE_VECTOR: embedding.dense,
                    SPARSE_VECTOR: models.SparseVector(
                        indices=list(embedding.sparse.keys()),
                        values=list(embedding.sparse.values()),
                    ),
                },
                payload=payload,
            )
            for chunk_id, embedding, payload in zip(chunk_ids, embeddings, payloads)
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=True)

    def hybrid_search(
        self,
        embedding: Embedding,
        *,
        limit: int,
        act_ids: list[str] | None = None,
        in_force_only: bool = True,
    ) -> list[Hit]:
        """Dense and sparse prefetch fused with Reciprocal Rank Fusion.

        RRF is used rather than score-weighted fusion because cosine similarity and
        lexical weights are not on a comparable scale; ranks are.
        """
        conditions: list[models.Condition] = [
            models.FieldCondition(key="jurisdiction", match=models.MatchValue(value="RU"))
        ]
        if in_force_only:
            conditions.append(
                models.FieldCondition(key="in_force", match=models.MatchValue(value=True))
            )
        if act_ids:
            conditions.append(
                models.FieldCondition(key="act_id", match=models.MatchAny(any=act_ids))
            )
        query_filter = models.Filter(must=conditions)

        prefetch = [
            models.Prefetch(
                query=embedding.dense, using=DENSE_VECTOR, limit=limit, filter=query_filter
            )
        ]
        if embedding.sparse:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=list(embedding.sparse.keys()),
                        values=list(embedding.sparse.values()),
                    ),
                    using=SPARSE_VECTOR,
                    limit=limit,
                    filter=query_filter,
                )
            )

        response = self._client.query_points(
            collection_name=self._collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [
            Hit(
                chunk_id=(point.payload or {}).get("chunk_id", str(point.id)),
                score=point.score,
                payload=point.payload or {},
            )
            for point in response.points
        ]


_store: QdrantStore | None = None


def get_store() -> QdrantStore:
    global _store
    if _store is None:
        _store = QdrantStore()
    return _store
