"""Vector store tests against an in-process Qdrant.

qdrant-client's local mode runs the same query API as the server, so collection layout,
sparse vectors, RRF fusion and the in-force filter are all exercised for real — without
Docker and without loading any model. Payload indexes are a no-op locally (they are a
server-side performance feature), which is the only behavioural difference that matters
here.
"""

from __future__ import annotations

import random
import warnings

import pytest
from qdrant_client import QdrantClient

from app.rag.embedder import Embedding
from app.storage.qdrant_store import QdrantStore

CURRENT_REVISION = 57


def _payload(index: int, revision: int = CURRENT_REVISION) -> dict:
    return {
        "chunk_id": f"102051516:{revision}:{index}:0",
        "act_id": "fz-14-ooo",
        "nd": "102051516",
        "revision_index": revision,
        "article_number": str(index),
        "in_force": True,
        "jurisdiction": "RU",
        "text": f"Текст статьи {index}.",
        "breadcrumbs": "Об ООО · Статья",
    }


def _embedding(seed: int) -> Embedding:
    rng = random.Random(seed)
    return Embedding(
        dense=[rng.random() for _ in range(1024)],
        sparse={seed * 10 + 1: 0.9, seed * 10 + 2: 0.4},
    )


@pytest.fixture
def store():
    with warnings.catch_warnings():
        # Local mode warns that payload indexes do nothing; expected and harmless.
        warnings.simplefilter("ignore", UserWarning)
        store = QdrantStore(client=QdrantClient(":memory:"))
        store.ensure_collection()
        yield store


@pytest.fixture
def populated(store):
    embeddings = [_embedding(i) for i in range(5)]
    store.upsert(
        [_payload(i)["chunk_id"] for i in range(5)],
        embeddings,
        [_payload(i) for i in range(5)],
    )
    return store, embeddings


def test_hybrid_search_ranks_the_matching_chunk_first(populated):
    store, embeddings = populated
    hits = store.hybrid_search(embeddings[2], limit=3)
    assert hits[0].chunk_id == "102051516:57:2:0"


def test_upsert_is_idempotent(populated):
    """Re-running ingestion updates points in place instead of duplicating them."""
    store, embeddings = populated
    assert store.count() == 5
    store.upsert(
        [_payload(i)["chunk_id"] for i in range(5)], embeddings, [_payload(i) for i in range(5)]
    )
    assert store.count() == 5


def test_superseded_edition_becomes_unretrievable(populated):
    """The ТЗ's "источник ошибок №1" guard: a stale redaction must never reach an answer."""
    store, embeddings = populated
    assert store.hybrid_search(embeddings[2], limit=3)

    # A newer edition arrives; everything not on it is retired.
    store.retire_act("102051516", keep_revision=CURRENT_REVISION + 1)

    assert store.hybrid_search(embeddings[2], limit=3) == []
    # Retired chunks are kept, not deleted, so past answers stay auditable.
    assert store.count() == 5


def test_retiring_one_act_leaves_others_alone(store):
    store.upsert(
        ["102051516:57:1:0", "999:3:1:0"],
        [_embedding(1), _embedding(2)],
        [
            _payload(1),
            {**_payload(1), "chunk_id": "999:3:1:0", "nd": "999", "revision_index": 3},
        ],
    )
    store.retire_act("102051516", keep_revision=99)
    remaining = {hit.payload["nd"] for hit in store.hybrid_search(_embedding(2), limit=5)}
    assert remaining == {"999"}


def test_act_filter_restricts_retrieval(populated):
    store, embeddings = populated
    assert store.hybrid_search(embeddings[0], limit=5, act_ids=["fz-14-ooo"])
    assert store.hybrid_search(embeddings[0], limit=5, act_ids=["tk-rf"]) == []
