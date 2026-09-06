from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.api.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    HealthResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.config import get_settings
from app.rag.embedder import get_embedder
from app.rag.pipeline import Fragment, get_pipeline
from app.rag.reranker import get_reranker
from app.storage.db import query_log
from app.storage.qdrant_store import get_store

log = logging.getLogger(__name__)
router = APIRouter()


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """No-op unless API_KEY is configured, so local development stays frictionless."""
    expected = get_settings().api_key
    if expected and x_api_key != expected:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-API-Key")


@router.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
async def ask(request: AskRequest) -> AskResponse:
    result = await get_pipeline().answer(
        request.question,
        history=[turn.model_dump() for turn in request.history],
        language=request.language,
        top_k=request.top_k,
        act_ids=request.act_ids,
    )
    citations = [_to_citation(index, fragment) for index, fragment in enumerate(result.fragments, 1)]

    await query_log.record(
        question=request.question,
        standalone=result.standalone_question,
        language=result.language,
        answer=result.answer,
        abstained=result.abstained,
        escalated=result.escalate,
        citations=[citation.model_dump() for citation in citations],
        model=result.model,
        latency_ms=result.latency_ms,
        reason=result.reason,
    )

    return AskResponse(
        answer=result.answer,
        language=result.language,
        abstained=result.abstained,
        escalate=result.escalate,
        reason=result.reason,
        citations=citations,
        standalone_question=result.standalone_question,
        model=result.model,
        latency_ms=result.latency_ms,
    )


@router.post("/search", response_model=SearchResponse, dependencies=[Depends(require_api_key)])
async def search(request: SearchRequest) -> SearchResponse:
    """Retrieval without generation. Used to debug recall and to measure it later."""
    started = time.perf_counter()
    fragments = await get_pipeline().retrieve(
        request.query, top_k=request.top_k, act_ids=request.act_ids, rerank=request.rerank
    )
    return SearchResponse(
        query=request.query,
        hits=[
            SearchHit(
                chunk_id=fragment.payload.get("chunk_id", ""),
                score=fragment.score,
                retrieval_score=fragment.retrieval_score,
                act_title=fragment.payload.get("act_title", ""),
                article_number=fragment.payload.get("article_number"),
                article_title=fragment.payload.get("article_title"),
                chapter=fragment.payload.get("chapter"),
                revision_index=fragment.payload.get("revision_index"),
                url=fragment.payload.get("url"),
                text=fragment.payload.get("text", ""),
            )
            for fragment in fragments
        ],
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    qdrant_status = "ok"
    points: int | None = None
    try:
        store = get_store()
        points = store.count() if store.client.collection_exists(store.collection) else None
        if points is None:
            qdrant_status = "collection missing"
    except Exception as error:  # noqa: BLE001 - health must report, not raise
        qdrant_status = f"unreachable: {error}"

    return HealthResponse(
        status="ok" if qdrant_status == "ok" and points else "degraded",
        qdrant=qdrant_status,
        collection=settings.qdrant_collection,
        points=points,
        embedder_loaded=get_embedder().is_loaded,
        reranker_loaded=get_reranker().is_loaded,
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
    )


def _to_citation(index: int, fragment: Fragment) -> Citation:
    payload = fragment.payload
    text = payload.get("text", "")
    return Citation(
        n=index,
        act_title=payload.get("act_title", ""),
        act_number=payload.get("act_number"),
        article_number=payload.get("article_number"),
        article_title=payload.get("article_title"),
        chapter=payload.get("chapter"),
        revision_index=payload.get("revision_index"),
        revision_date=payload.get("revision_date"),
        url=payload.get("url"),
        score=round(fragment.score, 4),
        snippet=text[:400] + ("…" if len(text) > 400 else ""),
    )
