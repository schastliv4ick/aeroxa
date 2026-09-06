from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings
from app.preflight import check_model_cache
from app.rag.embedder import get_embedder
from app.rag.reranker import get_reranker
from app.storage.db import query_log


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )
    # Weights load lazily, so a full cache disk would otherwise surface as an obscure
    # "Can't load the model" deep inside the first /ask. Say it up front instead.
    check_model_cache([settings.embedding_model, settings.reranker_model])

    await query_log.connect()

    if settings.models_eager_load:
        # Loading ~2.3 GB of weights takes a while. Eager loading trades a slow start for
        # a predictable first request; lazy loading is the default so tests stay fast.
        get_embedder().load()
        if settings.rerank_enabled:
            get_reranker().load()

    yield
    await query_log.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Aeroxa Legal RAG",
        description="RAG over Russian Federation legislation. See docs/SPEC.md.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port)
