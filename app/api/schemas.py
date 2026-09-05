"""Request and response models — the contract the website backend codes against."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[Turn] = Field(default_factory=list, max_length=20)
    # ISO-639-1. Omit to detect from the question's script.
    language: Literal["ru", "zh", "en"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=200)
    # Restrict retrieval to specific acts by their acts.yaml id.
    act_ids: list[str] | None = None


class Citation(BaseModel):
    n: int
    act_title: str
    act_number: str | None = None
    article_number: str | None = None
    article_title: str | None = None
    chapter: str | None = None
    revision_index: int | None = None
    revision_date: str | None = None
    url: str | None = None
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    language: str
    abstained: bool
    escalate: bool
    citations: list[Citation]
    standalone_question: str | None = None
    model: str | None = None
    latency_ms: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=200)
    act_ids: list[str] | None = None
    rerank: bool | None = None


class SearchHit(BaseModel):
    chunk_id: str
    score: float
    retrieval_score: float
    act_title: str
    article_number: str | None = None
    article_title: str | None = None
    chapter: str | None = None
    revision_index: int | None = None
    url: str | None = None
    text: str


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    latency_ms: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    qdrant: str
    collection: str
    points: int | None = None
    embedder_loaded: bool
    reranker_loaded: bool
    llm_base_url: str
    llm_model: str
