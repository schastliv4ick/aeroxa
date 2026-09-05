"""Runtime configuration. Every knob is an environment variable; see .env.example."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- service -------------------------------------------------------------
    app_name: str = "aeroxa-legal-rag"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    # When set, every request must carry a matching X-API-Key header.
    api_key: str | None = None

    # --- Qdrant --------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "ru_law"

    # --- Postgres (query log; optional -- unset disables logging) -------------
    database_url: str | None = None

    # --- models --------------------------------------------------------------
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_dim: int = 1024
    # BGE-M3 truncates at 8192; law articles never need that much and it costs latency.
    embedding_max_length: int = 1024
    use_fp16: bool = False  # fp16 is a GPU optimisation; on CPU it is slower.
    models_eager_load: bool = False  # load on startup instead of first request

    # --- retrieval -----------------------------------------------------------
    retrieve_top_k: int = 40
    rerank_enabled: bool = True
    rerank_top_n: int = 6
    # A long article is split into several chunks, and they all score alike on a query
    # about that article -- left alone they fill every context slot and starve the answer
    # of related norms. Cap how many parts of one article may reach the prompt.
    max_parts_per_article: int = 2
    # Sigmoid-normalised cross-encoder score below which we refuse to answer.
    #
    # Measured against ФЗ-14 (docs/SPEC.md §4): across ru/zh/en, every question whose top
    # hit was the correct article scored >= 0.119, and every off-topic question scored
    # <= 0.007. 0.05 sits between those bands for all three languages.
    #
    # It is deliberately low. The scale is strongly language-dependent -- a Chinese
    # question against a Russian article scores 4-8x lower than a Russian one even when
    # retrieval is perfect -- so a threshold picked from Russian examples alone (0.30 was
    # the first guess here) refuses correct Chinese answers outright and clips English
    # ones. Off-topic questions score near zero, so raising the threshold buys no safety
    # against them; it only buys false abstentions.
    abstain_threshold: float = 0.05
    # Per-language overrides, for when a labelled evaluation set gives real per-language
    # operating points. Empty by default: the measured data does not yet justify splitting
    # them, only lowering the common value.
    abstain_threshold_by_language: dict[str, float] = {}

    # --- LLM (OpenAI-compatible: vLLM, Ollama, OpenRouter, ...) --------------
    llm_base_url: str = "http://localhost:8001/v1"
    llm_api_key: str = "not-needed"
    llm_model: str = "qwen/qwen3-32b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1200
    llm_timeout_s: float = 120.0
    # Condensing a follow-up into a standalone question is a second, cheap LLM call.
    condense_enabled: bool = True
    condense_max_history_turns: int = 6

    # --- chunking ------------------------------------------------------------
    chunk_max_chars: int = 1800

    # --- ingestion -----------------------------------------------------------
    ips_base_url: str = "http://pravo.gov.ru/proxy/ips/"
    ips_timeout_s: float = 180.0
    ips_max_concurrency: int = 8
    raw_cache_dir: str = "data/raw"
    embed_batch_size: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
