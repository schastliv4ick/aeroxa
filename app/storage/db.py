"""Append-only query log.

Not needed to answer a question — but it is the seed of the data flywheel from ТЗ slide 3:
every question, the fragments retrieved for it and the draft answer are exactly the rows a
lawyer review queue will later read from, and the pairs a future LoRA run will train on.
Recording them from day one costs nothing and cannot be backfilled.

Disabled when ``DATABASE_URL`` is unset; the service runs fine without it.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg

from app.config import get_settings

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    question        TEXT        NOT NULL,
    standalone      TEXT,
    language        TEXT        NOT NULL,
    answer          TEXT        NOT NULL,
    abstained       BOOLEAN     NOT NULL,
    escalated       BOOLEAN     NOT NULL,
    citations       JSONB       NOT NULL,
    model           TEXT,
    latency_ms      INTEGER     NOT NULL,
    -- Why no answer was generated: "low_relevance" or "generation_unavailable".
    -- The lawyer queue and any future fine-tuning set must not treat an outage as
    -- evidence that the corpus lacks a norm.
    reason          TEXT
);
ALTER TABLE query_log ADD COLUMN IF NOT EXISTS reason TEXT;
CREATE INDEX IF NOT EXISTS query_log_created_at_idx ON query_log (created_at DESC);
CREATE INDEX IF NOT EXISTS query_log_escalated_idx ON query_log (escalated) WHERE escalated;
"""


class QueryLog:
    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None
        self._dsn = get_settings().database_url

    @property
    def enabled(self) -> bool:
        return self._dsn is not None

    async def connect(self) -> None:
        if not self.enabled or self._pool is not None:
            return
        try:
            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
            async with self._pool.acquire() as connection:
                await connection.execute(SCHEMA)
            log.info("query log ready")
        except Exception:
            # Logging is best-effort: never block answering a client because the log is down.
            log.exception("query log unavailable; continuing without it")
            self._pool = None

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def record(self, **row: Any) -> None:
        if self._pool is None:
            return
        try:
            async with self._pool.acquire() as connection:
                await connection.execute(
                    """
                    INSERT INTO query_log
                        (question, standalone, language, answer, abstained, escalated,
                         citations, model, latency_ms, reason)
                    VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10)
                    """,
                    row["question"],
                    row.get("standalone"),
                    row["language"],
                    row["answer"],
                    row["abstained"],
                    row["escalated"],
                    json.dumps(row.get("citations", []), ensure_ascii=False),
                    row.get("model"),
                    row["latency_ms"],
                    row.get("reason"),
                )
        except Exception:
            log.exception("failed to write query log row")


query_log = QueryLog()
