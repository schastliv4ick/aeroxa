"""Ingest an act: fetch the in-force redaction, parse, chunk, embed, upsert.

Re-running this for an act whose redaction has advanced is the update path: the new
edition's chunks are inserted and every older chunk of the same act is flipped to
``in_force = false``, so retrieval can only ever see the current text while past answers
stay auditable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.config import get_settings
from app.rag.embedder import get_embedder
from app.storage.qdrant_store import get_store
from ingest.chunker import build_chunks
from ingest.ips_client import IpsClient, IpsError
from ingest.parser import parse_articles

log = logging.getLogger(__name__)

ACTS_FILE = Path(__file__).with_name("acts.yaml")


@dataclass
class ActSpec:
    id: str
    nd: str | None
    kind: str
    number: str
    title_match: str
    scan_hint: tuple[int, int] | None = None


@dataclass
class IngestReport:
    act_id: str
    nd: str
    revision_index: int
    revision_date: str | None
    articles: int
    chunks: int
    skipped: bool = False
    reason: str | None = None


def load_acts(path: Path = ACTS_FILE) -> list[ActSpec]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    specs = []
    for entry in data["acts"]:
        hint = entry.get("scan_hint")
        specs.append(
            ActSpec(
                id=entry["id"],
                nd=str(entry["nd"]) if entry.get("nd") else None,
                kind=entry.get("kind", ""),
                number=entry.get("number", ""),
                title_match=entry["title_match"],
                scan_hint=(int(hint[0]), int(hint[1])) if hint else None,
            )
        )
    return specs


def ingest_act(
    spec: ActSpec,
    client: IpsClient,
    *,
    use_cache: bool = True,
    dry_run: bool = False,
) -> IngestReport:
    if not spec.nd:
        return IngestReport(
            act_id=spec.id,
            nd="",
            revision_index=-1,
            revision_date=None,
            articles=0,
            chunks=0,
            skipped=True,
            reason="no nd id resolved yet — run `ingest.cli find`",
        )

    settings = get_settings()
    revision = client.current_revision(spec.nd)
    log.info(
        "%s: nd=%s current redaction %s (%s)",
        spec.id,
        spec.nd,
        revision.index,
        revision.date or "original",
    )

    document = client.fetch_document(spec.nd, revision, use_cache=use_cache)

    # Identity guard: never index a document whose title does not match the registry.
    if not re.search(spec.title_match, document.title, re.IGNORECASE):
        raise IpsError(
            f"{spec.id}: nd={spec.nd} returned {document.title!r}, "
            f"which does not match {spec.title_match!r}"
        )

    articles = parse_articles(document.html)
    if not articles:
        raise IpsError(f"{spec.id}: no articles parsed from nd={spec.nd}")

    chunks = build_chunks(
        document,
        articles,
        act_id=spec.id,
        act_kind=spec.kind,
        act_number=spec.number,
        max_chars=settings.chunk_max_chars,
    )

    report = IngestReport(
        act_id=spec.id,
        nd=spec.nd,
        revision_index=revision.index,
        revision_date=revision.date.isoformat() if revision.date else None,
        articles=len(articles),
        chunks=len(chunks),
    )
    if dry_run:
        report.skipped = True
        report.reason = "dry run"
        return report

    store = get_store()
    store.ensure_collection()
    embedder = get_embedder()

    batch_size = settings.embed_batch_size
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings = embedder.encode([chunk.embedding_text for chunk in batch])
        store.upsert(
            [chunk.chunk_id for chunk in batch],
            embeddings,
            [chunk.payload for chunk in batch],
        )
        log.info("%s: indexed %d/%d chunks", spec.id, min(start + batch_size, len(chunks)), len(chunks))

    # Only after the new edition is fully in place: retire the previous one.
    store.retire_act(spec.nd, keep_revision=revision.index)
    return report
