"""Turn parsed articles into indexable chunks (ТЗ slide 7).

The base unit is one article. Articles longer than ``CHUNK_MAX_CHARS`` are split on
paragraph boundaries -- which in Russian legal text coincide with части and пункты, so the
split follows meaning rather than a character count. Every chunk repeats the breadcrumbs
(act, redaction, chapter, article heading) so a fragment retrieved on its own still tells
the reader and the model exactly what it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from ingest.parser import Article
from ingest.ips_client import RawDocument


@dataclass
class Chunk:
    chunk_id: str
    text: str  # article text only, without breadcrumbs
    breadcrumbs: str
    payload: dict = field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        """What actually gets embedded: breadcrumbs give the fragment its context."""
        return f"{self.breadcrumbs}\n{self.text}"


def build_chunks(
    document: RawDocument,
    articles: list[Article],
    *,
    act_id: str,
    act_kind: str,
    act_number: str,
    max_chars: int,
    include_repealed: bool = False,
) -> list[Chunk]:
    revision = document.revision
    revision_date = revision.date.isoformat() if revision.date else None
    indexed_at = datetime.now(timezone.utc).isoformat()
    act_label = f"{document.title} · редакция {revision.index}"
    if revision_date:
        act_label += f" от {revision_date}"

    chunks: list[Chunk] = []
    for article in articles:
        if article.repealed and not include_repealed:
            continue
        body = article.text.strip()
        if not body:
            continue

        heading = f"Статья {article.number}."
        if article.title:
            heading += f" {article.title}"

        parts = _split_paragraphs(body, max_chars)
        for part_index, part_text in enumerate(parts):
            breadcrumbs = "\n".join(
                filter(None, [act_label, article.structure, heading])
            )
            chunk_id = f"{document.nd}:{revision.index}:{article.number}:{part_index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=part_text,
                    breadcrumbs=breadcrumbs,
                    payload={
                        "chunk_id": chunk_id,
                        "act_id": act_id,
                        "act_title": document.title,
                        "act_kind": act_kind,
                        "act_number": act_number,
                        "nd": document.nd,
                        "revision_index": revision.index,
                        "revision_date": revision_date,
                        "revision_source": revision.amending_act,
                        "in_force": True,
                        "jurisdiction": "RU",
                        "chapter": article.structure,
                        "article_number": article.number,
                        "article_title": article.title,
                        "part_index": part_index,
                        "part_count": len(parts),
                        "text": part_text,
                        "breadcrumbs": breadcrumbs,
                        "url": document.url,
                        "indexed_at": indexed_at,
                    },
                )
            )
    return chunks


def _split_paragraphs(body: str, max_chars: int) -> list[str]:
    """Greedily pack whole paragraphs into parts of at most ``max_chars``.

    A single paragraph longer than the limit is emitted as its own part rather than being
    cut mid-sentence -- an over-long chunk retrieves worse, a truncated norm is wrong.
    """
    if len(body) <= max_chars:
        return [body]

    parts: list[str] = []
    buffer: list[str] = []
    size = 0
    for paragraph in body.split("\n"):
        addition = len(paragraph) + (1 if buffer else 0)
        if buffer and size + addition > max_chars:
            parts.append("\n".join(buffer))
            buffer, size = [paragraph], len(paragraph)
        else:
            buffer.append(paragraph)
            size += addition
    if buffer:
        parts.append("\n".join(buffer))
    return parts
