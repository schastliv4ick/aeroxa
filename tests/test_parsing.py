"""Parser, chunker and prompt-helper tests.

These run without Qdrant, without the models and without network access — the parsing
layer is where corpus bugs come from, and it must be cheap to test.
"""

from __future__ import annotations

from datetime import date

import pytest

from ingest.chunker import build_chunks
from ingest.ips_client import RawDocument, Revision
from ingest.parser import parse_articles
from app.rag.prompts import build_context, detect_language

# Mirrors the real portal output: a Word-exported document nested inside the portal page.
SAMPLE_HTML = """
<html><head><title>Об обществах с ограниченной ответственностью</title></head>
<body class="doc_content_bg_default">
<div id="text_content">
<html><head><title>Complex</title>
<style>p { margin: 0; }</style>
<script>var textCompleted = 'True';</script>
</head>
<body>
<p>&nbsp;</p>
<p class="T">ФЕДЕРАЛЬНЫЙ ЗАКОН</p>
<p class="I">Принят Государственной Думой&nbsp;&nbsp;&nbsp;14&nbsp;января 1998&nbsp;года</p>
<p class="H">Глава I. Общие положения</p>
<p class="H">Статья 1. Отношения, регулируемые настоящим Федеральным законом</p>
<p>1. Настоящий Федеральный закон определяет правовое положение общества.</p>
<p>2. Особенности определяются федеральными законами.</p>
<p class="H">Глава II. Учреждение общества</p>
<p class="H">Статья 2. Устав общества</p>
<p>Устав общества должен содержать сведения о фирменном наименовании общества.</p>
<p>Статья 20 настоящего Федерального закона применяется с учетом изложенного.</p>
<p class="H">Статья 3. Резервный фонд</p>
<p>(Утратила силу)</p>
</body></html>
</div></body></html>
"""


@pytest.fixture
def articles():
    return parse_articles(SAMPLE_HTML)


def test_parses_every_article(articles):
    assert [article.number for article in articles] == ["1", "2", "3"]


def test_carries_chapter_breadcrumb(articles):
    assert articles[0].structure == "Глава I. Общие положения"
    assert articles[1].structure == "Глава II. Учреждение общества"


def test_preamble_is_discarded(articles):
    # Text before the first article heading is not attached to any article.
    assert "ФЕДЕРАЛЬНЫЙ ЗАКОН" not in "".join(article.text for article in articles)


def test_cross_reference_is_not_mistaken_for_a_heading(articles):
    """"Статья 20 настоящего..." has no trailing dot, so it stays body text."""
    assert "Статья 20 настоящего" in articles[1].text
    assert "20" not in [article.number for article in articles]


def test_repealed_article_is_detected(articles):
    assert articles[2].repealed is True
    assert articles[0].repealed is False


def test_original_casing_and_numbers_are_preserved(articles):
    assert "Настоящий Федеральный закон" in articles[0].text
    assert "1." in articles[0].text


def _document() -> RawDocument:
    return RawDocument(
        nd="102051516",
        title="Об обществах с ограниченной ответственностью",
        revision=Revision(index=57, date=date(2026, 8, 4), amending_act="№ 319-ФЗ"),
        html=SAMPLE_HTML,
        url="http://example.invalid/doc",
    )


def test_chunks_skip_repealed_articles(articles):
    chunks = build_chunks(
        _document(), articles, act_id="fz-14-ooo", act_kind="ФЗ", act_number="14-ФЗ", max_chars=1800
    )
    assert [chunk.payload["article_number"] for chunk in chunks] == ["1", "2"]


def test_chunk_payload_carries_revision_metadata(articles):
    chunk = build_chunks(
        _document(), articles, act_id="fz-14-ooo", act_kind="ФЗ", act_number="14-ФЗ", max_chars=1800
    )[0]
    assert chunk.payload["revision_index"] == 57
    assert chunk.payload["revision_date"] == "2026-08-04"
    assert chunk.payload["revision_source"] == "№ 319-ФЗ"
    assert chunk.payload["in_force"] is True
    assert chunk.chunk_id == "102051516:57:1:0"


def test_breadcrumbs_are_embedded_with_the_text(articles):
    chunk = build_chunks(
        _document(), articles, act_id="fz-14-ooo", act_kind="ФЗ", act_number="14-ФЗ", max_chars=1800
    )[0]
    assert "редакция 57" in chunk.embedding_text
    assert "Глава I" in chunk.embedding_text
    assert "Статья 1." in chunk.embedding_text
    # The stored text stays clean; breadcrumbs are a retrieval aid, not part of the norm.
    assert "Глава I" not in chunk.text


def test_long_article_splits_on_paragraph_boundaries(articles):
    long_article = articles[0]
    long_article.paragraphs = [f"Пункт {i}. " + "текст " * 40 for i in range(10)]
    chunks = build_chunks(
        _document(), [long_article], act_id="x", act_kind="ФЗ", act_number="1", max_chars=600
    )
    assert len(chunks) > 1
    assert all(chunk.payload["part_count"] == len(chunks) for chunk in chunks)
    # No paragraph is cut in half: the parts rejoin into exactly the original body.
    rejoined = "\n".join(chunk.text for chunk in chunks)
    assert rejoined == long_article.text.strip()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Как зарегистрировать ООО в России?", "ru"),
        ("如何在俄罗斯注册有限责任公司？", "zh"),
        ("How do I register an LLC in Russia?", "en"),
        # Latin inside a Russian or Chinese question must not flip the answer language.
        ("ООО registration", "ru"),
        ("Как зарегистрировать LLC в РФ?", "ru"),
        ("如何注册 LLC?", "zh"),
        ("", "ru"),
    ],
)
def test_language_detection(text, expected):
    assert detect_language(text) == expected


def test_context_blocks_are_numbered_for_citation():
    context = build_context(
        [{"breadcrumbs": "Акт · Статья 14", "text": "текст"}, {"breadcrumbs": "Акт · Статья 15", "text": "иной"}]
    )
    assert context.startswith("[1]")
    assert "[2]" in context
