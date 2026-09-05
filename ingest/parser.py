"""Parse ИПС document HTML into structured articles.

The document body is Word-exported HTML: a flat sequence of ``<p>`` elements where
headings carry ``class="H"`` and everything else is running text. Structure is recovered
from the text of each paragraph rather than from its class, because class usage is not
consistent across acts (КоАП marks chapters differently from ФЗ-14).

Normalisation is deliberately minimal (ТЗ slide 6): entities are unescaped, markup is
dropped and runs of whitespace are collapsed. Case is preserved and no characters are
stripped -- lowercasing or punctuation stripping breaks article numbers and degrades
transformer retrieval.
"""

from __future__ import annotations

import html as html_module
import re
from dataclasses import dataclass, field

# The page nests the Word document's own <body> inside the portal's <body>.
_BODY_SPLIT_RE = re.compile(r"<body[^>]*>", re.I)
_SCRIPT_STYLE_RE = re.compile(r"(?s)<(script|style)[^>]*>.*?</\1>", re.I)
_PARAGRAPH_RE = re.compile(r"(?s)<p\b([^>]*)>(.*?)</p>", re.I)
_CLASS_RE = re.compile(r'class="([^"]*)"', re.I)
_TAG_RE = re.compile(r"<[^>]+>")

# Trailing dot after the number is required: it distinguishes the heading
# "Статья 12. Устав общества" from a cross-reference like "Статья 20 настоящего закона".
_ARTICLE_RE = re.compile(r"^Статья\s+(?P<number>\d+(?:[.\-]\d+)*)\.\s*(?P<title>.*)$")
_STRUCTURE_RE = re.compile(
    r"^(?:Раздел|Подраздел|Глава|Параграф|§)\s+[IVXLCDM\d]", re.IGNORECASE
)
_REPEALED_RE = re.compile(r"^\(?\s*Утратил[аи]?\s+силу", re.IGNORECASE)


@dataclass
class Article:
    number: str
    title: str
    structure: str | None  # nearest Раздел/Глава heading, e.g. "Глава IV. Управление…"
    paragraphs: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs)

    @property
    def repealed(self) -> bool:
        """Articles struck from the code keep a placeholder but carry no norm."""
        body = self.text.strip()
        if _REPEALED_RE.match(body):
            return True
        return bool(_REPEALED_RE.match(self.title.strip()))


def parse_articles(html: str) -> list[Article]:
    """Extract articles in document order.

    Text appearing before the first article heading (preamble, adopting formula) is
    discarded: it carries no citable norm and would dilute retrieval.
    """
    articles: list[Article] = []
    current: Article | None = None
    structure: str | None = None

    for text, css_class in _iter_paragraphs(html):
        if _STRUCTURE_RE.match(text):
            structure = text
            continue

        match = _ARTICLE_RE.match(text)
        # A heading-classed paragraph is trusted even without the trailing dot.
        if match is None and css_class == "H":
            match = re.match(r"^Статья\s+(?P<number>\d+(?:[.\-]\d+)*)\.?\s*(?P<title>.*)$", text)
        if match:
            current = Article(
                number=match.group("number"),
                title=match.group("title").strip(),
                structure=structure,
            )
            articles.append(current)
            continue

        if current is not None:
            current.paragraphs.append(text)

    return _deduplicate(articles)


def _iter_paragraphs(html: str):
    """Yield ``(text, css_class)`` for each non-empty paragraph of the document body."""
    body = _BODY_SPLIT_RE.split(html)[-1]
    body = _SCRIPT_STYLE_RE.sub("", body)

    for attrs, inner in _PARAGRAPH_RE.findall(body):
        text = _to_text(inner)
        if not text:
            continue
        class_match = _CLASS_RE.search(attrs)
        yield text, (class_match.group(1).strip() if class_match else "")


def _to_text(fragment: str) -> str:
    text = _TAG_RE.sub(" ", fragment)
    text = html_module.unescape(text)
    # NBSP is used for layout throughout these documents; treat it as ordinary space.
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _deduplicate(articles: list[Article]) -> list[Article]:
    """Keep the richest occurrence of each article number.

    Some acts repeat article headings in a table of contents. The real article is the one
    with the most text, so collapsing on number and keeping the longest body is safe.
    """
    best: dict[str, Article] = {}
    order: list[str] = []
    for article in articles:
        existing = best.get(article.number)
        if existing is None:
            best[article.number] = article
            order.append(article.number)
        elif len(article.text) > len(existing.text):
            best[article.number] = article
    return [best[number] for number in order]
