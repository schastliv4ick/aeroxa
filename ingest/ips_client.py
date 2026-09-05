"""Client for ИПС «Законодательство России» (pravo.gov.ru/proxy/ips).

Endpoint behaviour, verified against the live service:

* ``?docbody=&nd={nd}`` returns the document card. It embeds the full redaction list as
  ``<option value="N">N - от DD.MM.YYYY № X-ФЗ (изм.)</option>``. The highest option value
  is the currently in-force edition.
* ``?doc_itself=&nd={nd}&page=1&rdk={rdk}&fulltext=1`` returns the consolidated text of
  redaction ``rdk``. ``fulltext=1`` is mandatory: without it long documents are silently
  truncated (the page's own JS reports ``textCompleted = 'False'``) and the ``page``
  parameter does *not* paginate -- it returns the same truncated prefix.
* Everything is windows-1251.

The service's full-text search endpoint currently returns HTTP 500, so documents cannot be
discovered by query. :func:`scan_titles` brute-forces the ``nd`` id space instead; ids are
then pinned in ``ingest/acts.yaml``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

from app.config import get_settings

log = logging.getLogger(__name__)

_USER_AGENT = "Mozilla/5.0 (compatible; AeroxaLegalRAG/0.1)"
_ENCODING = "cp1251"

_OPTION_RE = re.compile(r'<option[^>]*value="(-?\d+)"[^>]*>(.*?)</option>', re.S)
_REVISION_RE = re.compile(
    r"^\s*(?P<idx>\d+)\s*-\s*от\s*(?P<day>\d{2})\.(?P<month>\d{2})\.(?P<year>\d{4})"
    r"\s*(?P<number>№\s*\S+)?"
)
_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.S)
_TEXT_COMPLETED_RE = re.compile(r"textCompleted\s*=\s*'(\w+)'")


@dataclass(frozen=True)
class Revision:
    """One edition of an act."""

    index: int
    date: date | None
    amending_act: str | None  # e.g. "№ 319-ФЗ"; absent for some entries

    @property
    def is_original(self) -> bool:
        return self.index == 0


@dataclass(frozen=True)
class RawDocument:
    nd: str
    title: str
    revision: Revision
    html: str
    url: str


class IpsError(RuntimeError):
    pass


class IpsClient:
    """Synchronous ИПС client. Ingestion is a batch job, so sync keeps it simple."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        settings = get_settings()
        self._base_url = settings.ips_base_url
        self._cache_dir = Path(settings.raw_cache_dir)
        self._client = client or httpx.Client(
            timeout=settings.ips_timeout_s,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "IpsClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- low level ---------------------------------------------------------

    def _get(self, params: str) -> str:
        url = f"{self._base_url}?{params}"
        response = self._client.get(url)
        response.raise_for_status()
        # The service declares windows-1251 inconsistently across endpoints, so decode
        # from bytes explicitly rather than trusting httpx's charset detection.
        return response.content.decode(_ENCODING, errors="replace")

    # -- public ------------------------------------------------------------

    def fetch_title(self, nd: str) -> str | None:
        """Cheap identity probe used by :func:`scan_titles`. ``None`` if no such doc."""
        try:
            html = self._get(f"doc_itself=&nd={nd}&page=1&rdk=0")
        except httpx.HTTPError:
            return None
        match = _TITLE_RE.search(html)
        title = _clean_title(match.group(1)) if match else ""
        return title or None

    def fetch_revisions(self, nd: str) -> list[Revision]:
        """All editions of an act, ascending. Index 0 is the text as originally signed."""
        html = self._get(f"docbody=&nd={nd}")
        revisions: dict[int, Revision] = {}
        for value, label in _OPTION_RE.findall(html):
            index = int(value)
            if index < 0:
                continue
            if index == 0:
                revisions[0] = Revision(index=0, date=None, amending_act=None)
                continue
            match = _REVISION_RE.match(_strip_tags(label))
            if not match:
                continue
            revisions[index] = Revision(
                index=index,
                date=date(
                    int(match.group("year")), int(match.group("month")), int(match.group("day"))
                ),
                amending_act=_normalise_space(match.group("number") or "") or None,
            )
        if not revisions:
            raise IpsError(f"no redaction list found for nd={nd}")
        return [revisions[i] for i in sorted(revisions)]

    def current_revision(self, nd: str) -> Revision:
        """The in-force edition: the highest redaction index the service publishes."""
        return self.fetch_revisions(nd)[-1]

    def fetch_document(self, nd: str, revision: Revision, *, use_cache: bool = True) -> RawDocument:
        """Consolidated text of one edition, cached on disk by (nd, revision)."""
        cache_path = self._cache_dir / f"{nd}_rdk{revision.index}.html"
        html: str | None = None
        if use_cache and cache_path.is_file():
            html = cache_path.read_text(encoding="utf-8")
            log.info("nd=%s rdk=%s served from cache", nd, revision.index)

        if html is None:
            params = f"doc_itself=&nd={nd}&page=1&rdk={revision.index}&fulltext=1"
            html = self._get(params)
            completed = _TEXT_COMPLETED_RE.search(html)
            if completed and completed.group(1).lower() != "true":
                # fulltext=1 should always complete the document. If it did not, the text
                # is truncated and indexing it would silently lose articles.
                raise IpsError(
                    f"nd={nd} rdk={revision.index}: service returned an incomplete document"
                )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(html, encoding="utf-8")

        title_match = _TITLE_RE.search(html)
        title = _clean_title(title_match.group(1)) if title_match else ""
        if not title:
            raise IpsError(f"nd={nd} rdk={revision.index}: document has no title")

        return RawDocument(
            nd=nd,
            title=title,
            revision=revision,
            html=html,
            url=f"{self._base_url}?doc_itself=&nd={nd}&rdk={revision.index}",
        )


def scan_titles(nd_from: int, nd_to: int, pattern: str, *, concurrency: int = 8) -> list[tuple[str, str]]:
    """Brute-force the ``nd`` id space for documents whose title matches ``pattern``.

    Only needed because the portal's search endpoint is broken. Ids are roughly
    chronological by publication date, so a range around an act's publication narrows the
    scan to a few hundred requests.
    """
    import concurrent.futures

    regex = re.compile(pattern, re.IGNORECASE)
    hits: list[tuple[str, str]] = []

    # One client for the whole scan: httpx.Client is thread-safe and pools connections,
    # so a thousand-id sweep reuses a handful of sockets instead of opening one per id.
    http = httpx.Client(
        timeout=30.0,
        headers={"User-Agent": _USER_AGENT},
        follow_redirects=True,
        limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
    )
    client = IpsClient(client=http)

    def probe(nd: int) -> tuple[str, str] | None:
        title = client.fetch_title(str(nd))
        if title and regex.search(title):
            return str(nd), title
        return None

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            for result in pool.map(probe, range(nd_from, nd_to + 1)):
                if result:
                    hits.append(result)
    finally:
        client.close()
    return hits


def _strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html)


def _normalise_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_title(raw: str) -> str:
    return _normalise_space(_unescape(_strip_tags(raw)))


def _unescape(text: str) -> str:
    import html as html_module

    return html_module.unescape(text.replace("&nbsp;", " "))
