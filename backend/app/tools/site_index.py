"""
Load and refresh the portfolio site index (site-index.json, contract v1).

The portfolio site publishes ``site-index.json``: ``version``, ``generated_at``,
``source_commit`` and ``chunks[]`` of ``{id, url, title, heading, text}``. This
module fetches it, checks it against contract v1, builds the BM25 search
(``app.tools.site_search``) and keeps the last good copy in memory.

Usage::

    from app.tools.site_index import get_site_index, run_site_index_refresh

    task = asyncio.create_task(run_site_index_refresh(url, 3600))  # at startup
    index = get_site_index()  # None until the first good load

Design notes:
- Stdlib only: ``urllib`` runs in a worker thread, so the event loop never blocks.
- A refresh never raises. A failed or invalid fetch is logged and the previous
  index (or none) stays, so startup and the refresh loop survive anything. The
  loop retries a failed load after 60 s rather than a full interval.
- Redirects are refused: the index must come from its own URL, https, on the
  configured host. Chunk urls must be on that host too.
- The content-type must be ``application/json``: CloudFront answers an unknown
  path with the 200 homepage HTML, which this check rejects.
- Chunk text is stored exactly as published. It may contain ``<``, ``>`` and
  ``&``; whoever renders it as HTML must escape it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import urllib.request
from dataclasses import dataclass
from http.client import HTTPMessage
from typing import IO, Any, NoReturn
from urllib.parse import urlsplit

from app.tools.site_search import SiteChunk, SiteSearch

logger = logging.getLogger(__name__)

INDEX_VERSION = 1
FETCH_TIMEOUT_SECONDS = 10.0
MAX_INDEX_BYTES = 2_000_000  # ~10x the published file (~170 KB)
RETRY_AFTER_FAILURE_SECONDS = 60.0
MAX_METADATA_CHARS = 64  # generated_at / source_commit longer than this are dropped
_ERROR_VALUE_CHARS = 40  # fetched values quoted in an error are cut to this
_WARNING_CHARS = 200  # the error text in the load-failure warning is cut to this

_CHUNK_FIELDS = ("id", "url", "title", "heading", "text")
# Characters that would break the markdown link a chunk url ends up in.
_URL_FORBIDDEN = re.compile(r"[\s()<>\"'`]")


class SiteIndexError(ValueError):
    """The site index could not be fetched, or it does not match contract v1."""


@dataclass(frozen=True)
class SiteIndex:
    """A validated site index plus its BM25 search."""

    search: SiteSearch
    generated_at: str | None
    source_commit: str | None
    size_bytes: int
    skipped: int  # chunks dropped as invalid at load
    metadata_dropped: int  # generated_at / source_commit present but not kept

    @property
    def chunks(self) -> tuple[SiteChunk, ...]:
        return self.search.chunks

    @property
    def page_count(self) -> int:
        return len({c.url.split("#", 1)[0] for c in self.chunks})


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Treat every 3xx as a load failure instead of following it."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> NoReturn:
        fp.close()
        raise SiteIndexError(f"redirect refused: HTTP {code} to {newurl}")


_OPENER = urllib.request.build_opener(_RefuseRedirects)


def _urlopen(request: urllib.request.Request, timeout: float) -> Any:
    """Open *request* with redirects refused (tests replace this seam)."""
    return _OPENER.open(request, timeout=timeout)


def fetch_site_index(
    url: str, *, timeout: float = FETCH_TIMEOUT_SECONDS, max_bytes: int = MAX_INDEX_BYTES
) -> tuple[str, str, bytes]:
    """
    GET *url* and return ``(final_url, content_type, body)``.

    ``content_type`` is lower-cased without parameters (``application/json``).
    Raises on an HTTP error, a timeout, a redirect, or a body over *max_bytes*.
    """
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with _urlopen(request, timeout) as response:
        length = response.headers.get("Content-Length")
        if length and length.isdigit() and int(length) > max_bytes:
            raise SiteIndexError(f"too large: Content-Length {length} > {max_bytes}")
        body = response.read(max_bytes + 1)  # cap the read even without a header
        if len(body) > max_bytes:
            raise SiteIndexError(f"too large: over {max_bytes} bytes")
        return response.geturl(), response.headers.get_content_type(), body


# ---------------------------------------------------------------------------
# Validate (contract v1)
# ---------------------------------------------------------------------------

def _parse_chunk(raw: Any, allowed_netloc: str) -> SiteChunk | None:
    """A valid chunk, or ``None`` (the caller skips it and counts it)."""
    if not isinstance(raw, dict):
        return None
    values: dict[str, str] = {}
    for field in _CHUNK_FIELDS:
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            return None
        values[field] = value
    url = values["url"]
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme != "https" or parts.netloc != allowed_netloc or _URL_FORBIDDEN.search(url):
        return None
    return SiteChunk(**values)


def _metadata(value: Any) -> str | None:
    """
    *value* if it is a short printable string, else ``None``. These values come
    from the fetched file and are logged as-is, so a newline or a huge value
    must not reach the log line.
    """
    if isinstance(value, str) and len(value) <= MAX_METADATA_CHARS and value.isprintable():
        return value
    return None


def _clip(value: Any) -> str:
    """``repr(value)``, cut to a short length, for fetched values in error text."""
    text = repr(value)
    return text if len(text) <= _ERROR_VALUE_CHARS else text[:_ERROR_VALUE_CHARS] + "..."


def _loggable(text: str, limit: int = _WARNING_CHARS) -> str:
    """
    *text* with non-printable characters escaped (``\\r``, ``\\x1b``) and cut to
    *limit* characters. Error text can carry server-controlled status, header and
    URL values, so a CR, an escape code or a 60 KB header must not reach the log.
    """
    escaped = "".join(ch if ch.isprintable() else ascii(ch)[1:-1] for ch in text[:limit])
    if len(text) > limit or len(escaped) > limit:
        return escaped[: limit - 3] + "..."
    return escaped


def parse_site_index(
    body: bytes, *, content_type: str, final_url: str, allowed_netloc: str
) -> SiteIndex:
    """
    Check *body* against contract v1 and build its search. Raises SiteIndexError.

    Invalid chunks are skipped (and counted); an index with no valid chunk is
    rejected as a whole.
    """
    if content_type != "application/json":
        raise SiteIndexError(f"content-type {content_type!r}, expected application/json")
    final = urlsplit(final_url)
    if final.scheme != "https" or final.netloc != allowed_netloc:
        raise SiteIndexError(f"served from {final_url!r}, expected https on {allowed_netloc}")

    try:
        doc = json.loads(body)
    except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError
        raise SiteIndexError(f"invalid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise SiteIndexError("top level is not a JSON object")

    version = doc.get("version")
    if type(version) is not int or version != INDEX_VERSION:  # JSON true == 1 in Python
        raise SiteIndexError(f"version {_clip(version)}, expected {INDEX_VERSION}")

    raw_chunks = doc.get("chunks")
    if not isinstance(raw_chunks, list):
        raise SiteIndexError("chunks is not a list")
    parsed = (_parse_chunk(raw, allowed_netloc) for raw in raw_chunks)
    chunks = [c for c in parsed if c is not None]
    if not chunks:
        raise SiteIndexError(f"no valid chunks (of {len(raw_chunks)})")

    raw_meta = {key: doc.get(key) for key in ("generated_at", "source_commit")}
    meta = {key: _metadata(value) for key, value in raw_meta.items()}
    return SiteIndex(
        search=SiteSearch(chunks),
        generated_at=meta["generated_at"],
        source_commit=meta["source_commit"],
        size_bytes=len(body),
        skipped=len(raw_chunks) - len(chunks),
        metadata_dropped=sum(
            1 for key, value in raw_meta.items() if value is not None and meta[key] is None
        ),
    )


def load_site_index(url: str) -> SiteIndex:
    """Fetch, validate and index *url*. Blocking: run it in a worker thread."""
    final_url, content_type, body = fetch_site_index(url)
    return parse_site_index(
        body,
        content_type=content_type,
        final_url=final_url,
        allowed_netloc=urlsplit(url).netloc,
    )


# ---------------------------------------------------------------------------
# In-memory copy + refresh
# ---------------------------------------------------------------------------

_index: SiteIndex | None = None


def get_site_index() -> SiteIndex | None:
    """The last good index, or ``None`` if none has loaded yet."""
    return _index


def set_site_index(index: SiteIndex | None) -> None:
    """Replace the in-memory index (the refresh uses it; so do tests)."""
    global _index  # noqa: PLW0603
    _index = index


async def refresh_site_index(url: str) -> bool:
    """
    Load *url* in a worker thread and swap it in. Returns whether it loaded.

    Never raises: on any failure the previous index stays and a warning is logged.
    """
    start = time.perf_counter()
    try:
        index = await asyncio.to_thread(load_site_index, url)
    except Exception as exc:
        logger.warning(
            "site_index load failed: %s (%s)",
            _loggable(str(exc)),
            "keeping the previous index" if _index is not None else "no index loaded",
        )
        return False

    set_site_index(index)
    logger.info(
        "site_index loaded chunks=%d pages=%d skipped=%d bytes=%d ms=%d "
        "generated_at=%s source_commit=%s metadata_dropped=%d",
        len(index.chunks),
        index.page_count,
        index.skipped,
        index.size_bytes,
        int((time.perf_counter() - start) * 1000),
        index.generated_at,
        index.source_commit,
        index.metadata_dropped,
    )
    return True


async def run_site_index_refresh(url: str, interval_seconds: float) -> None:
    """
    Load *url* now, then again after *interval_seconds*, until cancelled.

    After a failed load the next try comes after RETRY_AFTER_FAILURE_SECONDS
    instead, so one transient failure at a cold start doesn't leave the replica
    without an index for a whole interval.
    """
    while True:
        loaded = await refresh_site_index(url)
        await asyncio.sleep(interval_seconds if loaded else RETRY_AFTER_FAILURE_SECONDS)
