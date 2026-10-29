"""
Tests for the site-index loader (app.tools.site_index).

No network: fetches go through a fake ``_urlopen`` (or, for the redirect test, a
fake HTTPS handler inside a real urllib opener). The fixture
``fixtures/site_index_v1.json`` holds a few real chunks of the published index
plus synthetic edge chunks (markup in text, an injection line, an off-host url).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import urllib.error
import urllib.request
import urllib.response
from email.message import Message
from pathlib import Path

import pytest

from app.tools import site_index
from app.tools.site_index import (
    SiteIndexError,
    fetch_site_index,
    get_site_index,
    load_site_index,
    parse_site_index,
    refresh_site_index,
    run_site_index_refresh,
    set_site_index,
)
from app.tools.site_search import search_site

HOST = "www.markandrewmarquez.com"
INDEX_URL = f"https://{HOST}/site-index.json"
FIXTURE = Path(__file__).parent / "fixtures" / "site_index_v1.json"
FIXTURE_BYTES = FIXTURE.read_bytes()


def _doc(**overrides) -> dict:
    doc = json.loads(FIXTURE_BYTES)
    doc.update(overrides)
    return doc


def _parse(body: bytes, *, content_type: str = "application/json", final_url: str = INDEX_URL):
    return parse_site_index(
        body, content_type=content_type, final_url=final_url, allowed_netloc=HOST
    )


def _chunk(**overrides) -> dict:
    chunk = {
        "id": "page:0",
        "url": f"https://{HOST}/portfolio/page/",
        "title": "Page",
        "heading": "Overview",
        "text": "Terraform builds the canaries.",
    }
    chunk.update(overrides)
    return chunk


class FakeResponse:
    """The slice of an http.client.HTTPResponse that fetch_site_index uses."""

    def __init__(
        self,
        body: bytes = FIXTURE_BYTES,
        *,
        content_type: str | None = "application/json",
        content_length: int | None = None,
        url: str = INDEX_URL,
    ):
        self._body = body
        self._url = url
        self.headers = Message()
        if content_type is not None:
            self.headers["Content-Type"] = content_type
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        self.reads: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.reads.append(size)
        return self._body if size < 0 else self._body[:size]

    def geturl(self) -> str:
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


@pytest.fixture
def fake_urlopen(monkeypatch):
    """Replace the fetch seam; set ``.response`` or ``.error`` per test."""

    class _Fake:
        response: FakeResponse | None = FakeResponse()
        error: BaseException | None = None
        calls: list[tuple[urllib.request.Request, float]] = []

        def __call__(self, request, timeout):
            self.calls.append((request, timeout))
            if self.error is not None:
                raise self.error
            return self.response

    fake = _Fake()
    fake.calls = []
    monkeypatch.setattr(site_index, "_urlopen", fake)
    return fake


@pytest.fixture(autouse=True)
def reset_index():
    set_site_index(None)
    yield
    set_site_index(None)


# ---------------------------------------------------------------------------
# parse_site_index (contract v1)
# ---------------------------------------------------------------------------

class TestParse:
    def test_fixture_parses_and_skips_the_off_host_chunk(self):
        index = _parse(FIXTURE_BYTES)
        raw = json.loads(FIXTURE_BYTES)["chunks"]
        assert len(index.chunks) == len(raw) - 1
        assert index.skipped == 1
        assert all(c.url.startswith(f"https://{HOST}/") for c in index.chunks)
        assert index.generated_at == "2026-10-26T22:22:32Z"
        assert index.source_commit == "fixture"
        assert index.size_bytes == len(FIXTURE_BYTES)
        assert index.page_count == 5

    def test_chunk_text_is_kept_verbatim(self):
        # The loader stores published text as data; escaping happens on output.
        index = _parse(FIXTURE_BYTES)
        edge = next(c for c in index.chunks if c.id == "fixture-edge:0")
        assert "<script>alert(1)</script> & <b>bold</b>" in edge.text

    def test_the_built_search_finds_the_right_page(self):
        index = _parse(FIXTURE_BYTES)
        hit = search_site(index.search, "What does M365 Guardian's weekly security report check?",
                          min_score=0.0)
        assert hit is not None
        assert hit.url == f"https://{HOST}/portfolio/m365-guardian/"

    def test_cloudfront_homepage_html_is_rejected(self):
        # CloudFront answers an unknown path with the 200 homepage HTML.
        with pytest.raises(SiteIndexError, match="content-type 'text/html'"):
            _parse(b"<!doctype html><html></html>", content_type="text/html")

    @pytest.mark.parametrize("version", [2, "1", True, 1.0, None])
    def test_only_integer_version_1_is_accepted(self, version):
        with pytest.raises(SiteIndexError, match="version"):
            _parse(json.dumps(_doc(version=version)).encode())

    def test_missing_version_is_rejected(self):
        doc = _doc()
        del doc["version"]
        with pytest.raises(SiteIndexError, match="version None"):
            _parse(json.dumps(doc).encode())

    @pytest.mark.parametrize("body", [b"{not json", b'{"version": 1, "chunks": [', b"\xff\xfe\x00"])
    def test_invalid_json_is_rejected(self, body):
        with pytest.raises(SiteIndexError, match="invalid JSON"):
            _parse(body)

    def test_top_level_must_be_an_object(self):
        with pytest.raises(SiteIndexError, match="not a JSON object"):
            _parse(b"[1, 2]")

    @pytest.mark.parametrize("chunks", [None, {"id": "x"}, "chunks"])
    def test_chunks_must_be_a_list(self, chunks):
        with pytest.raises(SiteIndexError, match="chunks is not a list"):
            _parse(json.dumps(_doc(chunks=chunks)).encode())

    @pytest.mark.parametrize(
        "bad",
        [
            "not a dict",
            _chunk(text=""),
            _chunk(heading="   "),
            _chunk(title=7),
            {k: v for k, v in _chunk().items() if k != "id"},
            _chunk(url=f"http://{HOST}/portfolio/page/"),
            _chunk(url="https://evil.example/portfolio/page/"),
            _chunk(url=f"https://user@{HOST}/portfolio/page/"),
            _chunk(url=f"https://{HOST}/portfolio/page (1)/"),
            _chunk(url=f"https://{HOST}/portfolio/page/\"onclick"),
            _chunk(url="javascript:alert(1)"),
            _chunk(url="https://[bad"),
        ],
    )
    def test_invalid_chunks_are_skipped(self, bad):
        index = _parse(json.dumps(_doc(chunks=[_chunk(), bad])).encode())
        assert [c.id for c in index.chunks] == ["page:0"]
        assert index.skipped == 1

    def test_an_index_with_no_valid_chunk_is_rejected(self):
        body = json.dumps(_doc(chunks=[_chunk(text=""), _chunk(url="http://evil.example/")])).encode()
        with pytest.raises(SiteIndexError, match=r"no valid chunks \(of 2\)"):
            _parse(body)

    def test_non_string_metadata_is_ignored(self):
        index = _parse(json.dumps(_doc(generated_at=123, source_commit=None)).encode())
        assert index.generated_at is None
        assert index.source_commit is None
        assert index.metadata_dropped == 1  # 123 was present; a null wasn't

    @pytest.mark.parametrize(
        "value",
        [
            "2026-10-26T22:22:32Z\n2026-10-29 | INFO | forged log line",
            "abc\x1b[31m",
            "x" * 65,
        ],
    )
    def test_metadata_must_be_short_and_printable(self, value):
        index = _parse(json.dumps(_doc(generated_at=value, source_commit=value)).encode())
        assert index.generated_at is None
        assert index.source_commit is None
        assert index.metadata_dropped == 2

    def test_metadata_up_to_64_printable_characters_is_kept(self):
        commit = "a" * 64
        index = _parse(json.dumps(_doc(source_commit=commit)).encode())
        assert index.source_commit == commit
        assert index.generated_at == "2026-10-26T22:22:32Z"
        assert index.metadata_dropped == 0

    def test_a_long_version_is_cut_short_in_the_error(self):
        version = "2\n" + "x" * 5000
        with pytest.raises(SiteIndexError) as caught:
            _parse(json.dumps(_doc(version=version)).encode())
        message = str(caught.value)
        assert "\n" not in message
        assert len(message) < 80
        assert message.startswith("version '2\\nxxx")

    @pytest.mark.parametrize(
        "final_url",
        ["https://evil.example/site-index.json", f"http://{HOST}/site-index.json"],
    )
    def test_final_url_must_be_https_on_the_allowed_host(self, final_url):
        with pytest.raises(SiteIndexError, match="served from"):
            _parse(FIXTURE_BYTES, final_url=final_url)


# ---------------------------------------------------------------------------
# fetch_site_index / load_site_index (fake _urlopen)
# ---------------------------------------------------------------------------

class TestFetch:
    def test_fetch_sends_accept_json_with_the_timeout(self, fake_urlopen):
        final_url, content_type, body = fetch_site_index(INDEX_URL)
        request, timeout = fake_urlopen.calls[0]
        assert request.full_url == INDEX_URL
        assert request.get_header("Accept") == "application/json"
        assert timeout == 10.0
        assert (final_url, content_type, body) == (INDEX_URL, "application/json", FIXTURE_BYTES)

    def test_content_type_parameters_are_dropped(self, fake_urlopen):
        fake_urlopen.response = FakeResponse(content_type="Application/JSON; charset=utf-8")
        assert len(load_site_index(INDEX_URL).chunks) > 0

    def test_missing_content_type_is_rejected(self, fake_urlopen):
        # email.message.Message defaults to text/plain when the header is absent.
        fake_urlopen.response = FakeResponse(content_type=None)
        with pytest.raises(SiteIndexError, match="text/plain"):
            load_site_index(INDEX_URL)

    def test_oversize_content_length_is_rejected_before_reading(self, fake_urlopen):
        fake_urlopen.response = FakeResponse(content_length=2_000_001)
        with pytest.raises(SiteIndexError, match="Content-Length"):
            fetch_site_index(INDEX_URL)
        assert fake_urlopen.response.reads == []

    def test_oversize_body_is_rejected_without_a_header(self, fake_urlopen):
        fake_urlopen.response = FakeResponse(body=b"x" * 101)
        with pytest.raises(SiteIndexError, match="over 100 bytes"):
            fetch_site_index(INDEX_URL, max_bytes=100)
        assert fake_urlopen.response.reads == [101]

    def test_body_at_the_cap_is_accepted(self, fake_urlopen):
        fake_urlopen.response = FakeResponse(body=b"x" * 100, content_length=100)
        assert fetch_site_index(INDEX_URL, max_bytes=100)[2] == b"x" * 100

    def test_load_rejects_a_final_url_on_another_host(self, fake_urlopen):
        fake_urlopen.response = FakeResponse(url="https://evil.example/site-index.json")
        with pytest.raises(SiteIndexError, match="served from"):
            load_site_index(INDEX_URL)


class _FakeHTTPSRedirect(urllib.request.HTTPSHandler):
    """Answers every https request with a 301 to *location* (no network)."""

    def __init__(self, location: str):
        super().__init__()
        self.location = location

    def https_open(self, req):
        headers = Message()
        headers["Location"] = self.location
        response = urllib.response.addinfourl(io.BytesIO(b""), headers, req.full_url, 301)
        response.msg = "Moved Permanently"
        return response


class TestRedirects:
    @pytest.mark.parametrize(
        "location",
        [f"https://{HOST}/elsewhere/site-index.json", "https://evil.example/site-index.json"],
    )
    def test_every_redirect_is_refused(self, location):
        opener = urllib.request.build_opener(
            site_index._RefuseRedirects, _FakeHTTPSRedirect(location)
        )
        with pytest.raises(SiteIndexError, match="redirect refused: HTTP 301"):
            opener.open(INDEX_URL, timeout=1)

    def test_the_module_opener_refuses_redirects(self):
        handlers = site_index._OPENER.handlers
        redirect_handlers = [h for h in handlers if isinstance(h, urllib.request.HTTPRedirectHandler)]
        assert len(redirect_handlers) == 1
        assert isinstance(redirect_handlers[0], site_index._RefuseRedirects)

    def test_fetch_refuses_a_redirect_through_the_module_opener(self, monkeypatch):
        """The real _urlopen -> _OPENER path, with https answered by a 301 (no network)."""
        requests: list[str] = []

        def _answer_301(handler, req):
            requests.append(req.full_url)
            headers = Message()
            # Same host and https, so only the redirect refusal can stop it.
            headers["Location"] = f"https://{HOST}/moved/site-index.json"
            response = urllib.response.addinfourl(io.BytesIO(b""), headers, req.full_url, 301)
            response.msg = "Moved Permanently"
            return response

        monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", _answer_301)
        with pytest.raises(SiteIndexError, match="redirect refused: HTTP 301"):
            fetch_site_index(INDEX_URL)
        assert requests == [INDEX_URL]

    @pytest.mark.asyncio
    async def test_a_redirect_is_a_logged_load_failure(self, fake_urlopen, caplog):
        fake_urlopen.error = SiteIndexError("redirect refused: HTTP 302 to https://evil.example/")
        with caplog.at_level(logging.WARNING, logger="app.tools.site_index"):
            assert await refresh_site_index(INDEX_URL) is False
        assert get_site_index() is None
        assert "redirect refused" in caplog.text


# ---------------------------------------------------------------------------
# refresh_site_index / run_site_index_refresh
# ---------------------------------------------------------------------------

class TestRefresh:
    @pytest.mark.asyncio
    async def test_success_swaps_the_index_in(self, fake_urlopen, caplog):
        with caplog.at_level(logging.INFO, logger="app.tools.site_index"):
            assert await refresh_site_index(INDEX_URL) is True
        index = get_site_index()
        assert index is not None
        assert len(index.chunks) == 13
        assert "site_index loaded chunks=13 pages=5 skipped=1" in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            urllib.error.URLError(TimeoutError("timed out")),
            TimeoutError("timed out"),
            urllib.error.HTTPError(INDEX_URL, 503, "Service Unavailable", Message(), None),
            RuntimeError("anything else"),
        ],
    )
    async def test_failure_never_raises_and_keeps_the_previous_index(self, fake_urlopen, error):
        assert await refresh_site_index(INDEX_URL) is True
        previous = get_site_index()

        fake_urlopen.error = error
        assert await refresh_site_index(INDEX_URL) is False
        assert get_site_index() is previous

    @pytest.mark.asyncio
    async def test_an_invalid_file_keeps_the_previous_index(self, fake_urlopen, caplog):
        assert await refresh_site_index(INDEX_URL) is True
        previous = get_site_index()

        fake_urlopen.response = FakeResponse(b"<html></html>", content_type="text/html")
        with caplog.at_level(logging.WARNING, logger="app.tools.site_index"):
            assert await refresh_site_index(INDEX_URL) is False
        assert get_site_index() is previous
        assert "keeping the previous index" in caplog.text

    @pytest.mark.asyncio
    async def test_the_log_line_carries_no_dropped_metadata(self, fake_urlopen, caplog):
        forged = "2026-10-26T22:22:32Z\nCRITICAL forged log line"
        body = json.dumps(_doc(generated_at=forged)).encode()
        fake_urlopen.response = FakeResponse(body)
        with caplog.at_level(logging.INFO, logger="app.tools.site_index"):
            assert await refresh_site_index(INDEX_URL) is True
        assert "generated_at=None source_commit=fixture metadata_dropped=1" in caplog.text
        assert "forged log line" not in caplog.text

    @pytest.mark.asyncio
    async def test_failure_with_no_previous_index_leaves_none(self, fake_urlopen, caplog):
        fake_urlopen.error = TimeoutError("timed out")
        with caplog.at_level(logging.WARNING, logger="app.tools.site_index"):
            assert await refresh_site_index(INDEX_URL) is False
        assert get_site_index() is None
        assert "no index loaded" in caplog.text

    @pytest.mark.asyncio
    async def test_the_loop_reloads_every_interval_until_cancelled(self, monkeypatch):
        loads: list[str] = []
        sleeps: list[float] = []

        async def fake_refresh(url):
            loads.append(url)
            return True

        async def fake_sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) == 2:
                raise asyncio.CancelledError

        monkeypatch.setattr(site_index, "refresh_site_index", fake_refresh)
        monkeypatch.setattr(site_index.asyncio, "sleep", fake_sleep)
        with pytest.raises(asyncio.CancelledError):
            await run_site_index_refresh(INDEX_URL, 3600)
        assert loads == [INDEX_URL, INDEX_URL]
        assert sleeps == [3600, 3600]

    @pytest.mark.asyncio
    async def test_a_failed_load_is_retried_after_60_seconds(self, monkeypatch):
        outcomes = iter([False, True, False])
        sleeps: list[float] = []

        async def fake_refresh(url):
            return next(outcomes)

        async def fake_sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) == 3:
                raise asyncio.CancelledError

        monkeypatch.setattr(site_index, "refresh_site_index", fake_refresh)
        monkeypatch.setattr(site_index.asyncio, "sleep", fake_sleep)
        with pytest.raises(asyncio.CancelledError):
            await run_site_index_refresh(INDEX_URL, 3600)
        # fail -> 60 s, success -> the full interval, fail -> 60 s again
        assert sleeps == [60, 3600, 60]


# ---------------------------------------------------------------------------
# The default SITE_SEARCH_MIN_SCORE on the fixture
# ---------------------------------------------------------------------------

class TestDefaultThreshold:
    """
    The default min score (0.4) was calibrated on the full published index: of
    36 labeled questions, all 27 answerable ones picked their page and scored at
    least 0.437; the 9 off-site ones scored at most 0.355. This pins the same split
    on the fixture, so a ranking or default change that breaks it fails here.
    """

    ANSWERABLE = [
        ("Does the TARS chatbot hand leads off to Salesforce and Agentforce?",
         "salesforce-langgraph-ai-lead-bot"),
        ("What does M365 Guardian's weekly security report check?", "m365-guardian"),
        ("Can M365 Guardian run in Microsoft Teams?", "m365-guardian"),
        ("What did the KPMG virtual internship find about the ideal customer?", "kpmg-project"),
        ("How do I try api-medic?", "api-medic"),
    ]
    OFF_SITE = [
        "What's the weather in Paris today?",
        "Do you have an article about Kubernetes Helm charts?",
        "What did Mark write about React Native mobile apps?",
        "What is the capital of Australia?",
        "Has Mark written about blockchain smart contracts?",
        "Did Mark write about quantum computing?",
    ]

    @pytest.fixture
    def search(self):
        return _parse(FIXTURE_BYTES).search

    @pytest.fixture
    def min_score(self, monkeypatch) -> float:
        monkeypatch.delenv("SITE_SEARCH_MIN_SCORE", raising=False)
        from app.config import Settings

        return Settings(_env_file=None).site_search_min_score

    @pytest.mark.parametrize(("question", "slug"), ANSWERABLE)
    def test_answerable_question_hits_its_page(self, search, min_score, question, slug):
        hit = search_site(search, question, min_score=min_score)
        assert hit is not None, question
        assert hit.url == f"https://{HOST}/portfolio/{slug}/"

    @pytest.mark.parametrize("question", OFF_SITE)
    def test_off_site_question_is_a_miss(self, search, min_score, question):
        assert search_site(search, question, min_score=min_score) is None
