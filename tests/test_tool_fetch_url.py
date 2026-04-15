"""Tests for direct and registry-backed page fetching in src/tools/fetch_url.py."""
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, AsyncMock

from src.ai import AgentDeps
from src.source_registry import SourceRegistry
from src.tools.fetch_url import _extract, FetchedPage, MAX_BODY_CHARS, fetch_webpage

pytestmark = pytest.mark.unit


class TestFetchedPage:
    def test_source_url_equals_url(self):
        page = FetchedPage(url="https://example.com", title="Test", body="content")
        assert page.source_url == "https://example.com"

    def test_tag_defaults_empty(self):
        page = FetchedPage(url="https://example.com", title="Test", body="content")
        assert page.tag == ""


class TestExtract:
    def _bare_result(self, text="body text", title="Page Title"):
        result = MagicMock()
        result.text = text
        result.title = title
        return result

    def test_valid_html_returns_fetched_page(self):
        with patch("src.tools.fetch_url.trafilatura.bare_extraction",
                   return_value=self._bare_result("article content")):
            page = _extract("<html><body>content</body></html>", "https://example.com")
        assert page is not None
        assert isinstance(page, FetchedPage)

    def test_body_capped_at_max_body_chars(self):
        long_text = "x" * (MAX_BODY_CHARS + 500)
        with patch("src.tools.fetch_url.trafilatura.bare_extraction",
                   return_value=self._bare_result(long_text)):
            page = _extract("<html/>", "https://example.com")
        assert page is not None
        assert len(page.body) <= MAX_BODY_CHARS

    def test_trafilatura_returns_none_gives_none(self):
        with patch("src.tools.fetch_url.trafilatura.bare_extraction", return_value=None):
            page = _extract("<html/>", "https://example.com")
        assert page is None

    def test_empty_body_returns_none(self):
        with patch("src.tools.fetch_url.trafilatura.bare_extraction",
                   return_value=self._bare_result(text="   ")):
            page = _extract("<html/>", "https://example.com")
        assert page is None

    def test_none_text_returns_none(self):
        with patch("src.tools.fetch_url.trafilatura.bare_extraction",
                   return_value=self._bare_result(text=None)):
            page = _extract("<html/>", "https://example.com")
        assert page is None

    def test_url_passed_through(self):
        with patch("src.tools.fetch_url.trafilatura.bare_extraction",
                   return_value=self._bare_result("content")):
            page = _extract("<html/>", "https://specific.url/path")
        assert page.url == "https://specific.url/path"

    def test_title_extracted(self):
        with patch("src.tools.fetch_url.trafilatura.bare_extraction",
                   return_value=self._bare_result("content", "My Article Title")):
            page = _extract("<html/>", "https://example.com")
        assert page.title == "My Article Title"


class TestFetchWebpage:
    def _ctx(self):
        deps = AgentDeps(
            update_chat=AsyncMock(),
            user_input="Read this link",
            chat_id=1,
            source_registry=SourceRegistry(),
        )
        return SimpleNamespace(deps=deps)

    @pytest.mark.asyncio
    async def test_registers_fetched_page_in_source_registry(self):
        ctx = self._ctx()
        extracted = FetchedPage(
            url="https://example.com/article",
            title="Example Article",
            body="Example body text",
        )
        with patch("src.tools.fetch_url._fetch_page_from_url", AsyncMock(return_value=extracted)):
            page = await fetch_webpage(ctx, "https://example.com/article")

        assert page is not None
        assert page.tag == "[SOURCE_1]"
        source = ctx.deps.source_registry.lookup_by_key("SOURCE_1")
        assert source is not None
        assert source.url == "https://example.com/article"
        assert source.title == "Example Article"

    @pytest.mark.asyncio
    async def test_rejects_non_http_urls(self):
        ctx = self._ctx()
        page = await fetch_webpage(ctx, "javascript:alert(1)")
        assert page is None

    @pytest.mark.asyncio
    async def test_returns_none_when_fetch_fails(self):
        ctx = self._ctx()
        with patch("src.tools.fetch_url._fetch_page_from_url", AsyncMock(return_value=None)):
            page = await fetch_webpage(ctx, "https://example.com/missing")
        assert page is None
