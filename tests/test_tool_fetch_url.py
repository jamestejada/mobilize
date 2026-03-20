"""Tests for _extract() and FetchedPage in src/tools/fetch_url.py."""
import pytest
from unittest.mock import patch, MagicMock

from src.tools.fetch_url import _extract, FetchedPage, MAX_BODY_CHARS

pytestmark = pytest.mark.unit


class TestFetchedPage:
    def test_source_url_equals_url(self):
        page = FetchedPage(url="https://example.com", title="Test", body="content")
        assert page.source_url == "https://example.com"


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
