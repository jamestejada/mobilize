"""Tests for WebResult, NewsResult models in src/tools/web_search.py."""
import pytest
from src.tools.web_search import WebResult, NewsResult

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# WebResult
# ---------------------------------------------------------------------------

class TestWebResult:
    def _make(self, **kwargs):
        defaults = {"title": "Test Title", "href": "https://example.com/page", "body": "Some body text"}
        defaults.update(kwargs)
        return WebResult(**defaults)

    def test_source_url_equals_href(self):
        r = self._make(href="https://example.com/page")
        assert r.source_url == "https://example.com/page"

    def test_source_url_different_urls(self):
        r = self._make(href="https://news.org/article/123")
        assert r.source_url == "https://news.org/article/123"

    def test_str_contains_title(self):
        r = self._make(title="Breaking News")
        assert "Breaking News" in str(r)

    def test_str_contains_href(self):
        r = self._make(href="https://example.com")
        assert "https://example.com" in str(r)

    def test_str_contains_body(self):
        r = self._make(body="Article body content here")
        assert "Article body content here" in str(r)

    def test_tag_defaults_empty(self):
        r = self._make()
        assert r.tag == ""

    def test_tag_can_be_set(self):
        r = self._make(tag="[SOURCE_1]")
        assert r.tag == "[SOURCE_1]"

    def test_str_multiline(self):
        r = self._make(title="T", href="https://x.com", body="B")
        lines = str(r).split("\n")
        assert len(lines) >= 3


# ---------------------------------------------------------------------------
# NewsResult
# ---------------------------------------------------------------------------

class TestNewsResult:
    def _make(self, **kwargs):
        defaults = {
            "date": "2026-03-19",
            "title": "News Headline",
            "body": "News body text",
            "url": "https://news.example.com/story",
            "source": "Example News",
        }
        defaults.update(kwargs)
        return NewsResult(**defaults)

    def test_source_url_equals_url(self):
        r = self._make(url="https://news.example.com/story")
        assert r.source_url == "https://news.example.com/story"

    def test_str_contains_title(self):
        r = self._make(title="Major Event Today")
        assert "Major Event Today" in str(r)

    def test_str_contains_source(self):
        r = self._make(source="Reuters")
        assert "Reuters" in str(r)

    def test_str_contains_date(self):
        r = self._make(date="2026-01-15")
        assert "2026-01-15" in str(r)

    def test_str_contains_body(self):
        r = self._make(body="Detailed news coverage")
        assert "Detailed news coverage" in str(r)

    def test_optional_image_defaults_none(self):
        r = self._make()
        del r  # just instantiate without image
        r2 = NewsResult(date="2026-01-01", title="T", body="B", url="https://x.com")
        assert r2.image is None

    def test_optional_source_defaults_none(self):
        r = NewsResult(date="2026-01-01", title="T", body="B", url="https://x.com")
        assert r.source is None

    def test_str_omits_image_line_when_none(self):
        r = self._make(image=None)
        assert "Image:" not in str(r)

    def test_str_includes_image_line_when_set(self):
        r = self._make(image="https://img.example.com/photo.jpg")
        assert "https://img.example.com/photo.jpg" in str(r)

    def test_tag_defaults_empty(self):
        r = self._make()
        assert r.tag == ""
