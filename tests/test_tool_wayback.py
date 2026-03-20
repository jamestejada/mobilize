import pytest
"""Tests for ArchivedPage model in src/tools/wayback.py."""
from src.tools.wayback import ArchivedPage

pytestmark = pytest.mark.unit


class TestArchivedPage:
    def test_source_url_equals_snapshot_url(self):
        page = ArchivedPage(
            original_url="https://example.com/deleted",
            snapshot_url="https://web.archive.org/web/20230101000000/https://example.com/deleted",
            timestamp="20230101000000",
        )
        assert page.source_url == "https://web.archive.org/web/20230101000000/https://example.com/deleted"

    def test_source_url_not_original_url(self):
        page = ArchivedPage(
            original_url="https://original.com",
            snapshot_url="https://web.archive.org/web/20230601/https://original.com",
            timestamp="20230601",
        )
        assert page.source_url != page.original_url
        assert "web.archive.org" in page.source_url

    def test_title_defaults_empty(self):
        page = ArchivedPage(
            original_url="https://x.com", snapshot_url="https://web.archive.org/web/1/https://x.com",
            timestamp="20230101"
        )
        assert page.title == ""

    def test_body_defaults_empty(self):
        page = ArchivedPage(
            original_url="https://x.com", snapshot_url="https://s", timestamp="20230101"
        )
        assert page.body == ""
