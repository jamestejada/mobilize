import pytest
"""Tests for WikipediaSummary model in src/tools/wikipedia.py."""
from src.tools.wikipedia import WikipediaSummary

pytestmark = pytest.mark.unit


class TestWikipediaSummary:
    def test_source_url_equals_url(self):
        s = WikipediaSummary(title="Pandas", extract="A bear.", url="https://en.wikipedia.org/wiki/Giant_panda")
        assert s.source_url == "https://en.wikipedia.org/wiki/Giant_panda"

    def test_source_url_reflects_different_urls(self):
        s = WikipediaSummary(title="T", extract="E", url="https://en.wikipedia.org/wiki/Python")
        assert s.source_url == "https://en.wikipedia.org/wiki/Python"

    def test_fields_stored_correctly(self):
        s = WikipediaSummary(title="NumPy", extract="A library.", url="https://en.wikipedia.org/wiki/NumPy")
        assert s.title == "NumPy"
        assert s.extract == "A library."

    def test_tag_defaults_empty(self):
        s = WikipediaSummary(title="T", extract="E", url="https://en.wikipedia.org/wiki/T")
        assert s.tag == ""
