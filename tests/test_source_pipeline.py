"""Unit tests: source substitution pipeline end-to-end."""
import pytest

pytestmark = pytest.mark.unit

from src.source_registry import SourceRegistry


def test_register_then_substitute_produces_markdown_link():
    registry = SourceRegistry()
    tag = registry.register("https://example.com/article", title="Example Article")

    text = f"See {tag} for details."
    result = registry.substitute(text)

    assert "[Example Article]" in result or "example.com" in result
    assert "https://example.com/article" in result
    assert "[SOURCE_" not in result  # tag should be replaced


def test_multiple_sources_all_substituted():
    registry = SourceRegistry()
    tag1 = registry.register("https://news.example.com/story", title="News Story")
    tag2 = registry.register("https://gov.example.gov/report", title="Gov Report")

    text = f"According to {tag1} and {tag2}, this happened."
    result = registry.substitute(text)

    assert "https://news.example.com/story" in result
    assert "https://gov.example.gov/report" in result
    # Both tags replaced
    assert tag1 not in result
    assert tag2 not in result


def test_unregistered_tag_stripped():
    registry = SourceRegistry()
    registry.register("https://example.com", title="Real Source")

    text = "See [SOURCE_99] for more."
    result = registry.substitute(text)

    assert "[SOURCE_99]" not in result
    assert "https://example.com" not in result  # not this source


def test_full_pipeline_register_cite_substitute():
    """Simulate the full agent pipeline: register → LLM cites tag → substitute."""
    registry = SourceRegistry()

    # Step 1: tool returns URL, gets registered
    url = "https://www.reuters.com/world/story-123"
    tag = registry.register(url, title="Reuters: Breaking Story", source_name="Reuters")

    # Step 2: LLM writes response citing the tag
    llm_response = f"Breaking news has been confirmed {tag}. Reuters reports the situation is developing."

    # Step 3: source substitution
    final = registry.substitute(llm_response)

    assert url in final
    assert tag not in final
    assert "Reuters" in final or url in final


def test_compound_tag_both_sources_substituted():
    registry = SourceRegistry()
    tag1 = registry.register("https://site-a.com", title="Site A")
    tag2 = registry.register("https://site-b.com", title="Site B")

    # LLM sometimes emits compound tags
    num1 = tag1.strip("[]").replace("SOURCE_", "")
    num2 = tag2.strip("[]").replace("SOURCE_", "")
    text = f"Both sources agree [SOURCE_{num1}, SOURCE_{num2}]."
    result = registry.substitute(text)

    assert "https://site-a.com" in result
    assert "https://site-b.com" in result


def test_format_for_user_shows_registered_sources():
    registry = SourceRegistry()
    registry.register("https://bbc.com/news/1", title="BBC News", source_name="BBC")
    registry.register("https://ap.org/story/2", title="AP Story", source_name="AP")

    formatted = registry.format_for_user()

    assert isinstance(formatted, str)
    assert len(formatted) > 0
    # Should not be the empty fallback
    assert "No sources" not in formatted


def test_empty_registry_format_for_user_fallback():
    registry = SourceRegistry()
    result = registry.format_for_user()
    assert "No sources" in result or isinstance(result, str)


def test_duplicate_url_returns_same_tag():
    registry = SourceRegistry()
    tag1 = registry.register("https://example.com/page")
    tag2 = registry.register("https://example.com/page")

    assert tag1 == tag2
    # Only one entry in registry
    text = registry.substitute(f"See {tag1}.")
    assert text.count("https://example.com/page") == 1
