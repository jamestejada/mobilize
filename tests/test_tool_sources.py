"""Tests for get_registered_sources in src/tools/sources.py."""
import pytest
from unittest.mock import MagicMock

from src.tools.sources import get_registered_sources
from src.source_registry import SourceRegistry

pytestmark = pytest.mark.unit


def make_ctx(registry):
    ctx = MagicMock()
    ctx.deps.source_registry = registry
    return ctx


async def test_none_registry_returns_fallback():
    ctx = make_ctx(None)
    result = await get_registered_sources(ctx)
    assert result == "No sources have been collected yet."


async def test_empty_registry_returns_no_sources_message():
    ctx = make_ctx(SourceRegistry())
    result = await get_registered_sources(ctx)
    assert "No sources have been collected yet." in result


async def test_populated_registry_returns_formatted_list():
    registry = SourceRegistry()
    registry.register("https://example.com", source_name="Example")
    ctx = make_ctx(registry)
    result = await get_registered_sources(ctx)
    assert "SOURCE_1" in result
    assert "example.com" in result.lower() or "Example" in result


async def test_populated_registry_contains_sources_header():
    registry = SourceRegistry()
    registry.register("https://news.example.org", source_name="News")
    ctx = make_ctx(registry)
    result = await get_registered_sources(ctx)
    assert "Sources" in result or "SOURCE" in result


async def test_multiple_sources_all_appear():
    registry = SourceRegistry()
    registry.register("https://site-one.com", source_name="SiteOne")
    registry.register("https://site-two.com", source_name="SiteTwo")
    ctx = make_ctx(registry)
    result = await get_registered_sources(ctx)
    assert "SOURCE_1" in result
    assert "SOURCE_2" in result
