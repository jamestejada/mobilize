"""Tests for src/tools/rss/__init__.py (list_rss_feeds, get_feed)."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import aiohttp

from src.tools.rss import list_rss_feeds, get_feed

pytestmark = pytest.mark.unit

UTC = timezone.utc

FEEDS_JSON = {
    "White House": {"url": "https://www.whitehouse.gov/feed/"},
    "State Department": {"url": "https://www.state.gov/rss/"},
}


# ---------------------------------------------------------------------------
# list_rss_feeds
# ---------------------------------------------------------------------------

class TestListRssFeeds:
    async def test_returns_string_with_all_feed_names(self):
        result = await list_rss_feeds(FEEDS_JSON, "US Government")
        assert "White House" in result
        assert "State Department" in result

    async def test_returns_string(self):
        result = await list_rss_feeds(FEEDS_JSON, "US Government")
        assert isinstance(result, str)

    async def test_description_in_output(self):
        result = await list_rss_feeds(FEEDS_JSON, "US Government")
        assert "US Government" in result

    async def test_single_feed(self):
        feeds = {"EPA": {"url": "https://epa.gov/feed"}}
        result = await list_rss_feeds(feeds, "Environmental")
        assert "EPA" in result

    async def test_empty_feeds(self):
        result = await list_rss_feeds({}, "Nothing")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# get_feed — unknown name
# ---------------------------------------------------------------------------

class TestGetFeedUnknownName:
    async def test_unknown_feed_returns_error_string(self):
        result = await get_feed("Nonexistent Feed", FEEDS_JSON)
        assert isinstance(result, str)
        assert "not found" in result.lower() or "Nonexistent Feed" in result

    async def test_empty_feeds_dict_returns_error(self):
        result = await get_feed("Any Feed", {})
        assert isinstance(result, str)
        assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# get_feed — successful fetch with current items
# ---------------------------------------------------------------------------

def _make_feedparser_dict(entries):
    """Minimal feedparser.FeedParserDict mock with entries."""
    fd = MagicMock()
    fd.entries = entries
    return fd


def _make_entry(title="Story", link="https://example.com/story", published=None):
    """Minimal feedparser entry dict with a recent date."""
    from datetime import timedelta
    if published is None:
        published = datetime.now(tz=UTC) - timedelta(hours=1)
    return {
        "title": title,
        "link": link,
        "summary": "A story summary.",
        "published": published.isoformat(),
        "id": link,
    }


class TestGetFeedSuccess:
    async def test_known_feed_returns_list(self):
        entry = _make_entry()
        feed_dict = _make_feedparser_dict([entry])

        mock_response = AsyncMock()
        mock_response.text = AsyncMock(return_value="<rss>...</rss>")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.rss.aiohttp.ClientSession", return_value=mock_session), \
             patch("src.tools.rss.feedparser.parse", return_value=feed_dict):
            result = await get_feed("White House", FEEDS_JSON)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].title == "Story"

    async def test_stale_items_filtered_out(self):
        from datetime import timedelta
        stale_published = datetime.now(tz=UTC) - timedelta(days=10)
        stale_entry = _make_entry(title="Old Story", published=stale_published)
        feed_dict = _make_feedparser_dict([stale_entry])

        mock_response = AsyncMock()
        mock_response.text = AsyncMock(return_value="<rss/>")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.rss.aiohttp.ClientSession", return_value=mock_session), \
             patch("src.tools.rss.feedparser.parse", return_value=feed_dict):
            result = await get_feed("White House", FEEDS_JSON)

        assert result == []

    async def test_mix_of_current_and_stale_only_current_returned(self):
        from datetime import timedelta
        recent = _make_entry(title="Recent", link="https://example.com/recent")
        stale_dt = datetime.now(tz=UTC) - timedelta(days=5)
        stale = _make_entry(title="Stale", link="https://example.com/stale", published=stale_dt)
        feed_dict = _make_feedparser_dict([recent, stale])

        mock_response = AsyncMock()
        mock_response.text = AsyncMock(return_value="<rss/>")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.rss.aiohttp.ClientSession", return_value=mock_session), \
             patch("src.tools.rss.feedparser.parse", return_value=feed_dict):
            result = await get_feed("White House", FEEDS_JSON)

        assert len(result) == 1
        assert result[0].title == "Recent"


# ---------------------------------------------------------------------------
# get_feed — fetch error
# ---------------------------------------------------------------------------

class TestGetFeedError:
    async def test_client_error_returns_empty_list(self):
        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=aiohttp.ClientError("connection refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.rss.aiohttp.ClientSession", return_value=mock_session):
            result = await get_feed("White House", FEEDS_JSON)

        # fetch_feed catches ClientError and returns None → get_feed returns []
        assert result == []

    async def test_none_feed_returns_empty_list(self):
        """fetch_feed returning None (e.g. after ClientError) yields empty list."""
        with patch("src.tools.rss.fetch_feed", new=AsyncMock(return_value=None)):
            result = await get_feed("White House", FEEDS_JSON)
        assert result == []
