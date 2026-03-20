"""Tests for RSSFeedItem in src/tools/rss/models.py."""
import pytest
from datetime import datetime, timedelta, timezone

from src.tools.rss.models import RSSFeedItem

pytestmark = pytest.mark.unit

UTC = timezone.utc


def _now_utc():
    return datetime.now(tz=UTC)


def _make_item(published: datetime | None = None, link: str = "https://example.com/story") -> RSSFeedItem:
    return RSSFeedItem(
        title="Test Title",
        link=link,
        summary="Summary text",
        source_name="Test Source",
        published=published,
    )


# ---------------------------------------------------------------------------
# _resolve_published_date
# ---------------------------------------------------------------------------

class TestResolvePublishedDate:
    def test_uses_published_key(self):
        entry = {"published": "2024-01-15T12:00:00+00:00"}
        result = RSSFeedItem._resolve_published_date(entry)
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_falls_back_to_updated(self):
        entry = {"updated": "2024-02-20T08:00:00+00:00"}
        result = RSSFeedItem._resolve_published_date(entry)
        assert result is not None
        assert result.year == 2024
        assert result.month == 2

    def test_falls_back_to_created(self):
        entry = {"created": "2024-03-10T00:00:00+00:00"}
        result = RSSFeedItem._resolve_published_date(entry)
        assert result is not None
        assert result.year == 2024
        assert result.month == 3

    def test_published_takes_precedence_over_updated(self):
        entry = {"published": "2024-01-01T00:00:00+00:00", "updated": "2024-06-01T00:00:00+00:00"}
        result = RSSFeedItem._resolve_published_date(entry)
        assert result.month == 1  # published wins

    def test_missing_all_keys_returns_none(self):
        result = RSSFeedItem._resolve_published_date({})
        assert result is None

    def test_timezone_naive_string_made_aware(self):
        entry = {"published": "2024-05-01 10:30:00"}
        result = RSSFeedItem._resolve_published_date(entry)
        assert result is not None
        assert result.tzinfo is not None

    def test_invalid_date_falls_through_to_none(self):
        entry = {"published": "not-a-date"}
        result = RSSFeedItem._resolve_published_date(entry)
        assert result is None


# ---------------------------------------------------------------------------
# age property
# ---------------------------------------------------------------------------

class TestAge:
    def test_age_recent_item(self):
        published = _now_utc() - timedelta(hours=3)
        item = _make_item(published=published)
        age = item.age
        assert age is not None
        assert timedelta(hours=2, minutes=55) < age < timedelta(hours=3, minutes=5)

    def test_age_none_when_no_published(self):
        item = _make_item(published=None)
        assert item.age is None

    def test_age_old_item(self):
        published = _now_utc() - timedelta(days=400)
        item = _make_item(published=published)
        assert item.age > timedelta(days=399)


# ---------------------------------------------------------------------------
# freshness property
# ---------------------------------------------------------------------------

class TestFreshness:
    def test_brand_new_is_one(self):
        published = _now_utc() - timedelta(minutes=1)
        item = _make_item(published=published)
        assert item.freshness > 0.99

    def test_48h_old_is_near_zero(self):
        published = _now_utc() - timedelta(hours=48)
        item = _make_item(published=published)
        assert item.freshness <= 0.0

    def test_24h_old_is_near_half(self):
        published = _now_utc() - timedelta(hours=24)
        item = _make_item(published=published)
        assert 0.4 < item.freshness < 0.6

    def test_no_published_returns_zero(self):
        item = _make_item(published=None)
        assert item.freshness == 0.0

    def test_freshness_never_negative(self):
        published = _now_utc() - timedelta(days=10)
        item = _make_item(published=published)
        assert item.freshness >= 0.0


# ---------------------------------------------------------------------------
# current property
# ---------------------------------------------------------------------------

class TestCurrent:
    def test_recent_item_is_current(self):
        published = _now_utc() - timedelta(hours=6)
        item = _make_item(published=published)
        assert item.current is True

    def test_old_item_is_not_current(self):
        published = _now_utc() - timedelta(days=5)
        item = _make_item(published=published)
        assert item.current is False

    def test_no_published_is_not_current(self):
        item = _make_item(published=None)
        assert item.current is False


# ---------------------------------------------------------------------------
# outdated property
# ---------------------------------------------------------------------------

class TestOutdated:
    def test_very_old_item_is_outdated(self):
        published = _now_utc() - timedelta(days=400)
        item = _make_item(published=published)
        assert item.outdated is True

    def test_recent_item_is_not_outdated(self):
        published = _now_utc() - timedelta(days=10)
        item = _make_item(published=published)
        assert item.outdated is False

    def test_no_published_is_outdated(self):
        item = _make_item(published=None)
        assert item.outdated is True


# ---------------------------------------------------------------------------
# _resolve_thumbnail_url
# ---------------------------------------------------------------------------

class TestResolveThumbnailUrl:
    def test_media_thumbnail(self):
        entry = {"media_thumbnail": [{"url": "https://img.example.com/thumb.jpg"}]}
        assert RSSFeedItem._resolve_thumbnail_url(entry) == "https://img.example.com/thumb.jpg"

    def test_media_content_image(self):
        entry = {"media_content": [{"medium": "image", "url": "https://img.example.com/photo.jpg"}]}
        assert RSSFeedItem._resolve_thumbnail_url(entry) == "https://img.example.com/photo.jpg"

    def test_media_content_non_image_skipped(self):
        entry = {"media_content": [{"medium": "video", "url": "https://example.com/vid.mp4"}]}
        assert RSSFeedItem._resolve_thumbnail_url(entry) is None

    def test_no_media_returns_none(self):
        assert RSSFeedItem._resolve_thumbnail_url({}) is None

    def test_media_thumbnail_takes_precedence(self):
        entry = {
            "media_thumbnail": [{"url": "https://thumb.example.com/t.jpg"}],
            "media_content": [{"medium": "image", "url": "https://content.example.com/c.jpg"}],
        }
        assert RSSFeedItem._resolve_thumbnail_url(entry) == "https://thumb.example.com/t.jpg"


# ---------------------------------------------------------------------------
# _resolve_tags
# ---------------------------------------------------------------------------

class TestResolveTags:
    def test_tags_extracted(self):
        entry = {"tags": [{"term": "politics"}, {"term": "economy"}]}
        assert RSSFeedItem._resolve_tags(entry) == ["politics", "economy"]

    def test_tags_without_term_skipped(self):
        entry = {"tags": [{"term": "valid"}, {"label": "no-term"}]}
        assert RSSFeedItem._resolve_tags(entry) == ["valid"]

    def test_no_tags_returns_none(self):
        assert RSSFeedItem._resolve_tags({}) is None

    def test_empty_tags_list_returns_none(self):
        assert RSSFeedItem._resolve_tags({"tags": []}) is None


# ---------------------------------------------------------------------------
# from_feedparser_entry
# ---------------------------------------------------------------------------

class TestFromFeedparserEntry:
    def test_all_fields_populated(self):
        entry = {
            "title": "Breaking News",
            "link": "https://news.example.com/story-1",
            "summary": "A big story happened today.",
            "published": "2024-06-15T09:00:00+00:00",
            "id": "https://news.example.com/story-1",
            "author": "Jane Doe",
        }
        item = RSSFeedItem.from_feedparser_entry(entry, source="Example News")
        assert item.title == "Breaking News"
        assert item.link == "https://news.example.com/story-1"
        assert item.summary == "A big story happened today."
        assert item.source_name == "Example News"
        assert item.published is not None
        assert item.id == "https://news.example.com/story-1"
        assert item.author == "Jane Doe"

    def test_source_url_equals_link(self):
        entry = {"title": "T", "link": "https://feeds.example.org/item/42", "summary": ""}
        item = RSSFeedItem.from_feedparser_entry(entry, source="Feed")
        assert item.source_url == "https://feeds.example.org/item/42"

    def test_missing_fields_default_to_empty(self):
        item = RSSFeedItem.from_feedparser_entry({}, source="Empty")
        assert item.title == ""
        assert item.link == ""
        assert item.summary == ""
        assert item.published is None

    def test_thumbnail_and_tags_populated(self):
        entry = {
            "title": "Media Story",
            "link": "https://example.com/media",
            "summary": "With media.",
            "media_thumbnail": [{"url": "https://img.example.com/t.jpg"}],
            "tags": [{"term": "sports"}],
        }
        item = RSSFeedItem.from_feedparser_entry(entry, source="Media Feed")
        assert item.thumbnail_url == "https://img.example.com/t.jpg"
        assert item.tags == ["sports"]
