"""Unit tests: Sourceable protocol compliance for all model types."""
import pytest

pytestmark = pytest.mark.unit
from datetime import datetime, timezone

from src.source_registry import Sourceable

# Tool models
from src.tools.web_search import WebResult, NewsResult
from src.tools.bsky import BlueskyPost, BlueskyProfile, BlueskyTrendingTopic
from src.tools.reddit import RedditPost
from src.tools.wikipedia import WikipediaSummary
from src.tools.wayback import ArchivedPage
from src.tools.fetch_url import FetchedPage
from src.tools.polymarket import PolymarketEvent
from src.tools.fec import CandidateFinance, CommitteeFinance
from src.tools.congress import Bill
from src.tools.courtlistener import CourtCase
from src.tools.rss.models import RSSFeedItem
from src.tools.mobilize.models import Event, EventType


# ---------------------------------------------------------------------------
# Minimal valid instances of every Sourceable model
# ---------------------------------------------------------------------------

def _web_result():
    return WebResult(title="Title", href="https://example.com", body="Body")

def _news_result():
    return NewsResult(date="2024-01-01", title="News", body="Body", url="https://news.example.com")

def _bsky_post():
    return BlueskyPost(author_handle="alice.bsky.social", text="Hello!", post_id="abc123")

def _bsky_profile():
    return BlueskyProfile(handle="alice.bsky.social", display_name="Alice")

def _bsky_trending():
    return BlueskyTrendingTopic(topic="climate", link="https://bsky.app/hashtag/climate")

def _reddit_post():
    return RedditPost(title="Post", permalink="/r/test/comments/abc/post/")

def _wikipedia_summary():
    return WikipediaSummary(title="Python", extract="A language.", url="https://en.wikipedia.org/wiki/Python")

def _archived_page():
    return ArchivedPage(
        original_url="https://example.com",
        snapshot_url="https://web.archive.org/web/20240101/https://example.com",
        timestamp="20240101000000",
    )

def _fetched_page():
    return FetchedPage(url="https://example.com", title="Page Title", body="Page content.")

def _polymarket_event():
    return PolymarketEvent(
        id="evt-1",
        title="Will X happen?",
        slug="will-x-happen",
        description="A prediction market.",
        markets=[],
    )

def _candidate_finance():
    return CandidateFinance(
        candidate_id="P00000001",
        name="Jane Doe",
        party="DEM",
        state="CA",
        office="P",
        total_receipts=1000000.0,
        total_disbursements=900000.0,
        cash_on_hand=100000.0,
        coverage_end_date="2024-12-31",
    )

def _committee_finance():
    return CommitteeFinance(
        committee_id="C00000001",
        name="Friends of Jane",
        party="DEM",
        state="CA",
        committee_type="P",
        total_receipts=500000.0,
        total_disbursements=400000.0,
        cash_on_hand=100000.0,
        coverage_end_date="2024-12-31",
    )

def _bill():
    return Bill(
        bill_id="HR123",
        title="A bill to do things.",
        url="https://www.congress.gov/bill/119th-congress/house-bill/123",
        introduced_date="2025-01-15",
        sponsor="Rep. Smith",
        latest_action="Referred to committee.",
    )

def _court_case():
    return CourtCase(
        case_name="Smith v. Jones",
        url="https://www.courtlistener.com/opinion/1/smith-v-jones/",
        court="ca9",
        date_filed="2024-03-01",
        summary="A landmark decision.",
    )

def _rss_feed_item():
    return RSSFeedItem(
        title="Breaking News",
        link="https://feeds.example.com/item/1",
        summary="Something happened.",
        source_name="Example News",
        published=datetime.now(tz=timezone.utc),
    )

def _mobilize_event():
    sponsor = {
        "id": 1, "name": "Org", "slug": "org", "org_type": "C3",
        "is_coordinated": False, "is_independent": True, "is_nonelectoral": True,
        "is_primary_campaign": False, "state": "CA", "district": "",
        "candidate_name": "", "event_feed_url": "https://api.mobilize.us/v1/organizations/1/events",
        "created_date": 1700000000, "modified_date": 1700000001, "logo_url": "",
    }
    location = {
        "venue": "City Hall", "address_lines": ["1 Main St"],
        "locality": "Springfield", "region": "IL",
        "country": "US", "postal_code": "62701",
    }
    timeslot = {"id": 1, "start_date": 1700000000, "end_date": 1700003600, "is_full": False}
    return Event(
        id=1, title="Rally", summary="A rally.", description="Come join.",
        event_type="RALLY", timezone="America/Chicago",
        browser_url="https://www.mobilize.us/org/event/1/",
        created_date=1700000000, modified_date=1700000001,
        visibility="PUBLIC", address_visibility="PUBLIC",
        accessibility_status="not_accessible", approval_status="approved",
        is_virtual=False, created_by_volunteer_host=False,
        sponsor=sponsor, location=location, timeslots=[timeslot], tags=[],
    )


# ---------------------------------------------------------------------------
# All Sourceable instances
# ---------------------------------------------------------------------------

ALL_SOURCEABLE_INSTANCES = [
    ("WebResult",            _web_result()),
    ("NewsResult",           _news_result()),
    ("BlueskyPost",          _bsky_post()),
    ("BlueskyProfile",       _bsky_profile()),
    ("BlueskyTrendingTopic", _bsky_trending()),
    ("RedditPost",           _reddit_post()),
    ("WikipediaSummary",     _wikipedia_summary()),
    ("ArchivedPage",         _archived_page()),
    ("FetchedPage",          _fetched_page()),
    ("PolymarketEvent",      _polymarket_event()),
    ("CandidateFinance",     _candidate_finance()),
    ("CommitteeFinance",     _committee_finance()),
    ("Bill",                 _bill()),
    ("CourtCase",            _court_case()),
    ("RSSFeedItem",          _rss_feed_item()),
    ("Event (mobilize)",     _mobilize_event()),
]


@pytest.mark.parametrize("name,instance", ALL_SOURCEABLE_INSTANCES, ids=[n for n, _ in ALL_SOURCEABLE_INSTANCES])
def test_isinstance_sourceable(name, instance):
    assert isinstance(instance, Sourceable), f"{name} does not satisfy Sourceable protocol"


@pytest.mark.parametrize("name,instance", ALL_SOURCEABLE_INSTANCES, ids=[n for n, _ in ALL_SOURCEABLE_INSTANCES])
def test_source_url_is_string(name, instance):
    assert isinstance(instance.source_url, str), f"{name}.source_url is not a str"


@pytest.mark.parametrize("name,instance", ALL_SOURCEABLE_INSTANCES, ids=[n for n, _ in ALL_SOURCEABLE_INSTANCES])
def test_source_url_is_non_empty(name, instance):
    assert instance.source_url, f"{name}.source_url is empty"
