"""Tests for PolymarketMarket and PolymarketEvent in src/tools/polymarket.py."""
import pytest
from src.tools.polymarket import PolymarketMarket, PolymarketEvent

pytestmark = pytest.mark.unit


class TestPolymarketMarketFromApi:
    def test_outcomes_as_string_json_parsed(self):
        data = {"question": "Will X?", "outcomes": '["Yes","No"]', "outcomePrices": '["0.6","0.4"]'}
        market = PolymarketMarket.from_api(data)
        assert market.outcomes == ["Yes", "No"]

    def test_outcomes_already_list_unchanged(self):
        data = {"question": "Q", "outcomes": ["A", "B"], "outcomePrices": ["0.5", "0.5"]}
        market = PolymarketMarket.from_api(data)
        assert market.outcomes == ["A", "B"]

    def test_prices_as_string_json_parsed(self):
        data = {"question": "Q", "outcomes": "[]", "outcomePrices": '["0.7","0.3"]'}
        market = PolymarketMarket.from_api(data)
        assert market.prices == [0.7, 0.3]

    def test_prices_already_list_parsed_to_float(self):
        data = {"question": "Q", "outcomes": [], "outcomePrices": [0.55, 0.45]}
        market = PolymarketMarket.from_api(data)
        assert market.prices == [0.55, 0.45]

    def test_invalid_json_gives_empty_lists(self):
        data = {"question": "Q", "outcomes": "not-json", "outcomePrices": "bad"}
        market = PolymarketMarket.from_api(data)
        assert market.outcomes == []
        assert market.prices == []

    def test_volume_parsed(self):
        data = {"question": "Q", "outcomes": "[]", "outcomePrices": "[]", "volume": 12345.67}
        market = PolymarketMarket.from_api(data)
        assert market.volume == 12345.67

    def test_missing_volume_defaults_zero(self):
        data = {"question": "Q", "outcomes": "[]", "outcomePrices": "[]"}
        market = PolymarketMarket.from_api(data)
        assert market.volume == 0.0


class TestPolymarketEventFromApi:
    def test_source_url_with_slug(self):
        event = PolymarketEvent(title="T", slug="my-event-slug")
        assert event.source_url == "https://polymarket.com/event/my-event-slug"

    def test_source_url_empty_slug(self):
        event = PolymarketEvent(title="T", slug="")
        assert event.source_url == ""

    def test_from_api_title(self):
        data = {"title": "Election Outcome", "slug": "election-2026", "markets": []}
        event = PolymarketEvent.from_api(data)
        assert event.title == "Election Outcome"

    def test_from_api_slug(self):
        data = {"title": "T", "slug": "my-slug", "markets": []}
        event = PolymarketEvent.from_api(data)
        assert event.slug == "my-slug"

    def test_from_api_builds_markets(self):
        data = {
            "title": "T",
            "slug": "s",
            "markets": [
                {"question": "Q?", "outcomes": '["Y","N"]', "outcomePrices": '["0.6","0.4"]'}
            ]
        }
        event = PolymarketEvent.from_api(data)
        assert len(event.markets) == 1
        assert event.markets[0].question == "Q?"

    def test_from_api_no_markets(self):
        data = {"title": "T", "slug": "s"}
        event = PolymarketEvent.from_api(data)
        assert event.markets == []
