"""Tests for src/tools/mobilize/models.py."""
import pytest
from datetime import datetime, timezone

from src.tools.mobilize.models import (
    Coordinates, EventType, Location, Timeslot, Event, Sponsor
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sponsor(**overrides) -> dict:
    base = {
        "id": 1, "name": "Acme Org", "slug": "acme-org", "org_type": "C3",
        "is_coordinated": False, "is_independent": True, "is_nonelectoral": True,
        "is_primary_campaign": False, "state": "CA", "district": "",
        "candidate_name": "", "event_feed_url": "https://api.mobilize.us/v1/organizations/1/events",
        "created_date": 1700000000, "modified_date": 1700000001,
        "logo_url": "https://example.com/logo.png",
    }
    base.update(overrides)
    return base


def _make_location(**overrides) -> dict:
    base = {
        "venue": "City Hall", "address_lines": ["1 Main St"],
        "locality": "Springfield", "region": "IL",
        "country": "US", "postal_code": "62701",
        "location": {"latitude": 39.7817, "longitude": -89.6501},
    }
    base.update(overrides)
    return base


def _make_timeslot(start: int = 1700000000, end: int = 1700003600) -> dict:
    return {"id": 99, "start_date": start, "end_date": end, "is_full": False}


def _make_event(**overrides) -> dict:
    base = {
        "id": 42,
        "title": "Community Rally",
        "summary": "A community gathering.",
        "description": "Come join us for a rally!",
        "event_type": "RALLY",
        "timezone": "America/Chicago",
        "browser_url": "https://www.mobilize.us/acme/event/42/",
        "created_date": 1700000000,
        "modified_date": 1700000001,
        "visibility": "PUBLIC",
        "address_visibility": "PUBLIC",
        "accessibility_status": "not_accessible",
        "approval_status": "approved",
        "is_virtual": False,
        "created_by_volunteer_host": False,
        "sponsor": _make_sponsor(),
        "location": _make_location(),
        "timeslots": [_make_timeslot()],
        "tags": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# EventType enum
# ---------------------------------------------------------------------------

class TestEventType:
    def test_rally_accessible(self):
        assert EventType.RALLY.value == "RALLY"

    def test_town_hall_accessible(self):
        assert EventType.TOWN_HALL.value == "TOWN_HALL"

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            EventType("NOT_REAL")

    def test_canvass_accessible(self):
        assert EventType.CANVASS.value == "CANVASS"


# ---------------------------------------------------------------------------
# Timeslot — int timestamp → datetime
# ---------------------------------------------------------------------------

class TestTimeslot:
    def test_int_timestamp_coerced_to_datetime(self):
        ts = Timeslot(**_make_timeslot(start=1700000000, end=1700003600))
        assert isinstance(ts.start_date, datetime)
        assert isinstance(ts.end_date, datetime)

    def test_datetime_passthrough(self):
        dt = datetime(2024, 6, 1, 12, 0, 0)
        ts = Timeslot(id=1, start_date=dt, end_date=dt, is_full=False)
        assert ts.start_date == dt

    def test_is_full_preserved(self):
        ts = Timeslot(**_make_timeslot())
        assert ts.is_full is False


# ---------------------------------------------------------------------------
# Location — dict → Coordinates object
# ---------------------------------------------------------------------------

class TestLocation:
    def test_location_dict_coerced_to_coordinates(self):
        loc = Location(**_make_location())
        assert isinstance(loc.location, Coordinates)
        assert loc.location.latitude == pytest.approx(39.7817)
        assert loc.location.longitude == pytest.approx(-89.6501)

    def test_location_none_allowed(self):
        data = _make_location()
        data["location"] = None
        loc = Location(**data)
        assert loc.location is None

    def test_coordinates_passthrough(self):
        coords = Coordinates(latitude=40.0, longitude=-75.0)
        data = _make_location()
        data["location"] = coords
        loc = Location(**data)
        assert loc.location is coords


# ---------------------------------------------------------------------------
# Event.location_str
# ---------------------------------------------------------------------------

class TestEventLocationStr:
    def test_venue_city_state_format(self):
        event = Event(**_make_event())
        assert event.location_str == "City Hall, Springfield, IL"

    def test_missing_venue_graceful(self):
        loc = _make_location(venue="")
        event = Event(**_make_event(location=loc))
        # Should still produce a string without crashing
        assert isinstance(event.location_str, str)
        assert "Springfield" in event.location_str


# ---------------------------------------------------------------------------
# Event.coordinates
# ---------------------------------------------------------------------------

class TestEventCoordinates:
    def test_coordinates_formatted_as_lat_lng(self):
        event = Event(**_make_event())
        coords = event.coordinates
        assert "39.7817" in coords
        assert "-89.6501" in coords

    def test_no_location_returns_na(self):
        loc = _make_location()
        loc["location"] = None
        event = Event(**_make_event(location=loc))
        assert event.coordinates == "N/A"


# ---------------------------------------------------------------------------
# Event.llm_context
# ---------------------------------------------------------------------------

class TestEventLlmContext:
    def test_contains_title(self):
        event = Event(**_make_event())
        assert "Community Rally" in event.llm_context

    def test_contains_event_type(self):
        event = Event(**_make_event())
        assert "RALLY" in event.llm_context

    def test_contains_location(self):
        event = Event(**_make_event())
        assert "Springfield" in event.llm_context

    def test_contains_url(self):
        event = Event(**_make_event())
        assert "https://www.mobilize.us/acme/event/42/" in event.llm_context

    def test_multiline_string(self):
        event = Event(**_make_event())
        assert "\n" in event.llm_context


# ---------------------------------------------------------------------------
# Event.telegram_message
# ---------------------------------------------------------------------------

class TestEventTelegramMessage:
    def test_contains_title(self):
        event = Event(**_make_event())
        assert "Community Rally" in event.telegram_message

    def test_contains_location(self):
        event = Event(**_make_event())
        assert "Springfield" in event.telegram_message

    def test_contains_browser_url(self):
        event = Event(**_make_event())
        assert "https://www.mobilize.us/acme/event/42/" in event.telegram_message

    def test_contains_organizer(self):
        event = Event(**_make_event())
        assert "Acme Org" in event.telegram_message


# ---------------------------------------------------------------------------
# Event.source_url
# ---------------------------------------------------------------------------

class TestEventSourceUrl:
    def test_source_url_equals_browser_url(self):
        event = Event(**_make_event())
        assert event.source_url == "https://www.mobilize.us/acme/event/42/"


# ---------------------------------------------------------------------------
# Event coerce_event_type
# ---------------------------------------------------------------------------

class TestEventCoerceEventType:
    def test_string_coerced_to_enum(self):
        event = Event(**_make_event(event_type="TOWN_HALL"))
        assert event.event_type == EventType.TOWN_HALL

    def test_invalid_event_type_raises(self):
        with pytest.raises((ValueError, Exception)):
            Event(**_make_event(event_type="FAKE_TYPE"))
