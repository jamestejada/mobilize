"""Tests for src/tools/mobilize/__init__.py."""
import pytest
from unittest.mock import AsyncMock, patch

from src.tools.mobilize import build_params, get_zipcode_from_location
from src.tools.mobilize.models import EventType

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# build_params
# ---------------------------------------------------------------------------

class TestBuildParams:
    def test_zipcode_in_params(self):
        params = build_params("10001", 50)
        assert params["zipcode"] == "10001"

    def test_max_distance_in_params(self):
        params = build_params("10001", 50)
        assert params["max_dist"] == 50

    def test_default_max_distance(self):
        params = build_params("10001")
        assert params["max_dist"] == 75

    def test_event_types_present(self):
        params = build_params("10001")
        assert "event_types" in params
        assert isinstance(params["event_types"], list)
        assert len(params["event_types"]) > 0

    def test_rally_in_event_types(self):
        params = build_params("10001")
        assert EventType.RALLY.value in params["event_types"]

    def test_town_hall_in_event_types(self):
        params = build_params("10001")
        assert EventType.TOWN_HALL.value in params["event_types"]

    def test_integer_zipcode_accepted(self):
        params = build_params(10001)
        assert params["zipcode"] == 10001

    def test_timeslot_start_set(self):
        params = build_params("10001")
        assert params.get("timeslot_start") == "gte_now"


# ---------------------------------------------------------------------------
# get_zipcode_from_location
# ---------------------------------------------------------------------------

class TestGetZipcodeFromLocation:
    async def test_five_digit_string_returned_as_is(self):
        result = await get_zipcode_from_location("10001")
        assert result == "10001"

    async def test_five_digit_int_returned_as_string(self):
        result = await get_zipcode_from_location(10001)
        assert result == "10001"

    async def test_non_zipcode_string_calls_geocoder(self):
        with patch(
            "src.tools.mobilize.location_to_zipcode",
            new=AsyncMock(return_value="55401")
        ) as mock_geo:
            result = await get_zipcode_from_location("Minneapolis, MN")
        mock_geo.assert_called_once_with("Minneapolis, MN")
        assert result == "55401"

    async def test_geocoder_returns_none_propagated(self):
        with patch(
            "src.tools.mobilize.location_to_zipcode",
            new=AsyncMock(return_value=None)
        ):
            result = await get_zipcode_from_location("Nowhere Land")
        assert result is None

    async def test_four_digit_string_calls_geocoder(self):
        """4-digit number is not a zipcode — should go through geocoder."""
        with patch(
            "src.tools.mobilize.location_to_zipcode",
            new=AsyncMock(return_value="90210")
        ) as mock_geo:
            result = await get_zipcode_from_location("1234")
        mock_geo.assert_called_once()

    async def test_city_address_calls_geocoder(self):
        with patch(
            "src.tools.mobilize.location_to_zipcode",
            new=AsyncMock(return_value="62701")
        ) as mock_geo:
            result = await get_zipcode_from_location("123 Main St, Springfield, IL")
        mock_geo.assert_called_once()
        assert result == "62701"
