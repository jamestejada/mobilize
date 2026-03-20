"""Tests for GeocodingClient.location_to_zipcode in src/tools/geocoding.py."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from geopy.exc import GeocoderServiceError

from src.tools.geocoding import GeocodingClient

pytestmark = pytest.mark.unit


def _make_coordinates(lat=40.7128, lon=-74.0060):
    coords = MagicMock()
    coords.latitude = lat
    coords.longitude = lon
    return coords


def _make_address(postcode="10001"):
    addr = MagicMock()
    if postcode:
        addr.raw = {"properties": {"postcode": postcode}}
    else:
        addr.raw = {"properties": {}}
    return addr


async def _make_client(geocode_result, reverse_results, sleep_patch=True):
    """Helper: build a GeocodingClient with mocked geocode/reverse/sleep."""
    client = GeocodingClient.__new__(GeocodingClient)
    client.logger = MagicMock()
    client.geocode = AsyncMock(return_value=geocode_result)
    if isinstance(reverse_results, list):
        client.reverse = AsyncMock(side_effect=reverse_results)
    else:
        client.reverse = AsyncMock(return_value=reverse_results)
    return client


class TestLocationToZipcodeSuccess:
    async def test_returns_postcode_on_success(self):
        coords = _make_coordinates()
        addr = _make_address("10001")
        client = await _make_client(geocode_result=coords, reverse_results=addr)

        with patch("asyncio.sleep", new=AsyncMock()):
            result = await client.location_to_zipcode("New York, NY")

        assert result == "10001"

    async def test_geocode_called_with_location(self):
        coords = _make_coordinates()
        addr = _make_address("94102")
        client = await _make_client(geocode_result=coords, reverse_results=addr)

        with patch("asyncio.sleep", new=AsyncMock()):
            await client.location_to_zipcode("San Francisco, CA")

        client.geocode.assert_called_once_with("San Francisco, CA")


class TestLocationToZipcodeOffsetRetry:
    async def test_no_postcode_first_reverse_triggers_offset_retry(self):
        coords = _make_coordinates()
        # First reverse: no postcode; second (offset): has postcode
        addr_no_postcode = _make_address(None)
        addr_with_postcode = _make_address("60601")
        client = await _make_client(
            geocode_result=coords,
            reverse_results=[addr_no_postcode, addr_with_postcode]
        )

        with patch("asyncio.sleep", new=AsyncMock()):
            result = await client.location_to_zipcode("Chicago, IL")

        assert result == "60601"
        assert client.reverse.call_count == 2

    async def test_both_reverses_fail_returns_none(self):
        coords = _make_coordinates()
        addr_no_postcode = _make_address(None)
        client = await _make_client(
            geocode_result=coords,
            reverse_results=[addr_no_postcode, addr_no_postcode]
        )

        with patch("asyncio.sleep", new=AsyncMock()):
            result = await client.location_to_zipcode("Nowhere")

        assert result is None


class TestLocationToZipcodeNoCoordinates:
    async def test_geocode_returns_none_returns_none(self):
        client = await _make_client(geocode_result=None, reverse_results=None)

        with patch("asyncio.sleep", new=AsyncMock()):
            result = await client.location_to_zipcode("Unknown Place")

        assert result is None
        client.reverse.assert_not_called()


class TestLocationToZipcodeGeopyException:
    async def test_geocoder_service_error_returns_none_after_retries(self):
        client = GeocodingClient.__new__(GeocodingClient)
        client.logger = MagicMock()
        client.geocode = AsyncMock(side_effect=GeocoderServiceError("service down"))
        client.reverse = AsyncMock()

        with patch("asyncio.sleep", new=AsyncMock()):
            result = await client.location_to_zipcode("Anywhere")

        assert result is None
        # Tried MAX_RETRIES times
        assert client.geocode.call_count == GeocodingClient.MAX_RETRIES

    async def test_geocoder_error_then_success(self):
        coords = _make_coordinates()
        addr = _make_address("30301")
        client = GeocodingClient.__new__(GeocodingClient)
        client.logger = MagicMock()
        client.geocode = AsyncMock(
            side_effect=[GeocoderServiceError("retry"), coords]
        )
        client.reverse = AsyncMock(return_value=addr)

        with patch("asyncio.sleep", new=AsyncMock()):
            result = await client.location_to_zipcode("Atlanta, GA")

        assert result == "30301"


class TestGeocodingConstants:
    def test_max_retries(self):
        assert GeocodingClient.MAX_RETRIES == 3

    def test_retry_delay(self):
        assert GeocodingClient.RETRY_DELAY == 7
