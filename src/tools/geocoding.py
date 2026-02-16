import asyncio
import logging
from typing import Optional

from geopy.geocoders import Photon
from geopy.adapters import AioHTTPAdapter
from geopy.exc import GeocoderServiceError

logger = logging.getLogger(__name__)


class GeocodingClient:
    """Client for geocoding operations using Photon (Komoot) with async support."""

    MAX_RETRIES = 3
    RETRY_DELAY = 7  # seconds

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.geolocator = Photon(
            adapter_factory=AioHTTPAdapter,
            timeout=10
        )
    
    async def geocode(self, location: str):
        return await self.geolocator.geocode(location)
    
    async def reverse(self, latitude: float, longitude: float):
        return await self.geolocator.reverse(
                (latitude, longitude),
                exactly_one=True
            )
    
    async def rate_limit_delay(
                self,
                attempt: int,
                e: Optional[GeocoderServiceError] = None
            ):
        delay = self.RETRY_DELAY * (2 ** attempt)
        if e:
            self.logger.error(f"Geocoding error: {e}")
        else:
            self.logger.warning(
                f"Geocoding rate limited, retrying in {delay}s "
                f"(attempt {attempt + 1}/{self.MAX_RETRIES})"
            )
        await asyncio.sleep(delay)
    
    async def location_to_zipcode(self, location: str) -> str | None:
        for attempt in range(self.MAX_RETRIES):
            try:
                coordinates = await self.geocode(location)
                if coordinates:
                    # Try reverse geocode, offset slightly if city-level result lacks postcode
                    for lat_offset in [0, 0.005]:
                        await asyncio.sleep(2)
                        address = await self.reverse(
                                    latitude=coordinates.latitude + lat_offset,
                                    longitude=coordinates.longitude
                                )
                        postcode = address.raw.get("properties", {}).get("postcode")
                        if postcode:
                            return postcode
                return None
            except GeocoderServiceError as e:
                if attempt < self.MAX_RETRIES - 1:
                    self.logger.warning(
                        f"Geocoding rate limited for '{location}', "
                        f"retrying in {self.RETRY_DELAY}s (attempt {attempt + 1}/{self.MAX_RETRIES})"
                    )
                    await asyncio.sleep(self.RETRY_DELAY)
                else:
                    self.logger.error(f"Geocoding failed for '{location}' after {self.MAX_RETRIES} attempts: {e}")
        return None


async def location_to_zipcode(location: str) -> str | None:
    """Converts a location string to a zipcode using geocoding."""
    client = GeocodingClient()
    async with client.geolocator:
        # NOTE: __aenter__ and __aexit__ are handled
        # by the geolocator context manager to ensure proper session management
        return await client.location_to_zipcode(location)