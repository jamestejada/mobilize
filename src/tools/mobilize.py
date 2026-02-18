from typing import Optional, List
import logging

from pydantic_ai import RunContext

from ..models import Event, EventType
from ..ai import AgentDeps
from ..settings import MobilizeEndpoints
from .geocoding import location_to_zipcode
from .http_client import AsyncHTTPClient

logger = logging.getLogger(__name__)


class MobilizeClient(AsyncHTTPClient):
    BASE_URL = str(MobilizeEndpoints.API_ROOT)

def build_params(zipcode: int | str, max_distance: Optional[int] = 75) -> dict:
    return {
        "zipcode": zipcode,
        "event_types": [
            EventType.RALLY.value,
            EventType.SOLIDARITY_EVENT.value,
            EventType.VISIBILITY_EVENT.value,
            EventType.TOWN_HALL.value,
        ],
        "timeslot_start": "gte_now",
        "max_dist": max_distance,
    }


async def get_protests_for_llm(
            ctx: RunContext[AgentDeps],
            location: int | str,
            max_distance: Optional[int] = 75
        ) -> List[Event]:
    """Searches for protests events near a given location

        Args:
            location (int | str): This is the location give by the user. It can be:
                - A zipcode (int or str of 5 digits) e.g. "55401"
                - A city name (str) e.g. "Minneapolis, MN"
                - An address (str) e.g. "123 Main St, Minneapolis, MN"
                - A landmark (str) e.g. "Minneapolis City Hall"
            max_distance (Optional[int], optional): max distance in miles. Defaults to 75.

        Returns:
            List[Event]: List of protest events near the location.
    """
    logger.info(
        f"LLM Tool: get_protests_for_llm called with location={location}, max_distance={max_distance}"
        )
    await ctx.deps.update_chat(f"_Finding protest events {max_distance} miles around {location}_")
    events = await get_events(location=location, max_distance=max_distance)
    if not events:
        logger.info(f"No upcoming protest events found near {location}")
        return []
    return events


async def get_zipcode_from_location(
            location: int | str
        ) -> str:
    """Converts a location string to a zipcode."""
    if str(location).isdigit() and len(str(location)) == 5:
        return str(location)
    zipcode = await location_to_zipcode(location)
    return zipcode


async def get_events(
            location: int | str,
            max_distance: Optional[int] = 75
        ) -> list[Event]:
    """Fetches events from Mobilize API based on location and max distance."""
    zipcode = await get_zipcode_from_location(location)
    if zipcode is None:
        logger.warning(f"Could not resolve zipcode for '{location}', skipping API call")
        return []
    api_parameters = build_params(zipcode=zipcode, max_distance=max_distance)
    async with MobilizeClient() as client:
        data = await client.request(endpoint="events", params=api_parameters)
    if data is None:
        return []
    events = data.get("data") or []
    return [Event(**event) for event in events]