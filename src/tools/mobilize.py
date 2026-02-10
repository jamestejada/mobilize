from typing import Optional
import logging

import aiohttp
from pydantic_ai import RunContext

from ..models import Event, EventType
from ..ai import AgentDeps
from ..settings import MobilizeEndpoints
from .geocoding import location_to_zipcode

logger = logging.getLogger(__name__)

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
        ) -> str:
    """Searches for protests events near a given location

        Args:
            location (int | str): This is the location give by the user. It can be:
                - A zipcode (int or str of 5 digits) e.g. "55401"
                - A city name (str) e.g. "Minneapolis, MN"
                - An address (str) e.g. "123 Main St, Minneapolis, MN"
                - A landmark (str) e.g. "Minneapolis City Hall"
            max_distance (Optional[int], optional): max distance in miles. Defaults to 75.

        Returns:
            str: Formatted string of events for LLM context.
    """
    logger.info(
        f"LLM Tool: get_protests_for_llm called with location={location}, max_distance={max_distance}"
        )
    await ctx.deps.update_chat(f"_Finding protest events {max_distance} miles around {location}_")
    events = await get_events(location=location, max_distance=max_distance)
    if not events:
        return "No upcoming protest events found in the area."
    return (
            f"Found {len(events)} protest events near {location}:\n\n"
            + "------".join([event.llm_context for event in events])
        )


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
    async with aiohttp.ClientSession() as session:
        async with session.get(
                    MobilizeEndpoints.EVENTS,
                    params=build_params(
                        zipcode=zipcode,
                        max_distance=max_distance
                        )
                ) as response:
            data = await response.json()
            events = data.get('data', [])
            if not events:
                return []
            event_count = data.get('count', 0)
            logger.info(f"Number of events found: {event_count}")
            return [
                Event(**event) for event in data.get('data', [])
            ]
