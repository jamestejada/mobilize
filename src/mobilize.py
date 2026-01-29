from typing import Optional
import aiohttp
import logging

from .models import Event, EventType
from .settings import MobilizeEndpoints

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


async def get_events(
            zipcode: int | str,
            max_distance: Optional[int] = 75
        ) -> dict:
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
