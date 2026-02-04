from geopy.geocoders import Nominatim
from geopy.adapters import AioHTTPAdapter


async def location_to_zipcode(location: str) -> str | None:
    async with Nominatim(
                user_agent="oculis-apertis-bot",
                adapter_factory=AioHTTPAdapter,
                timeout=10
            ) as geolocator:
        coordinates = await geolocator.geocode(location)
        if coordinates:
            address = await geolocator.reverse(
                        (coordinates.latitude, coordinates.longitude),
                        exactly_one=True
                    )
            return address.raw.get("address", {}).get("postcode")
    return None