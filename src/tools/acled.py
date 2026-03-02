import time
import aiohttp
from datetime import date, timedelta
from typing import List, Literal, Optional
import logging

from pydantic import BaseModel
from pydantic_ai import RunContext

from ..ai import AgentDeps
from ..source_registry import SourceRegistry
from ..settings import ACLEDCredentials
from .http_client import AsyncHTTPClient

logger = logging.getLogger(__name__)

TOKEN_URL = "https://acleddata.com/oauth/token"

AcledEventType = Literal[
    "Battles",
    "Explosions/Remote violence",
    "Violence against civilians",
    "Protests",
    "Riots",
    "Strategic developments",
]


class AcledAuth:
    """Fetches and caches an OAuth 2.0 Bearer token for the ACLED API."""
    _token: Optional[str] = None
    _expires_at: float = 0.0

    @classmethod
    async def get_token(cls) -> Optional[str]:
        if cls._token and time.time() < cls._expires_at:
            return cls._token
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(TOKEN_URL, data={
                    "grant_type": "password",
                    "client_id": "acled",
                    "username": ACLEDCredentials.EMAIL,
                    "password": ACLEDCredentials.PASSWORD,
                }) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        logger.error(f"ACLED auth failed (HTTP {resp.status}): {data}")
                        return None
        except aiohttp.ClientError as e:
            logger.error(f"ACLED auth request failed: {e}")
            return None
        cls._token = data.get("access_token")
        if not cls._token:
            logger.error(f"ACLED auth returned no access_token: {data}")
            return None
        cls._expires_at = time.time() + data.get("expires_in", 3600) - 60
        logger.info("ACLED auth token refreshed")
        return cls._token


class AcledClient(AsyncHTTPClient):
    BASE_URL = "https://acleddata.com/api/acled"

    async def __aenter__(self):
        await super().__aenter__()
        self._token = await AcledAuth.get_token()
        return self

    async def http_request(self, url: str, params: dict) -> tuple[int, dict]:
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        params = {"_format": "json", **params}
        async with self.session.get(url, params=params, headers=headers) as response:
            data = await response.json()
            return response.status, data


class CastClient(AcledClient):
    """Inherits OAuth auth from AcledClient; targets the CAST forecast endpoint."""
    BASE_URL = "https://acleddata.com/api/cast"


# --- Models ---

class AcledEvent(BaseModel):
    event_date: str
    event_type: str
    sub_event_type: str
    actor1: str
    actor2: str = ""
    country: str
    location: str
    latitude: float = 0.0
    longitude: float = 0.0
    fatalities: int = 0
    notes: str = ""
    tag: str = ""

    @property
    def title(self) -> str:
        return "ACLED Conflict Data"

    @property
    def source_url(self) -> str:
        return "https://acleddata.com/data/"

    def __str__(self) -> str:
        parts = [
            f"Date: {self.event_date}",
            f"Type: {self.event_type} / {self.sub_event_type}",
            f"Actors: {self.actor1}" + (f" vs {self.actor2}" if self.actor2 else ""),
            f"Location: {self.location}, {self.country}",
            f"Fatalities: {self.fatalities}",
        ]
        if self.notes:
            parts.append(f"Notes: {self.notes[:300]}")
        return "\n".join(parts)

    @classmethod
    def from_api(cls, data: dict) -> "AcledEvent":
        return cls(
            event_date=data.get("event_date", ""),
            event_type=data.get("event_type", ""),
            sub_event_type=data.get("sub_event_type", ""),
            actor1=data.get("actor1", ""),
            actor2=data.get("actor2", ""),
            country=data.get("country", ""),
            location=data.get("location", ""),
            latitude=float(data.get("latitude") or 0),
            longitude=float(data.get("longitude") or 0),
            fatalities=int(data.get("fatalities") or 0),
            notes=data.get("notes", ""),
        )


class AcledForecast(BaseModel):
    country: str
    admin1: str = ""
    month: str = ""
    year: int = 0
    total_forecast: Optional[float] = None
    battles_forecast: Optional[float] = None
    erv_forecast: Optional[float] = None    # explosions/remote violence
    vac_forecast: Optional[float] = None    # violence against civilians
    total_observed: Optional[float] = None
    battles_observed: Optional[float] = None
    erv_observed: Optional[float] = None
    vac_observed: Optional[float] = None
    tag: str = ""

    @property
    def title(self) -> str:
        return f"ACLED Forecast: {self.country}"

    @property
    def source_url(self) -> str:
        return "https://acleddata.com/forecasting/"

    def __str__(self) -> str:
        lines = [f"Forecast: {self.country} {self.month}/{self.year}"]
        if self.total_forecast is not None:
            lines.append(f"  Total forecast: {self.total_forecast:.1f} | observed: {self.total_observed}")
        if self.battles_forecast is not None:
            lines.append(f"  Battles forecast: {self.battles_forecast:.1f} | observed: {self.battles_observed}")
        if self.erv_forecast is not None:
            lines.append(f"  Explosions/remote violence forecast: {self.erv_forecast:.1f} | observed: {self.erv_observed}")
        if self.vac_forecast is not None:
            lines.append(f"  Violence vs civilians forecast: {self.vac_forecast:.1f} | observed: {self.vac_observed}")
        return "\n".join(lines)

    @classmethod
    def from_api(cls, data: dict) -> "AcledForecast":
        def _float(v) -> Optional[float]:
            try:
                return float(v) if v not in (None, "") else None
            except (TypeError, ValueError):
                return None
        return cls(
            country=data.get("country", ""),
            admin1=data.get("admin1", ""),
            month=data.get("month", ""),
            year=int(data.get("year") or 0),
            total_forecast=_float(data.get("total_forecast")),
            battles_forecast=_float(data.get("battles_forecast")),
            erv_forecast=_float(data.get("erv_forecast")),
            vac_forecast=_float(data.get("vac_forecast")),
            total_observed=_float(data.get("total_observed")),
            battles_observed=_float(data.get("battles_observed")),
            erv_observed=_float(data.get("erv_observed")),
            vac_observed=_float(data.get("vac_observed")),
        )


# --- Tools ---

async def search_acled(
    ctx: RunContext[AgentDeps],
    country: str,
    days_back: int = 30,
    event_type: Optional[AcledEventType] = None,
    limit: int = 20,
) -> List[AcledEvent]:
    """Searches ACLED conflict database for armed conflict and security events in a country.

    Args:
        country (str): Country name. Example: "Ukraine", "Syria", "Sudan", "Mali", "Myanmar"
        days_back (int): How many days back to search. Defaults to 30.
        event_type (str, optional): Filter by event type. Omit to get all types.
        limit (int): Max events to return. Defaults to 20.

    Returns:
        List[AcledEvent]: Conflict events with date, type, actors, location, fatalities, notes.

    Example:
        search_acled(country="Ukraine", days_back=7, event_type="Battles", limit=15)
    """
    await ctx.deps.update_chat(f"_Searching ACLED conflict data: {country}_")
    start_date = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end_date = date.today().strftime("%Y-%m-%d")
    params: dict = {
        "country": country,
        "event_date": f"{start_date}|{end_date}",
        "event_date_where": "BETWEEN",
        "limit": limit,
    }
    if event_type:
        params["event_type"] = event_type
    async with AcledClient() as client:
        data = await client.request(endpoint="read", params=params)
    if data is None:
        logger.error(f"ACLED request failed for '{country}'")
        return []
    if "data" not in data:
        logger.error(f"ACLED response missing 'data' key for '{country}': {data}")
        return []
    events = [AcledEvent.from_api(e) for e in data["data"]]
    if not events:
        logger.info(f"No ACLED events found for '{country}'")
        return []
    SourceRegistry.register_all(ctx.deps.source_registry, events)
    return events


async def get_acled_forecast(
    ctx: RunContext[AgentDeps],
    country: str,
    year: int,
    month: Optional[int] = None,
) -> List[AcledForecast]:
    """Gets ACLED CAST conflict forecasts for a country — predicted vs observed conflict levels by month.

    Args:
        country (str): Country name. Example: "Mali", "Myanmar", "Ethiopia"
        year (int): Year to query. Example: 2026
        month (int, optional): Month number 1-12. Omit to get all months for the year.

    Returns:
        List[AcledForecast]: Forecast vs observed levels for battles, explosions/remote
        violence, and violence against civilians.

    Example:
        get_acled_forecast(country="Mali", year=2026, month=3)
    """
    await ctx.deps.update_chat(f"_Fetching ACLED conflict forecast: {country}_")
    params: dict = {"country": country, "year": str(year)}
    if month is not None:
        params["month"] = str(month).zfill(2)
    async with CastClient() as client:
        data = await client.request(endpoint="read", params=params)
    if data is None:
        logger.error(f"ACLED CAST request failed for '{country}'")
        return []
    if "data" not in data:
        logger.error(f"ACLED CAST response missing 'data' key for '{country}': {data}")
        return []
    forecasts = [AcledForecast.from_api(e) for e in data["data"]]
    if not forecasts:
        logger.info(f"No ACLED forecast found for '{country} {year}'")
        return []
    SourceRegistry.register_all(ctx.deps.source_registry, forecasts)
    return forecasts
