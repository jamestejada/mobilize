import json
import logging
from typing import List
from pydantic import BaseModel, field_validator

from pydantic_ai import RunContext

from ..ai import AgentDeps
from .http_client import AsyncHTTPClient

logger = logging.getLogger(__name__)


class PolymarketMarket(BaseModel):
    question: str
    outcomes: List[str]
    prices: List[float]
    volume: float = 0.0

    def __str__(self) -> str:
        odds = ", ".join(
            f"{o}: {p * 100:.1f}%"
            for o, p in zip(self.outcomes, self.prices)
        )
        return (
            f"Q: {self.question}\n"
            f"Odds: {odds}\n"
            f"Volume: ${self.volume:,.0f}"
        )

    @classmethod
    def from_api(cls, data: dict) -> "PolymarketMarket":
        outcomes_raw = data.get("outcomes", "[]")
        prices_raw = data.get("outcomePrices", "[]")
        try:
            outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
            prices = [float(p) for p in (json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw)]
        except (json.JSONDecodeError, ValueError):
            outcomes = []
            prices = []
        return cls(
            question=data.get("question", "Unknown"),
            outcomes=outcomes,
            prices=prices,
            volume=float(data.get("volume", 0)),
        )


class PolymarketEvent(BaseModel):
    title: str
    slug: str
    markets: List[PolymarketMarket] = []

    @property
    def source_url(self) -> str:
        return f"https://polymarket.com/event/{self.slug}" if self.slug else ""

    def __str__(self) -> str:
        lines = [f"Event: {self.title}"]
        for market in self.markets:
            lines.append(f"  {str(market).replace(chr(10), chr(10) + '  ')}")
            lines.append("  ---")
        return "\n".join(lines)

    @classmethod
    def from_api(cls, data: dict) -> "PolymarketEvent":
        return cls(
            title=data.get("title", "Unknown"),
            slug=data.get("slug", ""),
            markets=[
                PolymarketMarket.from_api(m)
                for m in data.get("markets", [])
            ],
        )


class GammaClient(AsyncHTTPClient):
    BASE_URL = "https://gamma-api.polymarket.com"


# Tools
async def search_polymarket(
            ctx: RunContext[AgentDeps],
            query: str,
            limit: int = 5
        ) -> List[PolymarketEvent]:
    """Searches Polymarket for prediction markets related to a query.
    Returns betting odds that reflect crowd sentiment on the topic.

    Args:
        query (str): The topic to search for. Examples: "tariffs", "recession 2026", "election results"
        limit (int, optional): Max number of events to return. Defaults to 5.

    Returns:
        List[PolymarketEvent]: List of prediction market events with betting odds.

    Example:
        search_polymarket(query="economic downturn", limit=3)
    """
    await ctx.deps.update_chat(f"_Searching Polymarket: {query}_")
    async with GammaClient() as client:
        data = await client.request(
            endpoint="public-search",
            params={
                "q": query,
                "limit_per_type": limit,
                "events_status": "active",
            }
        )
    if data is None:
        logger.error(f"Failed to fetch Polymarket data for '{query}'")
        return []
    events = [PolymarketEvent.from_api(e) for e in data.get("events", [])]
    if not events:
        logger.info(f"No active prediction markets found for '{query}'")
        return []
    return events


async def get_polymarket_event(
            ctx: RunContext[AgentDeps],
            slug: str
        ) -> PolymarketEvent | None:
    """Gets detailed information about a specific Polymarket event by its slug.

    Typically used after search_polymarket to get full details on a specific event.

    Args:
        slug (str): The event slug from search results or URL. Example: "will-recession-occur-2026"

    Returns:
        PolymarketEvent | None: The prediction market event or None if not found.

    Example:
        # Step 1: search_polymarket(query="recession") to find events
        # Step 2: get_polymarket_event(slug="will-recession-occur-2026") for details
    """
    await ctx.deps.update_chat(f"_Fetching Polymarket event: {slug}_")
    async with GammaClient() as client:
        data = await client.request(
            endpoint="events",
            params={"slug": slug}
        )
    if data is None:
        logger.error(f"Failed to fetch Polymarket event '{slug}'")
        return None
    if not data:
        logger.info(f"No Polymarket event found with slug '{slug}'")
        return None
    event_data = data[0] if isinstance(data, list) else data
    event = PolymarketEvent.from_api(event_data)
    return event
