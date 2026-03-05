import aiohttp
import logging
from typing import List
from pydantic import BaseModel
from pydantic_ai import RunContext
from ..ai import AgentDeps
from ..source_registry import SourceRegistry
from ..settings import CongressCredentials

logger = logging.getLogger(__name__)
CONGRESS_API = "https://api.congress.gov/v3"


class Bill(BaseModel):
    bill_id: str
    title: str
    congress: int = 0
    bill_type: str = ""
    introduced_date: str = ""
    latest_action: str = ""
    sponsor: str = ""
    url: str = ""
    tag: str = ""

    @property
    def source_url(self) -> str:
        return self.url


async def search_legislation(
    ctx: RunContext[AgentDeps], query: str, congress: int = 119
) -> List[Bill]:
    """Search US Congress bills and legislation by keyword.

    Args:
        query (str): Keywords to search. Example: "climate change", "immigration reform"
        congress (int): Congress number. 119 = current (2025-2026), 118 = prior (2023-2024).

    Returns:
        List[Bill]: Matching bills with title, sponsor, status, and congress.gov link.

    Example:
        search_legislation(query="voting rights", congress=119)
        search_legislation(query="healthcare", congress=118)
    """
    await ctx.deps.update_chat(f"_Congress.gov: searching legislation for '{query}'_")
    bills = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{CONGRESS_API}/bill",
                params={
                    "query": query,
                    "congress": congress,
                    "api_key": CongressCredentials.API_KEY,
                    "limit": 10,
                    "format": "json",
                    "sort": "updateDate+desc",
                },
            ) as resp:
                data = await resp.json()

        for item in (data.get("bills") or []):
            latest = (item.get("latestAction") or {})
            sponsor = (item.get("sponsors") or [{}])[0]
            bill_url = (item.get("url") or "").replace("?format=json", "")
            bills.append(Bill(
                bill_id=f"{item.get('type', '')}{item.get('number', '')}",
                title=item.get("title", ""),
                congress=item.get("congress", 0),
                bill_type=item.get("type", ""),
                introduced_date=item.get("introducedDate", ""),
                latest_action=latest.get("text", ""),
                sponsor=sponsor.get("fullName", ""),
                url=bill_url,
            ))

    except Exception as e:
        logger.warning(f"Congress.gov search failed for '{query}': {e}")

    SourceRegistry.register_all(ctx.deps.source_registry, bills)
    return bills
