import aiohttp
import logging
from typing import List
from pydantic import BaseModel
from pydantic_ai import RunContext
from ..ai import AgentDeps
from ..source_registry import SourceRegistry

logger = logging.getLogger(__name__)
CL_API = "https://www.courtlistener.com/api/rest/v4"


class CourtCase(BaseModel):
    case_name: str
    court: str = ""
    date_filed: str = ""
    summary: str = ""
    url: str = ""
    tag: str = ""

    @property
    def source_url(self) -> str:
        return self.url


async def search_court_cases(
    ctx: RunContext[AgentDeps], query: str, court: str = ""
) -> List[CourtCase]:
    """Search federal and state court opinions and case filings.

    Best for: tracking legal proceedings involving public figures, organizations,
    or civil rights/environmental cases. Returns federal and state court opinions.

    Args:
        query (str): Search terms. Example: "Standing Rock pipeline DAPL", "Proud Boys sedition"
        court (str): Optional court code to filter. Example: "ca9" (9th Circuit), "scotus"
                     Leave empty to search all courts.

    Returns:
        List[CourtCase]: Matching cases with name, court, date, and CourtListener URL.

    Example:
        search_court_cases(query="January 6 seditious conspiracy")
        search_court_cases(query="EPA climate regulations", court="cadc")
    """
    await ctx.deps.update_chat(f"_CourtListener: searching cases for '{query}'_")
    cases = []
    try:
        params = {
            "q": query,
            "type": "o",
            "order_by": "score desc",
            "format": "json",
        }
        if court:
            params["court"] = court

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{CL_API}/search/", params=params) as resp:
                data = await resp.json()

        for item in (data.get("results") or [])[:10]:
            case_url = f"https://www.courtlistener.com{item.get('absolute_url', '')}"
            cases.append(CourtCase(
                case_name=item.get("caseName", ""),
                court=item.get("court_citation_string", ""),
                date_filed=item.get("dateFiled", ""),
                summary=(item.get("snippet", "") or "")[:300],
                url=case_url,
            ))

    except Exception as e:
        logger.warning(f"CourtListener search failed for '{query}': {e}")

    SourceRegistry.register_all(ctx.deps.source_registry, cases)
    return cases
