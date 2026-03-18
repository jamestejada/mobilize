import asyncio
import logging
from typing import List
import aiohttp
from pydantic import BaseModel
from pydantic_ai import RunContext
from ..ai import AgentDeps
from ..settings import CourtListenerCredentials
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


async def _fetch_opinion_text(session: aiohttp.ClientSession, opinion_id: int, headers: dict) -> str:
    try:
        async with session.get(
            f"{CL_API}/opinions/{opinion_id}/",
            params={"format": "json"},
            headers=headers,
        ) as resp:
            data = await resp.json()
            return (data.get("plain_text", "") or "").strip()[:500]
    except Exception as e:
        logger.warning(f"CourtListener opinion fetch failed for id {opinion_id}: {e}")
        return ""


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
        List[CourtCase]: Matching cases with name, court, date, summary, and CourtListener URL.

    Example:
        search_court_cases(query="January 6 seditious conspiracy")
        search_court_cases(query="EPA climate regulations", court="cadc")
    """
    await ctx.deps.update_chat(f"_CourtListener: searching cases for '{query}'_")
    cases = []
    headers = {}
    if CourtListenerCredentials.API_KEY:
        headers["Authorization"] = f"Token {CourtListenerCredentials.API_KEY}"

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
            async with session.get(f"{CL_API}/search/", params=params, headers=headers) as resp:
                data = await resp.json()

            for item in (data.get("results") or [])[:3]:
                case_url = f"https://www.courtlistener.com{item.get('absolute_url', '')}"
                snippet = (item.get("snippet", "") or "").strip()
                opinion_id = ((item.get("opinions") or [{}])[0]).get("id")
                cases.append(CourtCase(
                    case_name=item.get("caseName", ""),
                    court=item.get("court_citation_string", ""),
                    date_filed=item.get("dateFiled", ""),
                    summary=snippet,  # placeholder; replaced below
                    url=case_url,
                    tag=str(opinion_id) if opinion_id else "",
                ))

            # Fetch full opinion text in parallel, fall back to snippet
            async def _empty() -> str:
                return ""

            opinion_ids = [int(c.tag) if c.tag else None for c in cases]
            texts = await asyncio.gather(
                *[_fetch_opinion_text(session, oid, headers) if oid else _empty() for oid in opinion_ids]
            )
            for case, text in zip(cases, texts):
                case.summary = text or case.summary
                case.tag = ""  # clear temp opinion_id storage

    except Exception as e:
        logger.warning(f"CourtListener search failed for '{query}': {e}")

    SourceRegistry.register_all(ctx.deps.source_registry, cases)
    return cases
