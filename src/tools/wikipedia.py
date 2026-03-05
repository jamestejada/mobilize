import aiohttp
import logging
from typing import List
from pydantic import BaseModel
from pydantic_ai import RunContext
from ..ai import AgentDeps

logger = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
MAX_EXTRACT = 600


class WikipediaSummary(BaseModel):
    title: str
    extract: str
    url: str
    tag: str = ""

    @property
    def source_url(self) -> str:
        return self.url


async def search_wikipedia(
    ctx: RunContext[AgentDeps], query: str
) -> List[WikipediaSummary]:
    """Search Wikipedia for background information on a topic, person, or organization.

    Best for: getting reliable background on named entities, historical events,
    organizations, or technical concepts encountered during research. Returns
    clean article summaries without noisy HTML. Results are context-only and
    are never cited as sources.

    Args:
        query (str): Topic to look up. Example: "Extinction Rebellion", "Standing Rock protest"

    Returns:
        List[WikipediaSummary]: Up to 3 Wikipedia summaries with title, extract, and URL.

    Example:
        search_wikipedia(query="Sunrise Movement")
        search_wikipedia(query="Alexandria Ocasio-Cortez")
    """
    await ctx.deps.update_chat(f"_Wikipedia: {query}_")
    results = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                WIKI_API,
                params={
                    "action": "opensearch",
                    "search": query,
                    "limit": 3,
                    "format": "json",
                },
            ) as resp:
                data = await resp.json()

            if not data or len(data) < 4:
                return []

            titles: list[str] = data[1][:3]
            urls: list[str] = data[3][:3]

            for title, url in zip(titles, urls):
                encoded = title.replace(" ", "_")
                async with session.get(WIKI_SUMMARY.format(encoded)) as resp:
                    if resp.status != 200:
                        continue
                    summary = await resp.json()
                extract = (summary.get("extract") or "")[:MAX_EXTRACT]
                if extract:
                    results.append(WikipediaSummary(title=title, extract=extract, url=url))

    except Exception as e:
        logger.warning(f"Wikipedia search failed for '{query}': {e}")

    return results
