import logging

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from pydantic_ai import RunContext
from pydantic import BaseModel

from typing import Optional, List

from ..ai import AgentDeps

logger = logging.getLogger(__name__)


class WebResult(BaseModel):
    title: str
    href: str
    body: str

    @property
    def source_url(self) -> str:
        return self.href

    def __str__(self) -> str:
        return "\n".join([
                f"Title: {self.title}",
                f"URL: {self.href}",
                f"Body: {self.body}"
            ])


async def search_web(ctx: RunContext[AgentDeps], query: str, num_results: int = 20) -> List[WebResult]:
    """Performs a web search and returns results as structured models.

    Args:
        query (str): The search query text. Example: "renewable energy policy 2026"
        num_results (int, optional): Number of search results to return. Defaults to 20.

    Returns:
        List[WebResult]: List of web search results with title, URL, and body text.

    Example:
        search_web(query="artificial intelligence regulation", num_results=15)
    """
    await ctx.deps.update_chat(f"_Searching web for: {query}_")
    try:
        results = [
            WebResult(**r) for r in DDGS().text(query, max_results=num_results)
        ]
    except DDGSException as e:
        logger.error(f"Web search failed for '{query}': {e}")
        return []

    return results


class NewsResult(BaseModel):
    date: str
    title: str
    body: str
    url: str
    image: Optional[str] = None
    source: Optional[str] = None

    @property
    def source_url(self) -> str:
        return self.url

    def __str__(self) -> str:
        return "\n".join([
                f"Date: {self.date}",
                f"Title: {self.title}",
                f"Source: {self.source}" if self.source else "",
                f"URL: {self.url}",
                f"Body: {self.body}",
                f"Image: {self.image}" if self.image else ""
            ])


async def search_news(ctx: RunContext[AgentDeps], query: str, num_results: int = 20) -> List[NewsResult]:
    """Performs a news search and returns results as structured models.

    Args:
        query (str): The search query text. Example: "federal reserve interest rates"
        num_results (int, optional): Number of news results to return. Defaults to 20.

    Returns:
        List[NewsResult]: List of news results with date, title, source, URL, and body.

    Example:
        search_news(query="supreme court decision", num_results=10)
    """
    await ctx.deps.update_chat(f"_Searching news for: {query}_")
    try:
        results = [
            NewsResult(**r) for r in DDGS().news(query, max_results=num_results)
        ]
    except DDGSException as e:
        logger.error(f"News search failed for '{query}': {e}")
        return []
    return results
