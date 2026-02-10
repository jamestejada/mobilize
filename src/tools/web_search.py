from ddgs import DDGS
from pydantic_ai import RunContext

from dataclasses import dataclass
from typing import Optional

from ..ai import AgentDeps


@dataclass
class WebResult:
    title: str
    href: str
    body: str

    def __str__(self) -> str:
        return "\n".join([
                f"Title: {self.title}",
                f"URL: {self.href}",
                f"Body: {self.body}"
            ])


async def search_web(ctx: RunContext[AgentDeps], query: str, num_results: int = 20) -> str:
    """Performs a web search and returns summarized results.
    
    Args:
        query (str): The search query.
        num_results (int, optional): Number of search results to return. Defaults to 20.
    
    Returns:
        str: A summary of the top search results.
    """
    await ctx.deps.update_chat(f"_Searching web for: {query}_")
    results = DDGS().text(
        query,
        max_results=num_results
    )
    return (
            f"Web search results for '{query}':"
            + "\n\n---------\n\n".join([str(result) for result in results])
        )


@dataclass
class NewsResult:
    date: str
    title: str
    body: str
    url: str
    image: Optional[str] = None
    source: Optional[str] = None

    def __str__(self) -> str:
        return "\n".join([
                f"Date: {self.date}",
                f"Title: {self.title}",
                f"Source: {self.source}" if self.source else "",
                f"URL: {self.url}",
                f"Body: {self.body}",
                f"Image: {self.image}" if self.image else ""
            ])


async def search_news(ctx: RunContext[AgentDeps], query: str, num_results: int = 20) -> str:
    """Performs a news search and returns summarized results.
    
    Args:
        query (str): The search query.
        num_results (int, optional): Number of news results to return. Defaults to 20.
    
    Returns:
        str: A summary of the top news results.
    """
    await ctx.deps.update_chat(f"_Searching news for: {query}_")
    results = DDGS().news(
        query,
        max_results=num_results
    )
    return (
            f"News search results for '{query}':"
            + "\n\n---------\n\n".join([str(result) for result in results])
        )