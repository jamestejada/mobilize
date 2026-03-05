import aiohttp
import logging
from typing import List
from pydantic import BaseModel
from pydantic_ai import RunContext
from ..ai import AgentDeps
from ..source_registry import SourceRegistry

logger = logging.getLogger(__name__)
PULLPUSH_API = "https://api.pullpush.io/reddit/search/submission/"


class RedditPost(BaseModel):
    title: str
    subreddit: str = ""
    score: int = 0
    url: str = ""
    permalink: str = ""
    selftext: str = ""
    author: str = ""
    num_comments: int = 0
    tag: str = ""

    @property
    def source_url(self) -> str:
        return f"https://www.reddit.com{self.permalink}"


async def search_reddit_history(
    ctx: RunContext[AgentDeps], query: str, subreddit: str = "", limit: int = 10
) -> List[RedditPost]:
    """Search Reddit post history for community discussion and background context.

    Note: Results may be 1-3 days behind real-time due to archive indexing lag.
    Best for: background context, community sentiment, historical discussions.
    For current Reddit threads use search_web with "site:reddit.com {query}" instead.

    Args:
        query (str): Search terms. Example: "pipeline protest environmental"
        subreddit (str): Subreddit to search, or leave empty for site-wide.
                         Example: "politics", "environment", "news"
        limit (int): Number of posts to return. Default 10, max 25.

    Returns:
        List[RedditPost]: Posts with title, score, comment count, and body text.

    Example:
        search_reddit_history(query="climate bill congress", subreddit="environment")
        search_reddit_history(query="standing rock pipeline")
    """
    await ctx.deps.update_chat(f"_Reddit: searching for '{query}'_")
    posts = []
    try:
        params: dict = {
            "q": query,
            "size": min(limit, 25),
            "sort": "desc",
            "sort_type": "score",
        }
        if subreddit:
            params["subreddit"] = subreddit

        async with aiohttp.ClientSession() as session:
            async with session.get(PULLPUSH_API, params=params) as resp:
                data = await resp.json()

        for item in (data.get("data") or []):
            posts.append(RedditPost(
                title=item.get("title", ""),
                subreddit=item.get("subreddit", ""),
                score=item.get("score", 0),
                url=item.get("url", ""),
                permalink=item.get("permalink", ""),
                selftext=(item.get("selftext") or "")[:500],
                author=item.get("author", ""),
                num_comments=item.get("num_comments", 0),
            ))

    except Exception as e:
        logger.warning(f"Reddit search failed for '{query}': {e}")

    SourceRegistry.register_all(ctx.deps.source_registry, posts)
    return posts
