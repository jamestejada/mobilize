import logging

import feedparser
import aiohttp
import asyncio
from pydantic_ai import RunContext

from typing import List

from ..models import RSSFeedItem
from ..settings import RSS
from ..ai import AgentDeps
from ..source_registry import SourceRegistry

logger = logging.getLogger(__name__)


# NOTE: Headers are needed because some government sites block requests without
#       proper User-Agent
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}   


async def fetch_feed(
            session: aiohttp.ClientSession,
            feed_name: str,
            url: str
         ) -> feedparser.FeedParserDict:
    try:
        async with session.get(url, headers=HEADERS) as response:
            content = await response.text()
            return feedparser.parse(content)
    except aiohttp.ClientError as e:
        logger.error(f"Failed to fetch RSS feed '{feed_name}' from {url}: {e}")


async def saturate_feed(feed_name: str, feed: feedparser.FeedParserDict) -> List[RSSFeedItem]:
    """Converts feedparser entries to RSSFeedItem dataclass instances."""
    return [
            RSSFeedItem.from_feedparser_entry(entry, feed_name)
            for entry in feed.entries
        ]


async def get_feed(feed_name: str, feeds_json: dict) -> List[RSSFeedItem] | str:
    """Fetches and returns current items from a single RSS feed."""
    data = feeds_json.get(feed_name)
    if not data or not data.get("url"):
        valid = ", ".join(f'"{k}"' for k in list(feeds_json.keys())[:10])
        msg = (
            f"RSS feed '{feed_name}' not found. "
            f"Call list_*_rss_feeds() first to get valid names. "
            f"Some valid names: {valid}..."
        )
        logger.warning(msg)
        return msg
    async with aiohttp.ClientSession() as session:
        feed_objects = await fetch_feed(
                session=session,
                feed_name=feed_name,
                url=data["url"]
            )
    if not feed_objects:
        logger.error(f"Failed to fetch RSS feed '{feed_name}'")
        return []
    saturated_items: List[RSSFeedItem] = await saturate_feed(feed_name, feed_objects)
    current_items = [
            item for item in saturated_items
            if item.current
        ]
    if not current_items:
        logger.info(f"No current items in RSS feed '{feed_name}'")
        return []
    return current_items


async def list_rss_feeds(json_feeds: dict, feed_description: str):
    """Lists all RSS feeds being monitored."""
    feed_names = list(json_feeds.keys())
    feed_list = "\n".join([f"- {name}" for name in feed_names])
    return (
            f"Please choose one of the currently monitored {feed_description}"
            f" RSS feeds to answer the user's question:\n{feed_list}"
        )


# Listing feeds for LLM use
async def list_gov_rss_feeds(ctx: RunContext[AgentDeps]):
    """Lists US Government RSS feed names — call first, then get_gov_rss_feed(feed_name).

    No parameters required.

    Returns:
        str: Formatted list of available US Government RSS feed names.

    Example:
        list_gov_rss_feeds()
    """
    await ctx.deps.update_chat("_Listing US Government RSS feeds_")
    return await list_rss_feeds(
            json_feeds=RSS.US_GOV_JSON,
            feed_description="US Government"
    )



async def list_world_news_rss_feeds(ctx: RunContext[AgentDeps]):
    """Lists world/international news RSS feed names — call first, then get_world_news_rss_feed(feed_name).

    No parameters required.

    Returns:
        str: Formatted list of available World News RSS feed names.

    Example:
        list_world_news_rss_feeds()
    """
    await ctx.deps.update_chat("_Listing World News RSS feeds_")
    return await list_rss_feeds(
            json_feeds=RSS.WORLD_NEWS_JSON,
            feed_description="World News"
    )


async def get_gov_rss_feed(ctx: RunContext[AgentDeps], feed_name: str) -> List[RSSFeedItem] | str:
    """Fetches current items from a US Government RSS feed by name (use after list_gov_rss_feeds).

    Args:
        feed_name (str): The exact name of the RSS feed to fetch. Use list_gov_rss_feeds
            first to get valid feed names. Example: "White House" or "State Department"

    Returns:
        List[RSSFeedItem]: List of current items from the specified RSS feed.

    Example:
        # Step 1: list_gov_rss_feeds() to see available feeds
        # Step 2: get_gov_rss_feed(feed_name="White House")
    """
    await ctx.deps.update_chat(f"_Fetching feeds for {feed_name}_")
    results = await get_feed(feed_name=feed_name, feeds_json=RSS.US_GOV_JSON)
    SourceRegistry.register_all(ctx.deps.source_registry, results)
    return results



async def get_world_news_rss_feed(ctx: RunContext[AgentDeps], feed_name: str) -> List[RSSFeedItem] | str:
    """Fetches a world news RSS feed by name, including specialized security/military outlets (use after list_world_news_rss_feeds).

    Args:
        feed_name (str): The exact name of the RSS feed to fetch. Use list_world_news_rss_feeds
            first to get valid feed names. Example: "BBC World" or "Al Jazeera"

    Returns:
        List[RSSFeedItem]: List of current items from the specified RSS feed.

    Example:
        # Step 1: list_world_news_rss_feeds() to see available feeds
        # Step 2: get_world_news_rss_feed(feed_name="BBC World")
    """
    await ctx.deps.update_chat(f"_Fetching feeds for {feed_name}_")
    results = await get_feed(feed_name=feed_name, feeds_json=RSS.WORLD_NEWS_JSON)
    SourceRegistry.register_all(ctx.deps.source_registry, results)
    return results


# =================== Cleanup Tools ===================


async def fetch_all_feeds(feeds_json: dict):
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_feed(session, name, data["url"])
            for name, data in feeds_json.items()
        ]
        return await asyncio.gather(*tasks)


async def find_outdated_feeds(feeds_json: dict):
    """Checks all feeds and reports which are outdated."""
    results = await fetch_all_feeds(feeds_json=feeds_json)
    outdated_feeds = []
    for result in results:
        name = result["name"]
        feed = result["feed"]
        if feed is None:
            outdated_feeds.append(name)
            continue
        entries = [
            not RSSFeedItem.from_feedparser_entry(
                    entry, name
                ).outdated
            for entry in feed.entries
        ]
        if not any(entries):
            outdated_feeds.append(name)
 
    if outdated_feeds:
        outdated_set = set(outdated_feeds)
        print("Outdated or inaccessible feeds detected:")
        for feed_name in outdated_set:
            print(
                f"- {feed_name}",
                "url:" ,
                feeds_json[feed_name]["url"]
            )

