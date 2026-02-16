import logging

import feedparser
import aiohttp
import asyncio
from pydantic_ai import RunContext

from typing import List

from ..models import RSSFeedItem
from ..settings import RSS
from ..ai import AgentDeps

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


def create_string_of_feed_items(feed_items: List[RSSFeedItem]) -> str:
    """Creates a string representation of all feeds."""
    return "Here are results from the RSS feeds:" + '\n'.join(
            str(item) for item in feed_items
        )


async def get_feed(feed_name: str, feeds_json: dict) -> str:
    """Fetches and returns current items from a single RSS feed."""
    data = feeds_json.get(feed_name)
    if not data or not data.get("url"):
        logger.warning(f"RSS feed '{feed_name}' not found in config")
        return ""
    async with aiohttp.ClientSession() as session:
        feed_objects = await fetch_feed(
                session=session,
                feed_name=feed_name,
                url=data["url"]
            )
    if not feed_objects:
        logger.error(f"Failed to fetch RSS feed '{feed_name}'")
        return ""
    saturated_items: List[RSSFeedItem] = await saturate_feed(feed_name, feed_objects)
    current_items = [
            item for item in saturated_items
            if item.current
        ]
    if not current_items:
        logger.info(f"No current items in RSS feed '{feed_name}'")
        return ""
    return create_string_of_feed_items(current_items)


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
    """Lists all US Government RSS feeds being monitored.
    
    Use this tool to get a list of all US Government RSS feeds
    so that you can select relevant feeds to query.
    """
    await ctx.deps.update_chat("_Listing US Government RSS feeds_")
    return await list_rss_feeds(
            json_feeds=RSS.US_GOV_JSON,
            feed_description="US Government"
    )


async def list_us_news_rss_feeds(ctx: RunContext[AgentDeps]):
    """Lists all US News RSS feeds being monitored.
    
    Use this tool to get a list of all US News RSS feeds
    so that you can select relevant feeds to query.
    """
    await ctx.deps.update_chat("_Listing US News RSS feeds_")
    return await list_rss_feeds(
            json_feeds=RSS.US_NEWS_JSON,
            feed_description="US News"
    )


async def list_world_news_rss_feeds(ctx: RunContext[AgentDeps]):
    """Lists all World News RSS feeds being monitored.
    
    Use this tool to get a list of all World News RSS feeds
    so that you can select relevant feeds to query.
    """
    await ctx.deps.update_chat("_Listing World News RSS feeds_")
    return await list_rss_feeds(
            json_feeds=RSS.WORLD_NEWS_JSON,
            feed_description="World News"
    )


async def get_gov_rss_feed(ctx: RunContext[AgentDeps], feed_name: str):
    """Fetches and returns current items from a single US Government RSS feed.
    This is for the United States Government ONLY. Use the other tools for
    news feeds. Only use feed names from the list_gov_rss_feeds tool.

    Parameters:
        feed_name: str - The name of the RSS feed to fetch. Use the
            list_gov_rss_feeds tool to get a list of valid feed names.
    Returns:
        str: A formatted string of current items from the specified RSS feed.
    """
    await ctx.deps.update_chat(f"_Fetching feeds for {feed_name}_")
    return await get_feed(
            feed_name=feed_name,
            feeds_json=RSS.US_GOV_JSON
        )

async def get_us_news_rss_feed(ctx: RunContext[AgentDeps], feed_name: str):
    """Fetches and returns current items from a single US News RSS feed.
    This is for US News ONLY. Use the other tools for government feeds 
    and world news feeds. Only use feed names from the list_us_news_rss_feeds
    tool.

    Parameters:
        feed_name: str - The name of the RSS feed to fetch. Use the
            list_us_news_rss_feeds tool to get a list of valid feed names.
    Returns:
        str: A formatted string of current items from the specified RSS feed.
    """
    await ctx.deps.update_chat(f"_Fetching feeds for {feed_name}_")
    return await get_feed(
            feed_name=feed_name,
            feeds_json=RSS.US_NEWS_JSON
        )


async def get_world_news_rss_feed(ctx: RunContext[AgentDeps], feed_name: str):
    """Fetches and returns current items from a single World News RSS feed.
    Only use feed names from the list_world_news_rss_feeds tool.

    Parameters:
        feed_name: str - The name of the RSS feed to fetch. Use the
            list_world_news_rss_feeds tool to get a list of valid feed names.
    Returns:
        str: A formatted string of current items from the specified RSS feed.
    """
    await ctx.deps.update_chat(f"_Fetching feeds for {feed_name}_")
    return await get_feed(
            feed_name=feed_name,
            feeds_json=RSS.WORLD_NEWS_JSON,
        )


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

