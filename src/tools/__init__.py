

from pydantic_ai import Tool

from src.tools.mobilize import get_protests_for_llm
from src.tools.web_search import search_web, search_news
from src.tools.bsky import (
    search_bluesky_posts,
    get_bluesky_profile,
    get_author_feed,
    get_trending_topics
    )
from src.tools.rss_feeds import (
    list_gov_rss_feeds,
    get_gov_rss_feed,
    list_world_news_rss_feeds,
    get_world_news_rss_feed
)
from src.tools.polymarket import search_polymarket, get_polymarket_event
from src.tools.fetch_url import fetch_url
from src.tools.sources import get_registered_sources


SHARED_TOOLS = [
    Tool(get_registered_sources, takes_ctx=True),
    Tool(fetch_url, takes_ctx=True),
]

# Web and social media research tools
EXPLORATOR_ONLY_TOOLS = [
    Tool(search_web, takes_ctx=True),
    Tool(search_news, takes_ctx=True),
    Tool(search_bluesky_posts, takes_ctx=True),
    Tool(get_bluesky_profile, takes_ctx=True),
    Tool(get_author_feed, takes_ctx=True),
    Tool(get_trending_topics, takes_ctx=True),
]

# Structured data sources: RSS feeds, events, prediction markets
TABULARIUS_ONLY_TOOLS = [
    Tool(list_gov_rss_feeds, takes_ctx=True),
    Tool(get_gov_rss_feed, takes_ctx=True),
    Tool(list_world_news_rss_feeds, takes_ctx=True),
    Tool(get_world_news_rss_feed, takes_ctx=True),
    Tool(get_protests_for_llm, takes_ctx=True),
    Tool(search_polymarket, takes_ctx=True),
    Tool(get_polymarket_event, takes_ctx=True),
]

EXPLORATOR_TOOLS = SHARED_TOOLS + EXPLORATOR_ONLY_TOOLS
TABULARIUS_TOOLS = SHARED_TOOLS + TABULARIUS_ONLY_TOOLS
ALL_RESEARCH_TOOLS = EXPLORATOR_ONLY_TOOLS + TABULARIUS_ONLY_TOOLS

