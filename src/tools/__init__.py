
from pydantic_ai.toolsets import FunctionToolset, CombinedToolset

from src.tools.mobilize import get_protests_for_llm
from src.tools.web_search import search_web, search_news
from src.tools.bsky import (
    search_bluesky_posts,
    get_bluesky_profile,
    get_author_feed,
    get_trending_topics
    )
from src.tools.rss import (
    get_gov_rss_feed,
    get_world_news_rss_feed,
)
from src.tools.polymarket import search_polymarket, get_polymarket_event
from src.tools.fetch_url import fetch_url
from src.tools.sources import get_registered_sources
from src.tools.wikipedia import search_wikipedia
from src.tools.reddit import search_reddit_history
from src.tools.wayback import fetch_archived_page
from src.tools.fec import search_candidate_finance, search_committee_finance
from src.tools.congress import search_legislation
from src.tools.courtlistener import search_court_cases


SHARED_TOOLSET = FunctionToolset([get_registered_sources, fetch_url])

# Web and social media research tools
EXPLORATOR_TOOLSET = FunctionToolset([
    search_web,
    search_news,
    search_bluesky_posts,
    get_bluesky_profile,
    get_author_feed,
    get_trending_topics,
    search_wikipedia,
    search_reddit_history,
    fetch_archived_page,
])

# Structured data sources: RSS feeds, events, prediction markets
TABULARIUS_TOOLSET = FunctionToolset([
    get_gov_rss_feed,
    get_world_news_rss_feed,
    get_protests_for_llm,
    search_polymarket,
    get_polymarket_event,
    search_candidate_finance,
    search_committee_finance,
    search_legislation,
    search_court_cases,
])

EXPLORATOR_AGENT_TOOLSET = CombinedToolset([SHARED_TOOLSET, EXPLORATOR_TOOLSET])
TABULARIUS_AGENT_TOOLSET = CombinedToolset([SHARED_TOOLSET, TABULARIUS_TOOLSET])
ALL_RESEARCH_TOOLSET = CombinedToolset([EXPLORATOR_TOOLSET, TABULARIUS_TOOLSET])
