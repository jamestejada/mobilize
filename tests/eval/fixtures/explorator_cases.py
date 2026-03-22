from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExploratorCase:
    id: str
    directive: str
    expected_tools: list[str]   # at least one of these should be called
    notes: str = ""


WEB_NEWS_SEARCH = ExploratorCase(
    id="web_news_search",
    directive=(
        "Use search_news and search_web to find recent coverage of US immigration policy. "
        "Summarize what you find."
    ),
    expected_tools=["search_news", "search_web"],
    notes="Basic web/news search — almost every model should handle this",
)

WIKIPEDIA_LOOKUP = ExploratorCase(
    id="wikipedia_lookup",
    directive=(
        "Use search_wikipedia to look up the US Electoral College. "
        "Summarize the key facts."
    ),
    expected_tools=["search_wikipedia"],
    notes="Single-tool directive — tests whether model follows explicit tool instruction",
)

REDDIT_SEARCH = ExploratorCase(
    id="reddit_search",
    directive=(
        "Use search_reddit_history to find Reddit discussions about climate change policy. "
        "Summarize the main viewpoints."
    ),
    expected_tools=["search_reddit_history"],
    notes="Tests PullPush API tool invocation",
)

MULTI_TOOL_FOLLOWUP = ExploratorCase(
    id="multi_tool_followup",
    directive=(
        "Search for news about the US-China trade relationship using search_news. "
        "Then use fetch_url or search_web to get more detail on any important story you find."
    ),
    expected_tools=["search_news"],
    notes="Tests whether model chains tools (news → fetch_url/web); at minimum must call search_news",
)

ALL_CASES: list[ExploratorCase] = [
    WEB_NEWS_SEARCH,
    WIKIPEDIA_LOOKUP,
    REDDIT_SEARCH,
    MULTI_TOOL_FOLLOWUP,
]
