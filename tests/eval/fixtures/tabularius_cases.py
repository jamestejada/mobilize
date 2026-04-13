from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TabulariusCase:
    id: str
    directive: str
    expected_tools: list[str]   # at least one of these should be called
    notes: str = ""


LEGISLATION_SEARCH = TabulariusCase(
    id="legislation_search",
    directive=(
        "Use search_legislation to find bills related to immigration in the 119th Congress. "
        "Summarize what you find."
    ),
    expected_tools=["search_legislation"],
    notes="Congress.gov tool — stable dataset, good for reproducibility",
)

COURT_CASES_SEARCH = TabulariusCase(
    id="court_cases_search",
    directive=(
        "Use search_court_cases to find Supreme Court cases about voting rights. "
        "Summarize the key cases."
    ),
    expected_tools=["search_court_cases"],
    notes="CourtListener tool — tests structured legal data retrieval",
)

RSS_FEED_TWO_STEP = TabulariusCase(
    id="rss_feed_two_step",
    directive=(
        "Use get_gov_rss_feed to fetch headlines from a US Government RSS feed. "
        "Summarize the headlines."
    ),
    expected_tools=["get_gov_rss_feed"],
    notes="RSS discovery pattern — model must call get_gov_rss_feed() twice: once to discover names, once to fetch",
)

FEC_FINANCE_SEARCH = TabulariusCase(
    id="fec_finance_search",
    directive=(
        "Use search_candidate_finance to look up campaign finance information "
        "for a major 2024 presidential candidate. Summarize the fundraising totals."
    ),
    expected_tools=["search_candidate_finance"],
    notes="FEC API — tests structured finance data retrieval",
)

ALL_CASES: list[TabulariusCase] = [
    LEGISLATION_SEARCH,
    COURT_CASES_SEARCH,
    RSS_FEED_TWO_STEP,
    FEC_FINANCE_SEARCH,
]
