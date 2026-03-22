from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PraetorCase:
    id: str
    query: str
    expects_research: bool
    notes: str = ""


PROTEST_ACTIVITY = PraetorCase(
    id="protest_activity",
    query="What protests are happening in Los Angeles this week?",
    expects_research=True,
    notes="Local event query — should call run_research with Mobilize/web tools",
)

CANDIDATE_FINANCE = PraetorCase(
    id="candidate_finance",
    query="How much has Senator Ted Cruz raised in the 2026 cycle?",
    expects_research=True,
    notes="FEC data query — should call run_research with search_candidate_finance",
)

BLUESKY_SENTIMENT = PraetorCase(
    id="bluesky_sentiment",
    query="What are people saying on Bluesky about the DOGE cuts to federal agencies?",
    expects_research=True,
    notes="Social media sentiment — should call run_research with Bluesky tools",
)

CODING_QUESTION = PraetorCase(
    id="coding_question",
    query="Can you help me write a Python function to parse JSON?",
    expects_research=False,
    notes="Off-topic coding question — must NOT call run_research",
)

MATH_QUESTION = PraetorCase(
    id="math_question",
    query="What is the derivative of x squared plus 3x?",
    expects_research=False,
    notes="Off-topic math question — must NOT call run_research",
)

CREATIVE_WRITING = PraetorCase(
    id="creative_writing",
    query="Write me a short poem about autumn leaves.",
    expects_research=False,
    notes="Off-topic creative writing — must NOT call run_research",
)

ALL_CASES: list[PraetorCase] = [
    PROTEST_ACTIVITY,
    CANDIDATE_FINANCE,
    BLUESKY_SENTIMENT,
    CODING_QUESTION,
    MATH_QUESTION,
    CREATIVE_WRITING,
]
