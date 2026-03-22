from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class CogitatorCase:
    id: str
    query: str
    draft: str
    expected_verdict: Literal["APPROVED", "IMPROVE"]
    notes: str = ""


GOOD_DRAFT = CogitatorCase(
    id="well_cited_draft",
    query="What happened at the Portland climate protest?",
    draft=(
        "Saturday's Portland climate march drew between 5,000 and 7,000 participants "
        "near city hall [SOURCE_1] [SOURCE_2]. Police reported no arrests [SOURCE_1]. "
        "A third major climate event is planned for June [SOURCE_3]."
    ),
    expected_verdict="APPROVED",
    notes="Every claim has a source tag; should be approved",
)

DRAFT_WITH_RAW_URL = CogitatorCase(
    id="raw_url_present",
    query="What is the status of the rail bill?",
    draft=(
        "The Rail Infrastructure Investment Act S.1234 has been introduced in the Senate "
        "[SOURCE_1]. More details at https://congress.gov/bill/s1234/text."
    ),
    expected_verdict="IMPROVE",
    notes="Contains a raw https:// URL; must be flagged",
)

DRAFT_MISSING_CITATIONS = CogitatorCase(
    id="missing_citation_mid_paragraph",
    query="What legislation is pending on climate?",
    draft=(
        "Multiple climate bills are pending in Congress. The Clean Energy Futures Act "
        "would invest $200B in renewables. Supporters argue it will create 2 million jobs. "
        "The bill has bipartisan co-sponsors [SOURCE_1]."
    ),
    expected_verdict="IMPROVE",
    notes="Three factual sentences before the first source tag",
)

DRAFT_INTERNALLY_CONSISTENT = CogitatorCase(
    id="single_source_primary",
    query="What is the FEC fine amount for the Acme PAC?",
    draft=(
        "The Federal Election Commission fined Acme PAC $450,000 for disclosure violations [SOURCE_1]."
    ),
    expected_verdict="APPROVED",
    notes="Single claim, single primary source — should approve",
)

ALL_CASES: list[CogitatorCase] = [
    GOOD_DRAFT,
    DRAFT_WITH_RAW_URL,
    DRAFT_MISSING_CITATIONS,
    DRAFT_INTERNALLY_CONSISTENT,
]
