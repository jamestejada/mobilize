from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ProbatorCase:
    id: str
    query: str
    research_findings: str
    expected_verdict: Literal["ADEQUATE", "GAPS"]
    notes: str = ""


COMPREHENSIVE_RESEARCH = ProbatorCase(
    id="well_covered_single_topic",
    query="What is the current legislative status of the Rail Infrastructure Investment Act?",
    research_findings="""Research Findings:
The Rail Infrastructure Investment Act S.1234 was introduced by Sen. Collins in March 2025 [SOURCE_1].
It passed the Senate Commerce Committee 12-8 on February 14, 2026 [SOURCE_2].
The Senate passed the bill 54-46 on April 7, 2026 [SOURCE_3].
Opposition from five Republican senators cited $40B cost [SOURCE_4].
A companion House bill H.R.2234 was introduced by Rep. Ocasio-Cortez and is pending committee review [SOURCE_5].""",
    expected_verdict="ADEQUATE",
    notes="Full coverage: introduction, committee vote, floor vote outcome, opposition, companion bill",
)

MISSING_KEY_ANGLE = ProbatorCase(
    id="missing_opposition_detail",
    query="What are both sides saying about the Rail Infrastructure Investment Act?",
    research_findings="""Research Findings:
The Rail Infrastructure Investment Act S.1234 has strong Democratic support.
Sen. Collins introduced the bill citing infrastructure needs [SOURCE_1].
Progressive groups have rallied for passage [SOURCE_2].""",
    expected_verdict="GAPS",
    notes="Query asks for both sides but research only covers proponents",
)

SINGLE_LOW_CONFIDENCE_SOURCE = ProbatorCase(
    id="important_claim_single_low_source",
    query="Was the Portland protest organizer arrested?",
    research_findings="""Research Findings:
According to a single tweet, protest organizer Jane Doe was detained briefly [SOURCE_1 LOW].
No official police statement has been found.""",
    expected_verdict="GAPS",
    notes="Important factual claim (arrest) backed by only a single LOW-confidence source",
)

ALL_CASES: list[ProbatorCase] = [
    COMPREHENSIVE_RESEARCH,
    MISSING_KEY_ANGLE,
    SINGLE_LOW_CONFIDENCE_SOURCE,
]
