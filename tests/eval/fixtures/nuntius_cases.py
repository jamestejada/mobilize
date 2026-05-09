from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class NuntiusCase:
    id: str
    query: str
    source_data: str
    research_findings: str
    notes: str = ""


WELL_SOURCED_INPUT = NuntiusCase(
    id="single_topic_two_sources",
    query="What is the status of the proposed rail bill in Congress?",
    source_data="""Source Data (use [SOURCE_N] tags to cite):

[search_legislation]:
- [SOURCE_1] Rail Infrastructure Investment Act 2025 [via Congress.gov] [HIGH]: \
Bill S.1234 introduced March 2025, proposes $40B for passenger rail expansion, \
currently in Senate Commerce Committee.
- [SOURCE_2] Rail Bill Opposition Summary [via Reuters] [LOW]: \
Several Republican senators cited cost concerns; the bill passed the Senate 54-46 on April 7, 2026.""",
    research_findings="""Research Findings (Web/Social):
Congress passed the Rail Infrastructure Investment Act S.1234, a $40B rail bill, in the Senate 54-46 on April 7, 2026.
It had bipartisan support in committee but faced floor opposition from fiscal conservatives.
The bill now moves to the House.""",
    notes="Clean sourced input; expect dense [SOURCE_N] usage",
)

HIGH_SOURCE_COUNT = NuntiusCase(
    id="multi_source_corroborated",
    query="What happened at the Portland climate protest?",
    source_data="""Source Data (use [SOURCE_N] tags to cite):

[search_news]:
- [SOURCE_1] Portland Climate March — 5,000 Attend [via AP News] [HIGH]: \
Saturday protest drew estimated 5,000 participants near city hall, \
police reported no arrests.
- [SOURCE_2] Portland Protest Coverage [via OregonLive] [HIGH]: \
March ended peacefully; organizers claim 7,000 attendees.

[search_web]:
- [SOURCE_3] Climate Activists Portland [via PortlandMercury] [MEDIUM]: \
Third major climate event this year in Portland; next rally planned for June.""",
    research_findings="Protest in Portland Saturday with thousands attending peacefully.",
    notes="Multiple corroborated sources; output should use all three",
)

ANDES_VIRUS_NARROW_QUERY = NuntiusCase(
    id="andes_virus_person_to_person",
    query="Does Andes virus spread person to person?",
    source_data="""Source Data (use [SOURCE_N] tags to cite):

[search_web]:
- [SOURCE_1] CDC Hantavirus Overview [via CDC] [HIGH]: \
Andes virus is the only hantavirus known to spread from person to person, though this is rare and has mainly been documented in Argentina and Chile.
- [SOURCE_2] PAHO Andes Virus Alert [via PAHO] [HIGH]: \
Health authorities note limited person-to-person transmission of Andes virus in close-contact settings.""",
    research_findings="""Andes virus differs from most hantaviruses because person-to-person spread has been documented.""",
    notes="Regression case: answer the narrow question directly, without a broad hantavirus overview",
)

IRRELEVANT_SOURCE_ONLY = NuntiusCase(
    id="irrelevant_source_only",
    query="What did the mayor say about the transit strike?",
    source_data="""Source Data (use [SOURCE_N] tags to cite):

[search_news]:
- [SOURCE_1] Weather Alert [via AP News] [HIGH]: \
Heavy rain is expected across the region this weekend.
- [SOURCE_2] Sports Brief [via Reuters] [MEDIUM]: \
The city football club won 2-1 on Saturday.""",
    research_findings="""No verified findings about the mayor or the transit strike were collected.""",
    notes="Relevant Source Data is absent; writer should give a short insufficiency response",
)

ALL_CASES: list[NuntiusCase] = [
    WELL_SOURCED_INPUT,
    HIGH_SOURCE_COUNT,
    ANDES_VIRUS_NARROW_QUERY,
    IRRELEVANT_SOURCE_ONLY,
]
