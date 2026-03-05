import aiohttp
import logging
from typing import Optional
from pydantic import BaseModel
from pydantic_ai import RunContext
from ..ai import AgentDeps
from ..source_registry import SourceRegistry
from ..settings import FECCredentials

logger = logging.getLogger(__name__)
FEC_API = "https://api.open.fec.gov/v1"


class CandidateFinance(BaseModel):
    candidate_id: str
    name: str
    party: str = ""
    office: str = ""
    state: str = ""
    cycle: int = 0
    receipts: float = 0.0
    disbursements: float = 0.0
    cash_on_hand: float = 0.0
    individual_contributions: float = 0.0
    pac_contributions: float = 0.0
    tag: str = ""

    @property
    def source_url(self) -> str:
        return f"https://www.fec.gov/data/candidate/{self.candidate_id}/"


class CommitteeFinance(BaseModel):
    committee_id: str
    name: str
    committee_type: str = ""
    state: str = ""
    receipts: float = 0.0
    disbursements: float = 0.0
    tag: str = ""

    @property
    def source_url(self) -> str:
        return f"https://www.fec.gov/data/committee/{self.committee_id}/"


async def search_candidate_finance(
    ctx: RunContext[AgentDeps], name: str, cycle: int = 2024
) -> Optional[CandidateFinance]:
    """Look up FEC campaign finance data for a US politician or candidate.

    Returns total raised, total spent, cash on hand, and breakdown of individual
    vs PAC contributions. Data sourced directly from FEC filings.

    Args:
        name (str): Politician's name. Example: "Alexandria Ocasio-Cortez", "Ted Cruz"
        cycle (int): Election cycle year. Example: 2024, 2022

    Returns:
        CandidateFinance with fundraising totals, or None if not found.

    Example:
        search_candidate_finance(name="Mitch McConnell", cycle=2024)
        search_candidate_finance(name="Bernie Sanders", cycle=2022)
    """
    await ctx.deps.update_chat(f"_FEC: candidate finance lookup for {name}_")
    key = FECCredentials.API_KEY
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{FEC_API}/candidates/search/",
                params={"q": name, "api_key": key, "per_page": 1, "sort": "-receipts"},
            ) as resp:
                data = await resp.json()
            results = data.get("results", [])
            if not results:
                return None
            cand = results[0]
            cid = cand["candidate_id"]

            async with session.get(
                f"{FEC_API}/candidates/totals/",
                params={"candidate_id": cid, "cycle": cycle, "api_key": key},
            ) as resp:
                data = await resp.json()
            totals = data.get("results", [{}])[0] if data.get("results") else {}

            result = CandidateFinance(
                candidate_id=cid,
                name=cand.get("name", name),
                party=cand.get("party_full", ""),
                office=cand.get("office_full", ""),
                state=cand.get("state", ""),
                cycle=cycle,
                receipts=totals.get("receipts", 0.0),
                disbursements=totals.get("disbursements", 0.0),
                cash_on_hand=totals.get("cash_on_hand_end_period", 0.0),
                individual_contributions=totals.get("individual_itemized_contributions", 0.0),
                pac_contributions=totals.get("other_political_committee_contributions", 0.0),
            )
            SourceRegistry.register_one(ctx.deps.source_registry, result)
            return result

    except Exception as e:
        logger.warning(f"FEC candidate search failed for '{name}': {e}")
        return None


async def search_committee_finance(
    ctx: RunContext[AgentDeps], org_name: str
) -> Optional[CommitteeFinance]:
    """Look up FEC financial data for a PAC, Super PAC, or political committee.

    Args:
        org_name (str): Organization or committee name. Example: "NRA Victory Fund", "ActBlue"

    Returns:
        CommitteeFinance with total receipts and disbursements, or None if not found.

    Example:
        search_committee_finance(org_name="Emily's List")
        search_committee_finance(org_name="Club for Growth")
    """
    await ctx.deps.update_chat(f"_FEC: committee finance lookup for {org_name}_")
    key = FECCredentials.API_KEY
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{FEC_API}/committees/",
                params={"q": org_name, "api_key": key, "per_page": 1, "sort": "-receipts"},
            ) as resp:
                data = await resp.json()
            results = data.get("results", [])
            if not results:
                return None
            comm = results[0]
            committee_id = comm["committee_id"]

            async with session.get(
                f"{FEC_API}/committee/{committee_id}/totals/",
                params={"api_key": key, "per_page": 1, "sort": "-cycle"},
            ) as resp:
                data = await resp.json()
            totals = data.get("results", [{}])[0] if data.get("results") else {}

            result = CommitteeFinance(
                committee_id=committee_id,
                name=comm.get("name", org_name),
                committee_type=comm.get("committee_type_full", ""),
                state=comm.get("state", ""),
                receipts=totals.get("receipts", 0.0),
                disbursements=totals.get("disbursements", 0.0),
            )
            SourceRegistry.register_one(ctx.deps.source_registry, result)
            return result

    except Exception as e:
        logger.warning(f"FEC committee search failed for '{org_name}': {e}")
        return None
