from __future__ import annotations

import asyncio
import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart

from src.ai import Tabularius
from src.agent_settings import AgentsConfiguration
from src.settings import Prompts
from tests.eval.conftest import TOOL_USE_MODELS, make_eval_deps
from tests.eval.evaluator import EvaluatorAgent
from tests.eval.tool_helpers import get_calls_for_tool, get_tool_invocations, tool_return_looks_successful
from tests.eval.fixtures.tabularius_cases import (
    LEGISLATION_SEARCH,
    COURT_CASES_SEARCH,
    RSS_FEED_TWO_STEP,
    FEC_FINANCE_SEARCH,
    TabulariusCase,
)

pytestmark = [pytest.mark.eval, pytest.mark.asyncio]

# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

TOOL_ARGS_QUALITY_CRITERION = """
You are given a research directive and the arguments a model passed to a structured data tool.
PASS if the arguments are specific and relevant to the directive (e.g. a meaningful query string, correct congress number).
FAIL if any required argument is empty, generic (e.g. just "query"), or clearly unrelated to the directive.
FAIL if the query argument is longer than 150 characters (likely a hallucinated prompt dump).
"""

RESEARCH_RELEVANCE_CRITERION = """
The agent was given a research directive and returned structured data findings.
PASS if the output contains substantive information relevant to the directive topic.
FAIL if the output is empty, off-topic, or just repeats the directive without findings.
FAIL if the output says it could not find anything when a tool was available.
"""

RSS_TWO_STEP_CRITERION = """
The agent was directed to first list RSS feeds, then fetch one.
PASS if the output contains actual RSS feed headlines or article summaries.
FAIL if the output only lists feed names without fetching any content.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_called_tools(result) -> set[str]:
    called = set()
    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    called.add(part.tool_name)
    return called


def _tabularius_params():
    return [
        pytest.param(
            (m, Prompts.TABULARIUS),
            id=f"model={m}"
        )
        for m in TOOL_USE_MODELS
    ]


async def _run_case(tabularius: Tabularius, case: TabulariusCase):
    deps = make_eval_deps(case.directive)
    async with asyncio.timeout(300):
        result = await tabularius.agent.run(
            user_prompt=(
                f"Research Directives:\n{case.directive}\n\n"
                f"IMPORTANT: Complete ONLY the above directives. "
                f"Do not research unrelated topics."
            ),
            deps=deps,
            model_settings=AgentsConfiguration.TABULARIUS.model_settings,
        )
    return result, deps

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tabularius", _tabularius_params(), indirect=True)
async def test_calls_legislation_tool(tabularius: Tabularius, judge: EvaluatorAgent):
    """Must call search_legislation when explicitly directed."""
    result, _ = await _run_case(tabularius, LEGISLATION_SEARCH)
    called = _get_called_tools(result)
    assert "search_legislation" in called, (
        f"Expected search_legislation but called: {called}\nOutput:\n{result.output}"
    )


@pytest.mark.parametrize("tabularius", _tabularius_params(), indirect=True)
async def test_calls_court_cases_tool(tabularius: Tabularius, judge: EvaluatorAgent):
    """Must call search_court_cases when explicitly directed."""
    result, _ = await _run_case(tabularius, COURT_CASES_SEARCH)
    called = _get_called_tools(result)
    assert "search_court_cases" in called, (
        f"Expected search_court_cases but called: {called}\nOutput:\n{result.output}"
    )


@pytest.mark.parametrize("tabularius", _tabularius_params(), indirect=True)
async def test_rss_two_step_lists_then_fetches(tabularius: Tabularius, judge: EvaluatorAgent):
    """Must call list_gov_rss_feeds followed by get_gov_rss_feed."""
    result, _ = await _run_case(tabularius, RSS_FEED_TWO_STEP)
    called = _get_called_tools(result)
    assert "list_gov_rss_feeds" in called, (
        f"Never called list_gov_rss_feeds. Called: {called}\nOutput:\n{result.output}"
    )
    assert "get_gov_rss_feed" in called, (
        f"Listed feeds but never fetched one. Called: {called}\nOutput:\n{result.output}"
    )
    eval_result = await judge.evaluate(RSS_TWO_STEP_CRITERION, result.output)
    assert eval_result.passed, (
        f"Score: {eval_result.score}\nReasoning: {eval_result.reasoning}\n"
        f"Output:\n{result.output}"
    )


@pytest.mark.parametrize("tabularius", _tabularius_params(), indirect=True)
async def test_calls_fec_tool(tabularius: Tabularius, judge: EvaluatorAgent):
    """Must call search_candidate_finance when explicitly directed."""
    result, _ = await _run_case(tabularius, FEC_FINANCE_SEARCH)
    called = _get_called_tools(result)
    assert "search_candidate_finance" in called, (
        f"Expected search_candidate_finance but called: {called}\nOutput:\n{result.output}"
    )


@pytest.mark.parametrize("tabularius", _tabularius_params(), indirect=True)
async def test_legislation_output_is_relevant(tabularius: Tabularius, judge: EvaluatorAgent):
    """Output must contain substantive legislative findings."""
    result, _ = await _run_case(tabularius, LEGISLATION_SEARCH)
    called = _get_called_tools(result)
    if not called:
        pytest.skip("No tools called — covered by test_calls_legislation_tool")
    eval_result = await judge.evaluate(RESEARCH_RELEVANCE_CRITERION, result.output)
    assert eval_result.passed, (
        f"Score: {eval_result.score}\nReasoning: {eval_result.reasoning}\n"
        f"Output:\n{result.output}"
    )


@pytest.mark.parametrize("tabularius", _tabularius_params(), indirect=True)
async def test_legislation_tool_args_are_meaningful(tabularius: Tabularius, judge: EvaluatorAgent):
    """Args passed to search_legislation must be a meaningful query, not empty or generic."""
    result, _ = await _run_case(tabularius, LEGISLATION_SEARCH)
    calls = get_calls_for_tool(result, "search_legislation")
    if not calls:
        pytest.skip("search_legislation not called — covered by test_calls_legislation_tool")

    for inv in calls:
        query = inv.args.get("query", "")
        assert query, f"search_legislation called with missing/empty query. Args: {inv.args}"
        prompt = (
            f"Directive: {LEGISLATION_SEARCH.directive}\n"
            f"Tool called: search_legislation\n"
            f"Args passed: {inv.args}"
        )
        eval_result = await judge.evaluate(TOOL_ARGS_QUALITY_CRITERION, prompt)
        assert eval_result.passed, (
            f"Args look poor.\nArgs: {inv.args}\n"
            f"Score: {eval_result.score}\nReason: {eval_result.reasoning}"
        )


@pytest.mark.parametrize("tabularius", _tabularius_params(), indirect=True)
async def test_tool_returns_contain_data(tabularius: Tabularius, judge: EvaluatorAgent):
    """All called tools must return non-empty data (not [] or error)."""
    result, _ = await _run_case(tabularius, LEGISLATION_SEARCH)
    invocations = get_tool_invocations(result)
    if not invocations:
        pytest.skip("No tools called — covered by test_calls_legislation_tool")

    for inv in invocations:
        assert tool_return_looks_successful(inv), (
            f"{inv.tool_name} returned empty or error.\n"
            f"Args: {inv.args}\nReturn: {inv.return_content[:300] if inv.return_content else None}"
        )
