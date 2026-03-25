from __future__ import annotations

import asyncio
import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart

from src.ai import Explorator
from src.agent_settings import AgentsConfiguration
from tests.eval.conftest import TOOL_USE_MODELS, EXPLORATOR_PROMPT_VARIANTS, make_eval_deps
from tests.eval.evaluator import EvaluatorAgent
from tests.eval.tool_helpers import get_calls_for_tool, get_tool_invocations, tool_return_looks_successful
from tests.eval.fixtures.explorator_cases import (
    WEB_NEWS_SEARCH,
    WIKIPEDIA_LOOKUP,
    REDDIT_SEARCH,
    MULTI_TOOL_FOLLOWUP,
    ExploratorCase,
)

pytestmark = [pytest.mark.eval, pytest.mark.asyncio]

# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

TOOL_ARGS_QUALITY_CRITERION = """
You are given a research directive and the arguments a model passed to a search tool.
PASS if the query argument is a specific, meaningful search string relevant to the directive.
FAIL if the query argument is empty, generic (e.g. just "query"), or unrelated to the directive.
FAIL if the query argument is longer than 150 characters (likely a hallucinated prompt dump).
"""

RESEARCH_RELEVANCE_CRITERION = """
The agent was given a research directive and returned findings.
PASS if the output text contains substantive information relevant to the directive topic.
FAIL if the output is empty, off-topic, or just repeats the directive without any findings.
FAIL if the output says it could not find anything when a tool was available.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_called_tools(result) -> set[str]:
    """Extract tool names called during an agent run from message history."""
    called = set()
    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    called.add(part.tool_name)
    return called


def _explorator_params():
    return [
        pytest.param((model, prompt.values[0]), id=f"model={model},{prompt.id}")
        for model in TOOL_USE_MODELS
        for prompt in EXPLORATOR_PROMPT_VARIANTS
    ]


async def _run_case(explorator: Explorator, case: ExploratorCase):
    deps = make_eval_deps(case.directive)
    async with asyncio.timeout(300):
        result = await explorator.agent.run(
            user_prompt=(
                f"Research Directives:\n{case.directive}\n\n"
                f"IMPORTANT: Complete ONLY the above directives. "
                f"Do not research unrelated topics."
            ),
            deps=deps,
            model_settings=AgentsConfiguration.EXPLORATOR.model_settings,
        )
    return result, deps

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("explorator", _explorator_params(), indirect=True)
async def test_calls_at_least_one_tool(explorator: Explorator, judge: EvaluatorAgent):
    """Model must invoke at least one tool rather than responding without tools."""
    result, _ = await _run_case(explorator, WEB_NEWS_SEARCH)
    called = _get_called_tools(result)
    assert called, (
        f"No tools called for directive: '{WEB_NEWS_SEARCH.directive}'\n"
        f"Output:\n{result.output}"
    )


@pytest.mark.parametrize("explorator", _explorator_params(), indirect=True)
async def test_calls_expected_tool_explicit(explorator: Explorator, judge: EvaluatorAgent):
    """When the directive names a specific tool, that tool must be called."""
    result, _ = await _run_case(explorator, WIKIPEDIA_LOOKUP)
    called = _get_called_tools(result)
    assert "search_wikipedia" in called, (
        f"Expected search_wikipedia but called: {called}\nOutput:\n{result.output}"
    )


@pytest.mark.parametrize("explorator", _explorator_params(), indirect=True)
async def test_research_output_is_relevant(explorator: Explorator, judge: EvaluatorAgent):
    """Output must contain substantive findings relevant to the directive."""
    result, _ = await _run_case(explorator, WEB_NEWS_SEARCH)
    called = _get_called_tools(result)
    # Only judge relevance if tools were actually called — tool invocation is a separate test
    if not called:
        pytest.skip("No tools called — covered by test_calls_at_least_one_tool")
    eval_result = await judge.evaluate(RESEARCH_RELEVANCE_CRITERION, result.output)
    assert eval_result.passed, (
        f"Score: {eval_result.score}\nReasoning: {eval_result.reasoning}\n"
        f"Output:\n{result.output}"
    )


@pytest.mark.parametrize("explorator", _explorator_params(), indirect=True)
async def test_reddit_tool_invocation(explorator: Explorator, judge: EvaluatorAgent):
    """Model must call search_reddit_history when explicitly directed."""
    result, _ = await _run_case(explorator, REDDIT_SEARCH)
    called = _get_called_tools(result)
    assert "search_reddit_history" in called, (
        f"Expected search_reddit_history but called: {called}\nOutput:\n{result.output}"
    )


@pytest.mark.parametrize("explorator", _explorator_params(), indirect=True)
async def test_multi_tool_chains_from_news(explorator: Explorator, judge: EvaluatorAgent):
    """Model must at minimum call search_news for a multi-tool directive."""
    result, _ = await _run_case(explorator, MULTI_TOOL_FOLLOWUP)
    called = _get_called_tools(result)
    assert "search_news" in called, (
        f"Expected search_news but called: {called}\nOutput:\n{result.output}"
    )


@pytest.mark.parametrize("explorator", _explorator_params(), indirect=True)
async def test_search_args_are_meaningful(explorator: Explorator, judge: EvaluatorAgent):
    """Query args passed to search tools must be specific and relevant, not empty or generic."""
    result, _ = await _run_case(explorator, WEB_NEWS_SEARCH)
    invocations = get_tool_invocations(result)
    search_calls = [i for i in invocations if i.tool_name in ("search_news", "search_web")]
    if not search_calls:
        pytest.skip("No search tools called — covered by test_calls_at_least_one_tool")

    for inv in search_calls:
        query = inv.args.get("query", "")
        assert query, f"{inv.tool_name} called with missing/empty query arg. Args: {inv.args}"
        prompt = (
            f"Directive: {WEB_NEWS_SEARCH.directive}\n"
            f"Tool called: {inv.tool_name}\n"
            f"Args passed: {inv.args}"
        )
        eval_result = await judge.evaluate(TOOL_ARGS_QUALITY_CRITERION, prompt)
        assert eval_result.passed, (
            f"{inv.tool_name} args look poor.\n"
            f"Args: {inv.args}\nScore: {eval_result.score}\nReason: {eval_result.reasoning}"
        )


@pytest.mark.parametrize("explorator", _explorator_params(), indirect=True)
async def test_tool_returns_contain_data(explorator: Explorator, judge: EvaluatorAgent):
    """Tools that were called must return non-empty data (not [] or error)."""
    result, _ = await _run_case(explorator, WEB_NEWS_SEARCH)
    invocations = get_tool_invocations(result)
    if not invocations:
        pytest.skip("No tools called — covered by test_calls_at_least_one_tool")

    for inv in invocations:
        assert tool_return_looks_successful(inv), (
            f"{inv.tool_name} returned empty or error.\n"
            f"Args: {inv.args}\nReturn: {inv.return_content[:300] if inv.return_content else None}"
        )
