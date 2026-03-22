from __future__ import annotations

import asyncio
import pytest

from src.ai import Probator
from src.settings import Prompts
from tests.eval.conftest import COORDINATE_MODELS, REVIEWER_SETTINGS_VARIANTS, GAP_ANALYSIS_PROMPT_VARIANTS
from tests.eval.evaluator import EvaluatorAgent
from tests.eval.fixtures.probator_cases import (
    COMPREHENSIVE_RESEARCH,
    MISSING_KEY_ANGLE,
    SINGLE_LOW_CONFIDENCE_SOURCE,
)

pytestmark = [pytest.mark.eval, pytest.mark.asyncio]

# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

FORMAT_CRITERION = """
A valid Probator response must begin with exactly one of:
  - 'ADEQUATE' — the research is complete
  - 'GAPS: <description>' followed by a 'SEARCH:' section with tool call lines
FAIL if the response begins with any other prefix.
FAIL if the response says 'GAPS:' without a following 'SEARCH:' section.
PASS if the output is 'ADEQUATE' or starts with 'GAPS:' and includes 'SEARCH:'.
"""

ADEQUATE_CRITERION = """
The research thoroughly covers the user's query from multiple angles with multiple sources.
PASS if the response is 'ADEQUATE' or begins with 'ADEQUATE'.
FAIL if the response begins with 'GAPS:' — the research is complete.
"""

DETECTS_ONE_SIDED_CRITERION = """
The research only covers proponents of a bill, but the user explicitly asked for 'both sides'.
The missing opposition coverage is a significant gap.
PASS if the response begins with 'GAPS:' and mentions the missing opposition or counter-arguments.
FAIL if the response is 'ADEQUATE' — there is a clear, significant gap.
"""

DETECTS_LOW_CONFIDENCE_CRITERION = """
An important factual claim (whether someone was arrested) rests on only
a single LOW-confidence source (a tweet) with no official confirmation found.
PASS if the response begins with 'GAPS:' referencing the need for official corroboration.
FAIL if the response is 'ADEQUATE' — the single LOW-confidence claim needs follow-up.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probator_params(models=COORDINATE_MODELS, prompts=GAP_ANALYSIS_PROMPT_VARIANTS):
    return [
        pytest.param((m, p.values[0]), id=f"model={m},{p.id}")
        for m in models
        for p in prompts
    ]

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("probator", _probator_params(), indirect=True)
@pytest.mark.parametrize("reviewer_settings", REVIEWER_SETTINGS_VARIANTS, indirect=True)
async def test_output_format(
    probator: Probator, reviewer_settings: dict, judge: EvaluatorAgent
):
    """Output must be ADEQUATE or GAPS: with a SEARCH: section."""
    # Call the agent directly (bypassing .analyze()'s None-return logic)
    from src.ai import strip_think_tags
    from src.agent_settings import AgentsConfiguration
    async with asyncio.timeout(300):
        result_obj = await probator.agent.run(
            user_prompt=(
                f"Original Question: {COMPREHENSIVE_RESEARCH.query}\n\n"
                f"Research Findings:\n{COMPREHENSIVE_RESEARCH.research_findings}"
            ),
            model_settings=reviewer_settings or AgentsConfiguration.PROBATOR.model_settings,
        )
    output = strip_think_tags(result_obj.output)
    result = await judge.evaluate(FORMAT_CRITERION, output)
    assert result.passed, f"Format violation.\nOutput: {output[:300]}\nReason: {result.reasoning}"


@pytest.mark.parametrize("probator", _probator_params(), indirect=True)
@pytest.mark.parametrize("reviewer_settings", REVIEWER_SETTINGS_VARIANTS, indirect=True)
async def test_adequate_for_complete_research(
    probator: Probator, reviewer_settings: dict, judge: EvaluatorAgent
):
    """Should return None (ADEQUATE) for thoroughly covered research."""
    gap = await probator.analyze(
        COMPREHENSIVE_RESEARCH.query,
        COMPREHENSIVE_RESEARCH.research_findings,
        model_settings=reviewer_settings,
    )
    assert gap is None, (
        f"Probator incorrectly flagged complete research as having gaps.\nGap returned:\n{gap}"
    )


@pytest.mark.parametrize("probator", _probator_params(), indirect=True)
@pytest.mark.parametrize("reviewer_settings", REVIEWER_SETTINGS_VARIANTS, indirect=True)
async def test_detects_one_sided_research(
    probator: Probator, reviewer_settings: dict, judge: EvaluatorAgent
):
    """Should identify missing opposition coverage for a 'both sides' query."""
    gap = await probator.analyze(
        MISSING_KEY_ANGLE.query,
        MISSING_KEY_ANGLE.research_findings,
        model_settings=reviewer_settings,
    )
    assert gap is not None, "Probator returned ADEQUATE but research was one-sided."
    result = await judge.evaluate(DETECTS_ONE_SIDED_CRITERION, gap)
    assert result.passed, f"Gap detected but didn't mention opposition.\nGap:\n{gap}\nReason: {result.reasoning}"


@pytest.mark.parametrize("probator", _probator_params(), indirect=True)
@pytest.mark.parametrize("reviewer_settings", REVIEWER_SETTINGS_VARIANTS, indirect=True)
async def test_detects_low_confidence_source(
    probator: Probator, reviewer_settings: dict, judge: EvaluatorAgent
):
    """Should flag important claims backed only by a single LOW-confidence source."""
    gap = await probator.analyze(
        SINGLE_LOW_CONFIDENCE_SOURCE.query,
        SINGLE_LOW_CONFIDENCE_SOURCE.research_findings,
        model_settings=reviewer_settings,
    )
    assert gap is not None, "Probator returned ADEQUATE for a LOW-confidence single-source claim."
    result = await judge.evaluate(DETECTS_LOW_CONFIDENCE_CRITERION, gap)
    assert result.passed, f"Gap detected but didn't flag low confidence.\nGap:\n{gap}\nReason: {result.reasoning}"
