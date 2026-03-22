from __future__ import annotations

import pytest

from src.ai import Cogitator
from src.settings import Prompts
from tests.eval.conftest import REFLECT_WRITE_MODELS, REVIEWER_SETTINGS_VARIANTS, REFLECTION_PROMPT_VARIANTS
from tests.eval.evaluator import EvaluatorAgent
from tests.eval.fixtures.cogitator_cases import (
    GOOD_DRAFT,
    DRAFT_WITH_RAW_URL,
    DRAFT_MISSING_CITATIONS,
    DRAFT_INTERNALLY_CONSISTENT,
)

pytestmark = [pytest.mark.eval, pytest.mark.asyncio]

# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

FORMAT_CRITERION = """
A valid Cogitator response must begin with exactly one of:
  - 'APPROVED: <reason>'
  - 'IMPROVE: <feedback>'
FAIL if the response begins with any other prefix.
FAIL if the response contains both 'APPROVED' and 'IMPROVE' as leading prefixes.
PASS if the first non-whitespace word is exactly 'APPROVED:' or 'IMPROVE:'.
"""

REJECTS_RAW_URL_CRITERION = """
The draft being reviewed contains a raw https:// URL, which violates citation rules.
PASS if the response begins with 'IMPROVE:' and mentions raw URLs or http.
FAIL if the response begins with 'APPROVED:'.
"""

REJECTS_MISSING_CITATIONS_CRITERION = """
The draft has three consecutive factual sentences before the first source tag.
The reviewer should flag the missing citations.
PASS if the response begins with 'IMPROVE:' and references missing source tags or citation density.
FAIL if the response begins with 'APPROVED:'.
"""

APPROVES_GOOD_DRAFT_CRITERION = """
The draft has correct [SOURCE_N] citations on every factual claim.
PASS if the response begins with 'APPROVED:'.
FAIL if the response begins with 'IMPROVE:' — this draft does not need improvement.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cogitator_params(models=REFLECT_WRITE_MODELS, prompts=REFLECTION_PROMPT_VARIANTS):
    return [
        pytest.param((m, p.values[0]), id=f"model={m},{p.id}")
        for m in models
        for p in prompts
    ]

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cogitator", _cogitator_params(), indirect=True)
@pytest.mark.parametrize("reviewer_settings", REVIEWER_SETTINGS_VARIANTS, indirect=True)
async def test_output_format_good_draft(
    cogitator: Cogitator, reviewer_settings: dict, judge: EvaluatorAgent
):
    """Output must always start with APPROVED: or IMPROVE:."""
    output = await cogitator.review(GOOD_DRAFT.query, GOOD_DRAFT.draft, reviewer_settings)
    result = await judge.evaluate(FORMAT_CRITERION, output)
    assert result.passed, f"Format violation.\nOutput: {output[:300]}\nReason: {result.reasoning}"


@pytest.mark.parametrize("cogitator", _cogitator_params(), indirect=True)
@pytest.mark.parametrize("reviewer_settings", REVIEWER_SETTINGS_VARIANTS, indirect=True)
async def test_output_format_bad_draft(
    cogitator: Cogitator, reviewer_settings: dict, judge: EvaluatorAgent
):
    """Output must always start with APPROVED: or IMPROVE: even on bad input."""
    output = await cogitator.review(DRAFT_WITH_RAW_URL.query, DRAFT_WITH_RAW_URL.draft, reviewer_settings)
    result = await judge.evaluate(FORMAT_CRITERION, output)
    assert result.passed, f"Format violation.\nOutput: {output[:300]}\nReason: {result.reasoning}"


@pytest.mark.parametrize("cogitator", _cogitator_params(), indirect=True)
@pytest.mark.parametrize("reviewer_settings", REVIEWER_SETTINGS_VARIANTS, indirect=True)
async def test_rejects_raw_url(
    cogitator: Cogitator, reviewer_settings: dict, judge: EvaluatorAgent
):
    """Must flag drafts containing raw URLs."""
    output = await cogitator.review(DRAFT_WITH_RAW_URL.query, DRAFT_WITH_RAW_URL.draft, reviewer_settings)
    result = await judge.evaluate(REJECTS_RAW_URL_CRITERION, output)
    assert result.passed, f"Failed to reject raw URL draft.\nOutput: {output}\nReason: {result.reasoning}"


@pytest.mark.parametrize("cogitator", _cogitator_params(), indirect=True)
@pytest.mark.parametrize("reviewer_settings", REVIEWER_SETTINGS_VARIANTS, indirect=True)
async def test_rejects_missing_citations(
    cogitator: Cogitator, reviewer_settings: dict, judge: EvaluatorAgent
):
    """Must flag drafts with multiple uncited factual sentences."""
    output = await cogitator.review(
        DRAFT_MISSING_CITATIONS.query, DRAFT_MISSING_CITATIONS.draft, reviewer_settings
    )
    result = await judge.evaluate(REJECTS_MISSING_CITATIONS_CRITERION, output)
    assert result.passed, f"Failed to flag missing citations.\nOutput: {output}\nReason: {result.reasoning}"


@pytest.mark.parametrize("cogitator", _cogitator_params(), indirect=True)
@pytest.mark.parametrize("reviewer_settings", REVIEWER_SETTINGS_VARIANTS, indirect=True)
async def test_approves_good_draft(
    cogitator: Cogitator, reviewer_settings: dict, judge: EvaluatorAgent
):
    """Must not flag well-cited drafts as needing improvement."""
    output = await cogitator.review(GOOD_DRAFT.query, GOOD_DRAFT.draft, reviewer_settings)
    result = await judge.evaluate(APPROVES_GOOD_DRAFT_CRITERION, output)
    assert result.passed, f"Incorrectly rejected a good draft.\nOutput: {output}\nReason: {result.reasoning}"
