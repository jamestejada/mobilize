from __future__ import annotations

import re
import pytest

from src.ai import Nuntius
from src.settings import Prompts
from tests.eval.conftest import REFLECT_WRITE_MODELS, WRITER_SETTINGS_VARIANTS, WRITER_PROMPT_VARIANTS
from tests.eval.evaluator import EvaluatorAgent
from tests.eval.fixtures.nuntius_cases import WELL_SOURCED_INPUT, HIGH_SOURCE_COUNT, NuntiusCase

pytestmark = [pytest.mark.eval, pytest.mark.asyncio]

# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

CITATION_DENSITY_CRITERION = """
Every factual sentence in the response must contain at least one [SOURCE_N] tag
(e.g., [SOURCE_1], [SOURCE_2]).
FAIL if any sentence containing a factual claim has no [SOURCE_N] tag.
FAIL if there are three or more consecutive factual sentences with no source tag.
Score 0.0 if there are zero source tags in the entire response.
PASS if every factual claim is followed by a source tag.
"""

NO_RAW_URL_CRITERION = """
The response must not contain any raw HTTP or HTTPS URLs.
FAIL if the response contains any text matching 'http://' or 'https://'.
PASS if the response uses only [SOURCE_N] placeholder tags for references.
"""

RELEVANCE_CRITERION = """
The response must directly answer the user's query using the provided source data.
FAIL if the response discusses topics unrelated to the query.
FAIL if the response claims it could not find information when source data was provided.
PASS if the response addresses the query and draws on the source data.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_research(case: NuntiusCase) -> str:
    return f"{case.source_data}\n\n{case.research_findings}"


def _nuntius_params(models=REFLECT_WRITE_MODELS, prompts=WRITER_PROMPT_VARIANTS):
    return [
        pytest.param((model, prompt.values[0]), id=f"model={model},{prompt.id}")
        for model in models
        for prompt in prompts
    ]

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nuntius", _nuntius_params(), indirect=True)
@pytest.mark.parametrize("writer_settings", WRITER_SETTINGS_VARIANTS, indirect=True)
async def test_citation_density(nuntius: Nuntius, writer_settings: dict, judge: EvaluatorAgent):
    """Well-sourced input must produce dense [SOURCE_N] citations."""
    output = await nuntius.write(
        WELL_SOURCED_INPUT.query,
        _build_research(WELL_SOURCED_INPUT),
        model_settings=writer_settings,
    )
    result = await judge.evaluate(CITATION_DENSITY_CRITERION, output)
    assert result.passed, (
        f"Score: {result.score}\nReasoning: {result.reasoning}\n"
        f"Violations: {result.violations}\nOutput:\n{output}"
    )


@pytest.mark.parametrize("nuntius", _nuntius_params(), indirect=True)
@pytest.mark.parametrize("writer_settings", WRITER_SETTINGS_VARIANTS, indirect=True)
async def test_no_raw_urls(nuntius: Nuntius, writer_settings: dict, judge: EvaluatorAgent):
    """Nuntius must never emit raw https:// URLs."""
    output = await nuntius.write(
        HIGH_SOURCE_COUNT.query,
        _build_research(HIGH_SOURCE_COUNT),
        model_settings=writer_settings,
    )
    raw_urls = re.findall(r'https?://\S+', output)
    assert not raw_urls, f"Raw URLs in output: {raw_urls}\n{output}"

    result = await judge.evaluate(NO_RAW_URL_CRITERION, output)
    assert result.passed, f"Judge: {result.reasoning}\nOutput:\n{output}"


@pytest.mark.parametrize("nuntius", _nuntius_params(), indirect=True)
@pytest.mark.parametrize("writer_settings", WRITER_SETTINGS_VARIANTS, indirect=True)
async def test_relevance(nuntius: Nuntius, writer_settings: dict, judge: EvaluatorAgent):
    """Response must address the query and draw on source data."""
    output = await nuntius.write(
        WELL_SOURCED_INPUT.query,
        _build_research(WELL_SOURCED_INPUT),
        model_settings=writer_settings,
    )
    result = await judge.evaluate(RELEVANCE_CRITERION, output)
    assert result.passed, f"Score: {result.score}\nReasoning: {result.reasoning}\nOutput:\n{output}"
