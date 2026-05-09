from __future__ import annotations

import re
import pytest

from src.ai import Nuntius
from src.settings import Prompts
from tests.eval.conftest import (
    GEMMA4_MODEL,
    REFLECT_WRITE_MODELS,
    WRITER_GEMMA_PROMPT_VARIANTS,
    WRITER_SETTINGS_VARIANTS,
    WRITER_PROMPT_VARIANTS,
)
from tests.eval.evaluator import EvaluatorAgent
from tests.eval.fixtures.nuntius_cases import (
    ANDES_VIRUS_NARROW_QUERY,
    HIGH_SOURCE_COUNT,
    IRRELEVANT_SOURCE_ONLY,
    WELL_SOURCED_INPUT,
    NuntiusCase,
)

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

NO_FALLBACK_SYNTHESIS_CRITERION = """
The response must stay grounded in Source Data.
FAIL if it uses generic background knowledge, broad educational filler, or unsupported synthesis.
FAIL if it says things like "if you provide results", "if you share sources", or similar boilerplate.
PASS if it answers only what the user asked using the supplied sources.
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


def _gemma_nuntius_params():
    return [
        pytest.param((GEMMA4_MODEL, prompt.values[0]), id=f"model={GEMMA4_MODEL},{prompt.id}")
        for prompt in WRITER_GEMMA_PROMPT_VARIANTS
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


@pytest.mark.parametrize("nuntius", _gemma_nuntius_params(), indirect=True)
@pytest.mark.parametrize("writer_settings", WRITER_SETTINGS_VARIANTS[:1], indirect=True)
async def test_gemma_writer_answers_narrow_question_without_broad_overview(
    nuntius: Nuntius, writer_settings: dict, judge: EvaluatorAgent
):
    """Gemma prompt should answer a narrow question directly from Source Data."""
    output = await nuntius.write(
        ANDES_VIRUS_NARROW_QUERY.query,
        _build_research(ANDES_VIRUS_NARROW_QUERY),
        model_settings=writer_settings,
    )
    result = await judge.evaluate(NO_FALLBACK_SYNTHESIS_CRITERION, output)
    assert result.passed, f"Score: {result.score}\nReasoning: {result.reasoning}\nOutput:\n{output}"
    assert "[SOURCE_" in output, f"Expected grounded citation in output:\n{output}"
    assert "if you provide" not in output.lower(), output
    assert "hantavirus pulmonary syndrome" not in output.lower(), output


@pytest.mark.parametrize("nuntius", _gemma_nuntius_params(), indirect=True)
@pytest.mark.parametrize("writer_settings", WRITER_SETTINGS_VARIANTS[:1], indirect=True)
async def test_gemma_writer_returns_brief_insufficiency_when_relevant_sources_missing(
    nuntius: Nuntius, writer_settings: dict
):
    """Gemma prompt should avoid generic synthesis when no relevant Source Data exists."""
    output = await nuntius.write(
        IRRELEVANT_SOURCE_ONLY.query,
        _build_research(IRRELEVANT_SOURCE_ONLY),
        model_settings=writer_settings,
    )
    lowered = output.lower()
    assert "if you provide" not in lowered, output
    assert "weather" not in lowered, output
    assert "football" not in lowered, output
    assert len(output.splitlines()) <= 3, output


@pytest.mark.parametrize("nuntius", _gemma_nuntius_params(), indirect=True)
@pytest.mark.parametrize("writer_settings", WRITER_SETTINGS_VARIANTS[:1], indirect=True)
async def test_gemma_writer_andes_virus_regression_uses_only_source_data(
    nuntius: Nuntius, writer_settings: dict
):
    """Regression: Andes virus query should stay narrowly grounded in the provided sources."""
    output = await nuntius.write(
        ANDES_VIRUS_NARROW_QUERY.query,
        _build_research(ANDES_VIRUS_NARROW_QUERY),
        model_settings=writer_settings,
    )
    lowered = output.lower()
    assert "the available sources say" not in lowered, output
    assert "if you share" not in lowered, output
    assert "only hantavirus known to spread from person to person" in lowered or "[source_" in lowered, output
