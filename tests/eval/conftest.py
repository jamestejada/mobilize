from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
import asyncio
import pytest
from pathlib import Path
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from src.ai import (
    Nuntius, Cogitator, Probator, Explorator, Tabularius,
    AgentDeps,
)
from src.agent_settings import AgentsConfiguration
from src.settings import OllamaEndpoints, OLLAMA_NUM_CTX, PROMPT_PATH
from src.ollama_transport import ollama_http_client
from src.source_registry import SourceRegistry
from tests.eval.evaluator import EvaluatorAgent


# ---------------------------------------------------------------------------
# Incremental writer plugin
# ---------------------------------------------------------------------------

class _IncrementalWriter:
    """Writes eval_partial.json after every test so partial results survive a crash."""

    def __init__(self):
        self._results: list[dict] = []
        self._path = Path("logs/eval/eval_partial.json")
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @pytest.hookimpl
    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            entry = {
                "nodeid": report.nodeid,
                "outcome": "passed" if report.passed else ("skipped" if report.skipped else "failed"),
            }
            if report.failed:
                entry["longrepr"] = str(report.longrepr)
            self._results.append(entry)
            self._path.write_text(json.dumps({"partial": True, "tests": self._results}, indent=2))
        elif report.when == "setup" and report.skipped:
            self._results.append({"nodeid": report.nodeid, "outcome": "skipped"})
            self._path.write_text(json.dumps({"partial": True, "tests": self._results}, indent=2))


def pytest_configure(config):
    config.pluginmanager.register(_IncrementalWriter(), "_incremental_writer")


def pytest_addoption(parser):
    parser.addoption(
        "--eval-repeats",
        type=int,
        default=1,
        help="Run each eval-marked test this many times (independent samples per "
             "model/prompt/settings cell). Judge-based (LLM-graded) tests are noisy at "
             "n=1 — use e.g. --eval-repeats=5 before trusting a pass-rate comparison.",
    )


def pytest_generate_tests(metafunc):
    """Add a hidden 'repeat' dimension to every eval test, without editing each test file.

    Appends a `repeat=runN` segment to the test id (matching the existing
    `model=...,prompt=...` id convention used throughout tests/eval) so the report
    tooling can tell how many independent samples back a given pass rate.
    """
    n = metafunc.config.getoption("--eval-repeats")
    if n <= 1:
        return
    if not any(metafunc.definition.iter_markers("eval")):
        return
    already_parametrized = "repeat" in metafunc.fixturenames
    if already_parametrized:
        return  # already parametrized explicitly by the test
    # The test function itself never references `repeat` — force it into the fixture
    # closure (this is the same trick pytest-repeat uses) so parametrize can attach a
    # hidden dimension to the id without every eval test needing to declare the arg.
    metafunc.fixturenames.append("repeat")
    metafunc.parametrize("repeat", range(n), ids=[f"run{i}" for i in range(n)], indirect=True)


@pytest.fixture
def repeat(request):
    """Phantom fixture: exists only so pytest_generate_tests can parametrize eval
    tests by repeat count without every test needing to declare the argument."""
    return request.param


# ---------------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------------

GEMMA4_MODEL = "gemma4:latest"

REFLECT_WRITE_MODELS = [
    "qwen2.5:14b-instruct-q4_k_m",
    "qwen3:14b",
    "cogito:14b",
    "mistral:latest",
    "qwen3.5:latest",
    GEMMA4_MODEL,
]

COORDINATE_MODELS = [
    "qwen3:8b-q4_K_M",
    "qwen3:14b",
    "cogito:14b",
    "mistral:latest",
    "qwen3.5:latest",
    GEMMA4_MODEL,
]

TOOL_USE_MODELS = [
    "qwen2.5:14b-instruct-q4_k_m",
    "qwen3:8b-q4_K_M",
    "qwen3:14b",
    "hermes3:8b",
    "ministral-3:8b",
    "mannix/llama3.1-8b-lexi:tools-q6_k",
    "llama3-groq-tool-use:8b",
    "qwen3.5:latest",
    "cogito:14b",
    GEMMA4_MODEL,
]


# ---------------------------------------------------------------------------
# Hyperparameter variants
# ---------------------------------------------------------------------------

def _write_settings(temperature: float, top_p: float, think: bool = True) -> dict:
    return {
        "extra_body": {
            "think": think,
            "options": {
                "num_ctx": OLLAMA_NUM_CTX,
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": 1.1,
                "kv_cache_type": "q8_0",
            },
        }
    }


def _tool_settings(temperature: float, top_p: float, think: bool = False) -> dict:
    return {
        "extra_body": {
            "think": think,
            "options": {
                "num_ctx": OLLAMA_NUM_CTX,
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": 1.1,
                "kv_cache_type": "q8_0",
            },
        }
    }


# `think=False` is the single biggest latency lever for the qwen3/qwen3.5 reasoning
# models (qwen3.5 averages 60-200s/call with thinking on) but was never swept in any
# eval run before this. Add no-think variants alongside the existing temp/top_p sweeps
# so a future run can show whether disabling it costs any accuracy.
WRITER_SETTINGS_VARIANTS = [
    pytest.param(AgentsConfiguration.NUNTIUS.model_settings, id="temp=0.35,top_p=0.9,think=default"),
    pytest.param(_write_settings(0.1, 0.7), id="temp=0.1,top_p=0.7,think=True"),
    pytest.param(_write_settings(0.6, 0.95), id="temp=0.6,top_p=0.95,think=True"),
    pytest.param(_write_settings(0.35, 0.9, think=False), id="temp=0.35,top_p=0.9,think=False"),
]

REVIEWER_SETTINGS_VARIANTS = [
    pytest.param(AgentsConfiguration.COGITATOR.model_settings, id="temp=0.1,top_p=0.7,think=default"),
    pytest.param(_tool_settings(0.3, 0.85), id="temp=0.3,top_p=0.85,think=False"),
    pytest.param(_tool_settings(0.3, 0.85, think=True), id="temp=0.3,top_p=0.85,think=True"),
]


# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------

def _prompt_variants(agent_name: str) -> list:
    """Collect all prompt files matching {agent_name}*.md, using filename as the param id."""
    return [
        pytest.param(f.read_text(), id=f"prompt={f.name}")
        for f in sorted(PROMPT_PATH.glob(f"{agent_name}*.md"))
    ]


def _prompt_variants_exact(*filenames: str) -> list:
    """Collect specific prompt files by exact filename."""
    return [
        pytest.param(PROMPT_PATH.joinpath(filename).read_text(), id=f"prompt={filename}")
        for filename in filenames
    ]


WRITER_PROMPT_VARIANTS = _prompt_variants("writer")
REFLECTION_PROMPT_VARIANTS = _prompt_variants("reflection")
GAP_ANALYSIS_PROMPT_VARIANTS = _prompt_variants("gap_analysis")
EXPLORATOR_PROMPT_VARIANTS = _prompt_variants("explorator")
TABULARIUS_PROMPT_VARIANTS = _prompt_variants("tabularius")
COORDINATOR_PROMPT_VARIANTS = _prompt_variants("coordinator")
WRITER_GEMMA_PROMPT_VARIANTS = _prompt_variants_exact("writer_gemma.md")
COORDINATOR_GEMMA_PROMPT_VARIANTS = _prompt_variants_exact("coordinator_gemma.md")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_ollama_model(model_name: str) -> OpenAIChatModel:
    return OpenAIChatModel(
        model_name=model_name,
        provider=OllamaProvider(
            base_url=str(OllamaEndpoints.CHAT),
            http_client=ollama_http_client,
        ),
    )


# ---------------------------------------------------------------------------
# Fixtures
# Two-tuple param: (model_name, instructions)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def judge() -> EvaluatorAgent:
    return EvaluatorAgent()


@pytest.fixture
def nuntius(request) -> Nuntius:
    model_name, instructions = request.param
    return Nuntius(model=_make_ollama_model(model_name), instructions=instructions)


@pytest.fixture
def cogitator(request) -> Cogitator:
    model_name, instructions = request.param
    return Cogitator(model=_make_ollama_model(model_name), instructions=instructions)


@pytest.fixture
def probator(request) -> Probator:
    model_name, instructions = request.param
    return Probator(model=_make_ollama_model(model_name), instructions=instructions)


@pytest.fixture
def writer_settings(request) -> dict:
    return request.param


@pytest.fixture
def reviewer_settings(request) -> dict:
    return request.param


# ---------------------------------------------------------------------------
# Tool-use agent fixtures
# ---------------------------------------------------------------------------

def make_eval_deps(query: str = "eval") -> AgentDeps:
    """Minimal AgentDeps for eval tests — no-op update_chat, fresh registry."""
    async def _noop(msg: str) -> None:
        pass

    return AgentDeps(
        update_chat=_noop,
        user_input=query,
        chat_id=0,
        source_registry=SourceRegistry(),
    )


@pytest.fixture
def explorator(request) -> Explorator:
    model_name, instructions = request.param
    return Explorator(model=_make_ollama_model(model_name), instructions=instructions)


@pytest.fixture
def tabularius(request) -> Tabularius:
    model_name, instructions = request.param
    return Tabularius(model=_make_ollama_model(model_name), instructions=instructions)
