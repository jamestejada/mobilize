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
from src.settings import OllamaEndpoints, Prompts, OLLAMA_NUM_CTX, PROMPT_PATH
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


# ---------------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------------

REFLECT_WRITE_MODELS = [
    "qwen2.5:14b-instruct-q4_k_m",
    "qwen3:14b",
    "cogito:14b",
    "mistral:latest",
    "qwen3.5:latest",
]

COORDINATE_MODELS = [
    "qwen3:8b-q4_K_M",
    "qwen3:14b",
    "cogito:14b",
    "mistral:latest",
    "qwen3.5:latest",
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
]


# ---------------------------------------------------------------------------
# Hyperparameter variants
# ---------------------------------------------------------------------------

def _write_settings(temperature: float, top_p: float) -> dict:
    return {
        "extra_body": {
            "options": {
                "num_ctx": OLLAMA_NUM_CTX,
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": 1.1,
                "kv_cache_type": "q8_0",
            }
        }
    }


def _tool_settings(temperature: float, top_p: float) -> dict:
    return {
        "extra_body": {
            "think": False,
            "options": {
                "num_ctx": OLLAMA_NUM_CTX,
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": 1.1,
                "kv_cache_type": "q8_0",
            },
        }
    }


WRITER_SETTINGS_VARIANTS = [
    pytest.param(AgentsConfiguration.NUNTIUS.model_settings, id="temp=0.35,top_p=0.9"),
    pytest.param(_write_settings(0.1, 0.7), id="temp=0.1,top_p=0.7"),
    pytest.param(_write_settings(0.6, 0.95), id="temp=0.6,top_p=0.95"),
]

REVIEWER_SETTINGS_VARIANTS = [
    pytest.param(AgentsConfiguration.COGITATOR.model_settings, id="temp=0.1,top_p=0.7"),
    pytest.param(_tool_settings(0.3, 0.85), id="temp=0.3,top_p=0.85"),
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


WRITER_PROMPT_VARIANTS = _prompt_variants("writer")
REFLECTION_PROMPT_VARIANTS = _prompt_variants("reflection")
GAP_ANALYSIS_PROMPT_VARIANTS = _prompt_variants("gap_analysis")
EXPLORATOR_PROMPT_VARIANTS = _prompt_variants("explorator")
TABULARIUS_PROMPT_VARIANTS = _prompt_variants("tabularius")
COORDINATOR_PROMPT_VARIANTS = _prompt_variants("coordinator")


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
