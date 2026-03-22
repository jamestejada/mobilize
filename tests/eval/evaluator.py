from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from src.ollama_transport import ollama_http_client
from src.settings import OllamaEndpoints


class EvalResult(BaseModel):
    passed: bool
    score: float = Field(ge=0.0, le=1.0, description="0.0=failing, 1.0=perfect")
    reasoning: str = Field(description="Concise justification, max 3 sentences")
    violations: list[str] = Field(
        default_factory=list,
        description="Specific rules violated; empty if passed",
    )


_JUDGE_INSTRUCTIONS = """
You are an objective evaluator for an OSINT intelligence system's LLM outputs.
Given a criterion and an agent output, evaluate strictly against the criterion — not general quality.
You do NOT have current world knowledge; trust the agent's content as factual.
Score 0.0 to 1.0 where 0.5 is marginally acceptable, 1.0 is perfect, 0.0 is a clear violation.
Always respond in English. Output only the structured result — no preamble, no explanation outside the result fields.
""".strip()


@dataclass
class EvaluatorAgent:
    """Judge LLM that evaluates agent output against a plain-English criterion."""

    model_name: str = ""

    def __post_init__(self):
        name = self.model_name or os.getenv(
            "EVAL_JUDGE_MODEL", "qwen2.5:14b-instruct-q4_k_m"
        )
        model = OpenAIChatModel(
            model_name=name,
            provider=OllamaProvider(
                base_url=str(OllamaEndpoints.CHAT),
                http_client=ollama_http_client,
            ),
        )
        self._agent: Agent[None, EvalResult] = Agent(
            model=model,
            instructions=_JUDGE_INSTRUCTIONS,
            output_type=EvalResult,
            retries=3,
        )

    async def evaluate(self, criterion: str, agent_output: str) -> EvalResult:
        """
        Args:
            criterion: Plain-English rule with pass/fail definition.
            agent_output: Raw text output from the agent under test.
        """
        prompt = f"CRITERION:\n{criterion}\n\nAGENT OUTPUT:\n{agent_output}"
        async with asyncio.timeout(120):
            result = await self._agent.run(user_prompt=prompt)
        return result.output
