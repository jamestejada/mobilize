import re
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
from typing import Callable, Awaitable, List

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior, ModelHTTPError
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from .settings import (
        Models,
        Prompts,
        OllamaEndpoints,
        OLLAMA_NUM_CTX
    )
from .chat_history import ChatHistoryManager
from .source_registry import SourceRegistry, SourceDataBuilder
from .ollama_transport import ollama_http_client

logger = logging.getLogger(__name__)



def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output.
    This is needed for qwen models.
    """
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    if '</think>' in text:
        text = text.split('</think>', 1)[-1]
    return text.strip()


#                      User Query
#                           │
#               ┌─────────────────────────┐
#               │   Praetor.agent.run()   │
#               │  (run_research tool)    │
#               └────────────┬────────────┘
#                            │
#               ┌────────────┴──────────────────┐
#               │ deps.research_findings?        │
#              YES                              NO
#               │                               │
#               ▼                               ▼
#     _write_and_review()            result.output delivered
#     [always runs — Python]         (clarifications, out-of-scope)
#               │
#      run_research routes internally:
#      ├── Explorator (web/social)
#      └── Tabularius (data/events)
#               │
#               ├── nuntius.write()
#               ├── cogitator.review()   ←─┐
#               │       │         │        │  (up to 3 iterations)
#               │    APPROVED   feedback ──┘
#               ├── source_registry.substitute()
#               └── update_chat(final)  → User Response



research_coordinate_model = OpenAIChatModel(
    model_name=Models.Research_Coordinate,
    provider=OllamaProvider(
        base_url=str(OllamaEndpoints.API_ROOT),
        http_client=ollama_http_client,
    )
)
reflect_write_model = OpenAIChatModel(
    model_name=Models.Reflect_Write,
    provider=OllamaProvider(
        base_url=str(OllamaEndpoints.API_ROOT),
        http_client=ollama_http_client,
    )
)

# Agents that must emit valid JSON tool calls — keep deterministic
OLLAMA_TOOL_SETTINGS = {
    "extra_body": {"options": {"num_ctx": OLLAMA_NUM_CTX, "temperature": 0.1, "think": False}}
}

# Writer only — prose benefits from slightly more variability
OLLAMA_WRITE_SETTINGS = {
    "extra_body": {"options": {"num_ctx": OLLAMA_NUM_CTX, "temperature": 0.3}}
}

MAX_SPECIAL_RETRIES = 2

def inject_date(agent: Agent) -> None:
    @agent.instructions
    def add_date() -> str:
        current_date = datetime.now(
            tz=ZoneInfo("UTC")
        ).strftime("%B %d, %Y")
        return f"Today's date is {current_date}"


def inject_tool_list(agent: Agent, toolset) -> None:
    """Append a dynamic tool list to agent instructions, generated from a toolset."""
    lines = []
    for ts in toolset.toolsets:
        for tool in ts.tools.values():
            fn = tool.function
            first_line = (fn.__doc__ or "").strip().split("\n")[0].rstrip(".")
            lines.append(f"- `{fn.__name__}`: {first_line}")
    tool_text = "\n".join(lines)

    @agent.instructions
    def _add_tool_list() -> str:
        return f"## Available Tools\n{tool_text}"


MAX_RESEARCH_CALLS = 5   # initial + 4 follow-ups
MAX_REVIEW_CALLS = 3     # write + up to 2 revisions
MAX_SPECIAL_RETRIES = 2     # retries for tool errors in Praetor before giving up



@dataclass
class CallCounter:
    max_calls: int
    count: int = 0

    def calls_exhausted(self) -> bool:
        """Increment the counter. Returns True if the limit has been exceeded."""
        self.count += 1
        return self.count > self.max_calls


@dataclass
class AgentDeps:
    update_chat: Callable[[str], Awaitable[None]]
    user_input: str = ""
    chat_id: int = 0
    research_findings: str = ""
    web_research_counter: CallCounter = field(
            default_factory=lambda: CallCounter(MAX_RESEARCH_CALLS)
        )
    data_research_counter: CallCounter = field(
            default_factory=lambda: CallCounter(MAX_RESEARCH_CALLS)
        )
    praetor_counter: CallCounter = field(
            default_factory=lambda: CallCounter(MAX_SPECIAL_RETRIES)
        )
    
    source_registry: SourceRegistry | None = None


class Explorator:
    """Web and Social Media Research Agent"""
    def __init__(self):
        from .tools import EXPLORATOR_AGENT_TOOLSET, EXPLORATOR_TOOLSET
        self.tool_names = EXPLORATOR_TOOLSET.tools
        self.agent = Agent(
            model=research_coordinate_model,
            deps_type=AgentDeps,
            instructions=Prompts.EXPLORATOR,
            toolsets=[EXPLORATOR_AGENT_TOOLSET],
            model_settings=OLLAMA_TOOL_SETTINGS,
            retries=2,
        )
        inject_date(agent=self.agent)
        inject_tool_list(self.agent, EXPLORATOR_AGENT_TOOLSET)

    def should_handle(self, directive: str) -> bool:
        return any(name in directive for name in self.tool_names)

    async def run(self, directive: str, deps: AgentDeps):
        """Run web/social research
        Returns None if the agent fails with an unrecoverable tool error.
        """
        while not deps.web_research_counter.calls_exhausted():
            try:
                return await self.agent.run(
                    user_prompt=(
                        f"Research Directives:\n{directive}\n\n"
                        f"IMPORTANT: Complete ONLY the above directives. "
                        f"Do not research unrelated topics."
                    ),
                    deps=deps,
                    model_settings=OLLAMA_TOOL_SETTINGS,
                )
            except (UnexpectedModelBehavior, ModelHTTPError) as e:
                logger.warning(f"Explorator failure: {e}")
        else:
            return None


class Tabularius:
    """Structured Data Research Agent"""
    def __init__(self):
        from .tools import TABULARIUS_AGENT_TOOLSET, TABULARIUS_TOOLSET
        self.tool_names = TABULARIUS_TOOLSET.tools
        self.agent = Agent(
            model=research_coordinate_model,
            deps_type=AgentDeps,
            instructions=Prompts.TABULARIUS,
            toolsets=[TABULARIUS_AGENT_TOOLSET],
            model_settings=OLLAMA_TOOL_SETTINGS,
            retries=2,
        )
        inject_date(agent=self.agent)
        inject_tool_list(self.agent, TABULARIUS_AGENT_TOOLSET)

    def should_handle(self, directive: str) -> bool:
        return any(name in directive for name in self.tool_names)

    async def run(self, directive: str, deps: AgentDeps):
        """Run structured-data research.
        Returns None if the agent fails with an unrecoverable tool error.
        """
        while not deps.data_research_counter.calls_exhausted():
            try:
                return await self.agent.run(
                    user_prompt=(
                        f"Research Directives:\n{directive}\n\n"
                        f"IMPORTANT: Complete ONLY the above directives. "
                        f"Do not research unrelated topics."
                    ),
                    deps=deps,
                    model_settings=OLLAMA_TOOL_SETTINGS,
                )
            except (UnexpectedModelBehavior, ModelHTTPError) as e:
                logger.warning(f"Tabularius failure: {e}")
        else:
            return None


class Nuntius:
    """Report/Response generator
    """
    def __init__(self):
        self.agent = Agent(
            model=reflect_write_model,
            instructions=Prompts.WRITER
        )
        inject_date(agent=self.agent)

    async def write(self, user_input: str, research: str) -> str:
        """Write a sourced response from research findings."""
        result = await self.agent.run(
            user_prompt=f"User Question: {user_input}\n\n{research}",
            model_settings=OLLAMA_WRITE_SETTINGS,
        )
        output = strip_think_tags(result.output)
        return output


class Cogitator:
    """Reflection agent
    """
    def __init__(self):
        self.agent = Agent(
            model=reflect_write_model,
            instructions=Prompts.REFLECTION
        )
        inject_date(agent=self.agent)

    async def review(self, user_input: str, draft: str) -> str:
        """Review a draft response; returns 'APPROVED' or improvement feedback."""
        result = await self.agent.run(
            user_prompt=f"Original Question: {user_input}\n\nDraft Response:\n{draft}",
            model_settings=OLLAMA_TOOL_SETTINGS,
        )
        return strip_think_tags(result.output)


class Praetor:
    """Coordination Agent
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.history = ChatHistoryManager()
        self._explorator: Explorator | None = None
        self._tabularius: Tabularius | None = None
        self._nuntius: Nuntius | None = None
        self._cogitator: Cogitator | None = None
        self.source_builder = SourceDataBuilder()
        self._registries: dict[int, SourceRegistry] = {}
        from .tools import ALL_RESEARCH_TOOLSET
        self.agent = Agent(
            model=research_coordinate_model,
            instructions=Prompts.COORDINATOR,
            deps_type=AgentDeps,
            output_type=str,
            tools=[
                self.run_research,
                self.get_sources,
            ],
            retries=2,
        )
        inject_date(self.agent)
        inject_tool_list(self.agent, ALL_RESEARCH_TOOLSET)

    def _get_registry(self, chat_id: int) -> SourceRegistry:
        if chat_id not in self._registries:
            self._registries[chat_id] = SourceRegistry()
        return self._registries[chat_id]

    def clear(self, chat_id: int) -> None:
        """Clear conversation history and source registry for a chat."""
        self.history.clear(chat_id)
        self._registries.pop(chat_id, None)

    @property
    def explorator(self) -> Explorator:
        if not self._explorator:
            self._explorator = Explorator()
        return self._explorator

    @property
    def tabularius(self) -> Tabularius:
        if not self._tabularius:
            self._tabularius = Tabularius()
        return self._tabularius

    @property
    def nuntius(self) -> Nuntius:
        if not self._nuntius:
            self._nuntius = Nuntius()
        return self._nuntius

    @property
    def cogitator(self) -> Cogitator:
        if not self._cogitator:
            self._cogitator = Cogitator()
        return self._cogitator

    async def run_research(self, ctx: RunContext[AgentDeps], directive: str) -> str:
        """Researches the directive using OSINT tools.

        Args:
            directive (str): What to investigate and which tools to use.

        Returns:
            str: Research findings and source data for the writer.
        """
        await ctx.deps.update_chat("_Researching..._")
        logger.debug(f"[Praetor→Research] directive:\n{directive}")

        use_explorator = self.explorator.should_handle(directive)
        use_tabularius = self.tabularius.should_handle(directive)
        if not use_explorator and not use_tabularius:
            use_explorator = True  # fallback: default to web search

        summaries: list[str] = []

        async def _run_agent(agent_runner, label: str) -> None:
            result = await agent_runner.run(directive, ctx.deps)
            if result is None:
                logger.warning(f"{label} returned no result")
                return
            logger.debug(f"[{label}] output:\n{result.output}")
            tool_data, _ = self.source_builder.build(
                result.all_messages(), self._get_registry(ctx.deps.chat_id)
            )
            ctx.deps.research_findings += (
                    f"\n\n{tool_data}\n\nResearch Findings "
                    f"({label}):\n{result.output}"
                )
            summaries.append(f"[{label}]:\n{result.output}")

        if use_explorator:
            await ctx.deps.update_chat("_Starting web/social research..._")
            await _run_agent(self.explorator, "Web/Social")
        if use_tabularius:
            await ctx.deps.update_chat("_Starting data/events research..._")
            await _run_agent(self.tabularius, "Data/Events")

        summary = "\n\n".join(summaries) if summaries else "No findings returned."
        return (
            f"Research complete.\n\n{summary}\n\n"
            "If the above reveals important leads (key people, orgs, breaking events) "
            "not yet fully investigated, call run_research again with a targeted follow-up "
            "directive. Otherwise stop."
        )

    async def get_sources(self, ctx: RunContext[AgentDeps]) -> str:
        """Returns sources already collected in this conversation.

        Use this when the user asks a follow-up question about a topic already
        researched. Review the returned sources to determine whether prior
        research is sufficient or whether a new run_research call is needed
        to capture recent developments. Cite [SOURCE_N] tags in your response;
        they are automatically resolved to real links before delivery.

        Returns:
            str: List of [SOURCE_N] tags with titles, URLs, and short descriptions,
                 or a message indicating no sources have been collected yet.
        """
        return self._get_registry(ctx.deps.chat_id).format_for_agent()

    async def _write_and_review(
                self,
                user_input: str,
                research: str,
                chat_id: int,
                update_chat,
            ) -> str:
        """Write, review, and return the final response text."""
        await update_chat("_Writing response..._")
        draft = await self.nuntius.write(user_input, research)
        logger.debug(f"[Nuntius] draft:\n{draft}")
        for _ in range(MAX_REVIEW_CALLS):
            await update_chat("_Reviewing..._")
            feedback = await self.cogitator.review(user_input, draft)
            logger.debug(f"[Cogitator] feedback:\n{feedback}")
            if "APPROVED" in feedback.upper():
                break
            await update_chat("_Revising..._")
            draft = await self.nuntius.write(
                user_input,
                f"{research}\n\nPrevious Draft:\n{draft}\n\nReviewer Feedback:\n{feedback}",
            )
            logger.debug(f"[Nuntius] revision:\n{draft}")
        return self._get_registry(chat_id).substitute(strip_think_tags(draft))

    async def handle_query(
                self,
                user_input: str,
                chat_id: int,
                update_chat: Callable[[str], Awaitable[None]]
            ):
        await update_chat("_Thinking..._")
        deps = AgentDeps(
            update_chat=update_chat,
            user_input=user_input,
            chat_id=chat_id,
            source_registry=self._get_registry(chat_id),
        )
        for attempt in range(MAX_SPECIAL_RETRIES + 1):
            try:
                result = await self.agent.run(
                    user_prompt=user_input,
                    message_history=self.history.get(chat_id),
                    deps=deps,
                    model_settings=OLLAMA_TOOL_SETTINGS,
                )
                logger.debug(f"[Praetor] output:\n{result.output}")
                self.history.update(chat_id, result.all_messages())
                if deps.research_findings:
                    # OSINT path — always goes through Nuntius+Cogitator
                    final = await self._write_and_review(
                        user_input, deps.research_findings, chat_id, update_chat
                    )
                    await update_chat(final)
                elif result.output:
                    # Direct path — clarifications, out-of-scope, simple answers
                    final = self._get_registry(chat_id).substitute(strip_think_tags(result.output))
                    await update_chat(final)
                return
            except (UnexpectedModelBehavior, ModelHTTPError) as e:
                self.logger.warning(f"Praetor failure (attempt {attempt + 1}): {e}", exc_info=True)
        await update_chat("_Something went wrong. Please try again._")
