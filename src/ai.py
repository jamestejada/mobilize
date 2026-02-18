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
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)


#                      User Query
#                           │
#               ┌─────────────────────────┐
#               │   Praetor.agent.run()   │
#               │  (pydantic-ai tool loop)│
#               └────────────┬────────────┘
#                            │
#               ┌────────────┴─────────────┐
#               │                          │
#         OSINT query                  Out-of-scope
#               │                          │
#               ▼                          ▼
#     run_research(directive)       text response
#               │
#               ▼  ← can repeat if gaps found
#     write_response(research)←─┐
#               │               │
#               ▼               │   (up to 3 iterations)
#     review_response(draft)    │
#               │        │      │
#            APPROVED  feedback ┘
#               │
#               ▼
#         text response (result.output)
#     ├── strip_think_tags()
#     ├── source_registry.substitute()
#     └── update_chat(final)  → User Response



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
    "extra_body": {"options": {"num_ctx": OLLAMA_NUM_CTX, "temperature": 0.1}}
}

# Writer only — prose benefits from slightly more variability
OLLAMA_WRITE_SETTINGS = {
    "extra_body": {"options": {"num_ctx": OLLAMA_NUM_CTX, "temperature": 0.3}}
}

# Mistral special-token leak detection (e.g. <SPECIAL_27>)
_SPECIAL_TOKEN_RE = re.compile(r'<SPECIAL_\d+>')
MAX_SPECIAL_RETRIES = 2

def inject_date(agent: Agent) -> None:
    @agent.instructions
    def add_date() -> str:
        current_date = datetime.now(
            tz=ZoneInfo("UTC")
        ).strftime("%B %d, %Y")
        return f"Today's date is {current_date}"


MAX_RESEARCH_CALLS = 3   # initial + 2 follow-ups
MAX_WRITE_CALLS = 4      # initial + 3 revisions
MAX_REVIEW_CALLS = 3


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
    research_counter: CallCounter = field(
            default_factory=lambda: CallCounter(MAX_RESEARCH_CALLS)
        )
    write_counter: CallCounter = field(
            default_factory=lambda: CallCounter(MAX_WRITE_CALLS)
        )
    review_counter: CallCounter = field(
            default_factory=lambda: CallCounter(MAX_REVIEW_CALLS)
        )


class Quaesitor:
    """Research Agent
    """
    def __init__(self):
        # Import research tools — lazy to avoid circular dependency
        from .tools import TOOLS
        self.agent = Agent(
            model=research_coordinate_model,
            deps_type=AgentDeps,
            instructions=Prompts.RESEARCHER,
            tools=TOOLS,
            model_settings=OLLAMA_TOOL_SETTINGS,
            retries=2,
        )
        inject_date(agent=self.agent)

    async def run(self, directive: str, deps: AgentDeps, usage):
        """Run research, retrying if Mistral special tokens leak into output.

        Returns None if the agent fails with an unrecoverable tool error.
        """
        total_attempts = MAX_SPECIAL_RETRIES + 1
        for attempt in range(1, total_attempts + 1):
            try:
                result = await self.agent.run(
                    user_prompt=f"Research Directives:\n{directive}",
                    deps=deps,
                    usage=usage,
                    model_settings=OLLAMA_TOOL_SETTINGS,
                )
            except (UnexpectedModelBehavior, ModelHTTPError) as e:
                logger.warning(f"Quaesitor failure (attempt {attempt}/{total_attempts}): {e}")
                if attempt < total_attempts:
                    continue
                return None
            if not _SPECIAL_TOKEN_RE.search(result.output):
                return result
            logger.warning(
                f"Quaesitor special-token output (attempt {attempt}/{total_attempts}), retrying..."
            )
        return result


class Nuntius:
    """Report/Response generator
    """
    _META_RE = re.compile(
        r'\n+(?:[-–—]{2,}\s*\n+)?'          # optional horizontal rule
        r'(?:[-–—]\s*)?'                      # optional leading dash
        r'(?:Give_Final_Answer'
        r'|Call confirmed'
        r'|waiting on'
        r'|write_response'
        r'|run_research'
        r'|review_response'
        r')[^\n]*',
        re.IGNORECASE,
    )

    def __init__(self):
        self.agent = Agent(
            model=reflect_write_model,
            instructions=Prompts.WRITER
        )
        inject_date(agent=self.agent)

    async def write(self, user_input: str, research: str, usage) -> str:
        """Write a sourced response from research findings."""
        result = await self.agent.run(
            user_prompt=f"User Question: {user_input}\n\n{research}",
            usage=usage,
            model_settings=OLLAMA_WRITE_SETTINGS,
        )
        output = strip_think_tags(result.output)
        return self._META_RE.sub('', output).strip()


class Cogitator:
    """Reflection agent
    """
    def __init__(self):
        self.agent = Agent(
            model=reflect_write_model,
            instructions=Prompts.REFLECTION
        )
        inject_date(agent=self.agent)

    async def review(self, user_input: str, draft: str, usage) -> str:
        """Review a draft response; returns 'APPROVED' or improvement feedback."""
        result = await self.agent.run(
            user_prompt=f"Original Question: {user_input}\n\nDraft Response:\n{draft}",
            usage=usage,
            model_settings=OLLAMA_TOOL_SETTINGS,
        )
        return strip_think_tags(result.output)


class Praetor:
    """Coordination Agent
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.history = ChatHistoryManager()
        self._quaesitor: Quaesitor | None = None
        self._nuntius: Nuntius | None = None
        self._cogitator: Cogitator | None = None
        self.source_builder = SourceDataBuilder()
        self._registries: dict[int, SourceRegistry] = {}
        self.agent = Agent(
            model=research_coordinate_model,
            instructions=Prompts.COORDINATOR,
            deps_type=AgentDeps,
            output_type=str,
            tools=[
                self.run_research,
                self.write_response,
                self.review_response,
            ]
        )
        inject_date(self.agent)

    def _get_registry(self, chat_id: int) -> SourceRegistry:
        if chat_id not in self._registries:
            self._registries[chat_id] = SourceRegistry()
        return self._registries[chat_id]

    def clear(self, chat_id: int) -> None:
        """Clear conversation history and source registry for a chat."""
        self.history.clear(chat_id)
        self._registries.pop(chat_id, None)

    @property
    def quaesitor(self) -> Quaesitor:
        if not self._quaesitor:
            self._quaesitor = Quaesitor()
        return self._quaesitor

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
        if ctx.deps.research_counter.calls_exhausted():
            return "Research limit reached. Proceed with findings gathered so far."
        await ctx.deps.update_chat("_Researching..._")
        result = await self.quaesitor.run(directive, ctx.deps, ctx.usage)
        if result is None:
            return "Research failed due to a repeated tool call error. Proceed with any findings gathered so far."
        clean_output = _SPECIAL_TOKEN_RE.sub('', result.output).strip()
        tool_data, _ = self.source_builder.build(
            result.all_messages(), self._get_registry(ctx.deps.chat_id)
        )
        return f"{tool_data}\n\nResearch Findings:\n{clean_output}"

    async def write_response(self, ctx: RunContext[AgentDeps], research: str) -> str:
        """Writes a sourced intelligence report from research findings.

        Args:
            research (str): Combined source data and research findings from run_research.

        Returns:
            str: Draft response with [SOURCE_N] citation tags.
        """
        if ctx.deps.write_counter.calls_exhausted():
            return "Writing limit reached. Use the most recent draft as the final answer."
        await ctx.deps.update_chat("_Writing response..._")
        return await self.nuntius.write(ctx.deps.user_input, research, ctx.usage)

    async def review_response(self, ctx: RunContext[AgentDeps], draft: str) -> str:
        """Reviews a draft response for quality and source compliance.

        Args:
            draft (str): The draft response to review.

        Returns:
            str: "APPROVED" if the draft is acceptable, or specific improvement feedback.
        """
        if ctx.deps.review_counter.calls_exhausted():
            return "APPROVED"  # force acceptance once review limit is reached
        await ctx.deps.update_chat("_Reviewing response quality..._")
        return await self.cogitator.review(ctx.deps.user_input, draft, ctx.usage)

    async def handle_query(
                self,
                user_input: str,
                chat_id: int,
                update_chat: Callable[[str], Awaitable[None]]
            ):
        await update_chat("_Thinking..._")
        deps = AgentDeps(update_chat=update_chat, user_input=user_input, chat_id=chat_id)
        try:
            result = await self.agent.run(
                user_prompt=user_input,
                message_history=self.history.get(chat_id),
                deps=deps,
                model_settings=OLLAMA_TOOL_SETTINGS,
            )
            self.history.update(chat_id, result.all_messages())
            if result.output:
                final = self._get_registry(chat_id).substitute(strip_think_tags(result.output))
                await update_chat(final)
        except (UnexpectedModelBehavior, ModelHTTPError) as e:
            self.logger.warning(f"Praetor failure: {e}")
            await update_chat("_Something went wrong. Please try again._")
