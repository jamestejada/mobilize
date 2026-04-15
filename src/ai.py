import re
import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Dict, List

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior, ModelHTTPError
from .settings import (
        Prompts,
        LOG_DIR
    )
from .agent_settings import AgentsConfiguration
from .chat_history import ChatHistoryManager
from .source_registry import SourceRegistry, SourceDataBuilder, get_query_embedding
from .training_logger import TrainingLogger

logger = logging.getLogger(__name__)



def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> and other model artifact tags from output.
    Qwen models emit <think>, <thought>, <model>, and similar wrapper tags.
    Gemma-family models may emit a visible plain-text thinking preamble such as
    "Thinking..." followed by "...done thinking." before the real answer.
    """
    # Strip Gemma-style visible thinking preambles when they clearly bracket the
    # model's chain-of-thought before the answer begins.
    text = re.sub(
        r'^\s*Thinking(?:\.\.\.|…)\s*.*?(?:\.\.\.|…)?\s*done thinking\.\s*',
        '',
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    if '</think>' in text:
        text = text.split('</think>', 1)[-1]
    # Strip <thought>...</thought> wrapper tags
    text = re.sub(r'<thought>.*?</thought>\s*', '', text, flags=re.DOTALL)
    # Unwrap <model>...</model> tags, preserving inner content (e.g. "APPROVED:")
    text = re.sub(r'<model>(.*?)</model>', r'\1', text, flags=re.DOTALL)
    # Remove any trailing garbage after endoftext/im_start tokens
    text = re.sub(r'<\|endoftext\|>.*', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|im_start\|>.*', '', text, flags=re.DOTALL)
    # Strip \boxed{...} LaTeX artifacts (Qwen math training bleed)
    text = re.sub(r'\s*\\boxed\{[^}]*\}', '', text)
    return text.strip()


#                          User Query
#                               │
#                  ┌────────────┴────────────┐
#                  │        Praetor          │
#                  │  (coordinator agent)    │
#                  │                         │
#                  │  Tools: run_research,   │
#                  │  create_research_plan,  │
#                  │  get_sources            │
#                  └────────────┬────────────┘
#                               │
#              Praetor decides whether to call
#              run_research or answer directly
#                               │
#            ┌──────────────────┴───────────────────┐
#      called run_research?                   no research needed
#           YES                                    NO
#            │                                      │
#            ▼                                      ▼
#    run_research (parallel)              result.output delivered
#    ┌──────────┬──────────┐              (clarifications, out-of-scope)
#    │ asyncio.gather()    │
#    ├── Explorator        │
#    │   (web/social)      │
#    └── Tabularius        │
#        (data/events)     │
#            │
#            ▼
#    deps.research_findings populated
#            │
#            ▼
#    Probator gap loop (max 2 rounds)
#    ┌──────────────────────┐
#    │ probator.analyze()   │
#    │       │              │
#    │   None (adequate)  gaps found
#    │       │              │
#    │       │    _run_followup_research()
#    │       │         (parallel)
#    │       │              │
#    │       │    no new sources? → break
#    │       │              │
#    │    break        loop back ──┐
#    │       │              │      │
#    └───────┴──────────────┘      │
#            │          ▲──────────┘
#            ▼
#    _write_and_review()
#            │
#    ┌───────┴────────┐
#    │ nuntius.write() │
#    └───────┬────────┘
#            │
#    ┌───────┴─────────────┐
#    │ cogitator.review()  │←──┐
#    │       │             │   │ (up to 3 iterations)
#    │    APPROVED         │   │
#    │       │    IMPROVE  │   │
#    │       │       │     │   │
#    │       │    has SEARCH: section?
#    │       │    YES    NO│   │
#    │       │     │     └─┼───┘ (rewrite with feedback)
#    │       │     ▼       │
#    │       │  _run_followup_research()
#    │       │  no new sources? → revise with existing
#    │       │     │       │
#    │       │  nuntius.write() (with new data)
#    └───────┴─────┴───────┘
#            │
#    source_registry.substitute()
#            │
#    update_chat(final) → User Response



MAX_FINDINGS_CHARS = 12000  # ~3K tokens; cap narrative summary accumulation

MAX_SPECIAL_RETRIES = 2

def inject_date(agent: Agent) -> None:
    @agent.instructions
    def add_date() -> str:
        current_date = datetime.now(
            tz=ZoneInfo("UTC")
        ).strftime("%B %d, %Y")
        return f"Today's date is {current_date}"


def inject_tool_list(agent: Agent, toolset, extra_tools: list | None = None) -> None:
    """Append a dynamic tool list to agent instructions, generated from toolsets/functions."""
    lines = []

    def _iter_tools(ts):
        if hasattr(ts, "tools"):
            yield from ts.tools.values()
            return
        if hasattr(ts, "toolsets"):
            for child in ts.toolsets:
                yield from _iter_tools(child)

    for tool in _iter_tools(toolset):
        fn = tool.function
        first_line = (fn.__doc__ or "").strip().split("\n")[0].rstrip(".")
        lines.append(f"- `{fn.__name__}`: {first_line}")
    for fn in extra_tools or []:
        first_line = (fn.__doc__ or "").strip().split("\n")[0].rstrip(".")
        lines.append(f"- `{fn.__name__}`: {first_line}")
    tool_text = "\n".join(lines)

    @agent.instructions
    def _add_tool_list() -> str:
        return f"## Available Tools\n{tool_text}"


training_logger = TrainingLogger(LOG_DIR)

MAX_RESEARCH_CALLS = 5   # initial + 4 follow-ups
MAX_REVIEW_CALLS = 3     # write + up to 2 revisions
MAX_SPECIAL_RETRIES = 2  # retries for tool errors in Praetor before giving up
MAX_GAP_RESEARCH_CALLS = 2  # max follow-up research rounds for gap-filling


@dataclass
class CallCounter:
    max_calls: int
    count: int = 0

    def calls_exhausted(self) -> bool:
        """Increment the counter. Returns True if the limit has been exceeded."""
        self.count += 1
        return self.count > self.max_calls


@dataclass
class ResearchObjective:
    description: str
    tool_names: List[str]
    completed: bool = False
    findings_summary: str = ""


@dataclass
class ResearchPlan:
    query: str
    objectives: List[ResearchObjective] = field(default_factory=list)

    def summary(self) -> str:
        first_line = [f"Research Plan for user input: {self.query}"]
        lines: List[str] = first_line + [
            # status of objective
            f"{i}. [{'DONE' if objective.completed else 'PENDING'}] "
            # description
            + objective.description
            # findings summary if exists
            + (
                f"\n\tFindings: {objective.findings_summary[:150]}"
                if objective.findings_summary else ""
            )
            for i, objective in enumerate(self.objectives, 1)
        ]
        return "\n".join(lines)

    def pending_objectives(self) -> List[ResearchObjective]:
        return [
                objective for objective in self.objectives
                if not objective.completed
            ]


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
    gap_research_counter: CallCounter = field(
            default_factory=lambda: CallCounter(MAX_GAP_RESEARCH_CALLS)
        )
    review_research_counter: CallCounter = field(
            default_factory=lambda: CallCounter(MAX_GAP_RESEARCH_CALLS)
        )

    source_registry: SourceRegistry | None = None
    interaction_id: str = ""
    research_plan: ResearchPlan | None = None


class Explorator:
    """Web and Social Media Research Agent"""
    def __init__(self, model=None, instructions: str | None = None):
        from .tools import EXPLORATOR_AGENT_TOOLSET, EXPLORATOR_TOOLSET
        self.tool_names = EXPLORATOR_TOOLSET.tools
        self.agent = Agent(
            model=model or AgentsConfiguration.EXPLORATOR.make_model(),
            deps_type=AgentDeps,
            instructions=instructions if instructions is not None else Prompts.EXPLORATOR,
            toolsets=[EXPLORATOR_AGENT_TOOLSET],
            model_settings=AgentsConfiguration.EXPLORATOR.model_settings,
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
                    model_settings=AgentsConfiguration.EXPLORATOR.model_settings,
                )
            except (UnexpectedModelBehavior, ModelHTTPError) as e:
                logger.warning(f"Explorator failure: {e}")
        else:
            return None


class Tabularius:
    """Structured Data Research Agent"""
    def __init__(self, model=None, instructions: str | None = None):
        from .tools import TABULARIUS_AGENT_TOOLSET, TABULARIUS_TOOLSET
        self.tool_names = TABULARIUS_TOOLSET.tools
        self.agent = Agent(
            model=model or AgentsConfiguration.TABULARIUS.make_model(),
            deps_type=AgentDeps,
            instructions=instructions if instructions is not None else Prompts.TABULARIUS,
            toolsets=[TABULARIUS_AGENT_TOOLSET],
            model_settings=AgentsConfiguration.TABULARIUS.model_settings,
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
                    model_settings=AgentsConfiguration.TABULARIUS.model_settings,
                )
            except (UnexpectedModelBehavior, ModelHTTPError) as e:
                logger.warning(f"Tabularius failure: {e}")
        else:
            return None


class Nuntius:
    """Report/Response generator
    """
    def __init__(self, model=None, instructions: str | None = None):
        self.agent = Agent(
            model=model or AgentsConfiguration.NUNTIUS.make_model(),
            instructions=instructions if instructions is not None else Prompts.WRITER
        )
        inject_date(agent=self.agent)

    async def write(
        self,
        user_input: str,
        research: str,
        model_settings: dict | None = None,
    ) -> str:
        """Write a sourced response from research findings."""
        result = await self.agent.run(
            user_prompt=f"User Question: {user_input}\n\n{research}",
            model_settings=model_settings or AgentsConfiguration.NUNTIUS.model_settings,
        )
        output = strip_think_tags(result.output)
        return output


class Cogitator:
    """Reflection agent
    """
    def __init__(self, model=None, instructions: str | None = None):
        self.agent = Agent(
            model=model or AgentsConfiguration.COGITATOR.make_model(),
            instructions=instructions if instructions is not None else Prompts.REFLECTION
        )
        inject_date(agent=self.agent)

    async def review(
        self,
        user_input: str,
        draft: str,
        model_settings: dict | None = None,
    ) -> str:
        """Review a draft response; returns 'APPROVED' or improvement feedback."""
        result = await self.agent.run(
            user_prompt=f"Original Question: {user_input}\n\nDraft Response:\n{draft}",
            model_settings=model_settings or AgentsConfiguration.COGITATOR.model_settings,
        )
        return strip_think_tags(result.output)


class Probator:
    """Research gap analysis agent — evaluates research completeness."""
    def __init__(self, model=None, instructions: str | None = None):
        self.agent = Agent(
            model=model or AgentsConfiguration.PROBATOR.make_model(),
            instructions=instructions if instructions is not None else Prompts.GAP_ANALYSIS
        )
        inject_date(agent=self.agent)

    async def analyze(
        self,
        user_input: str,
        research: str,
        model_settings: dict | None = None,
    ) -> str | None:
        """Analyze research for gaps. Returns gap description or None if adequate."""
        result = await self.agent.run(
            user_prompt=f"Original Question: {user_input}\n\nResearch Findings:\n{research}",
            model_settings=model_settings or AgentsConfiguration.PROBATOR.model_settings,
        )
        output = strip_think_tags(result.output)
        if "ADEQUATE" in output.upper():
            return None
        return output


class ResearchOrchestrator:
    """Dispatches research agents, builds sources, and fills gaps."""

    def __init__(
        self,
        source_builder: SourceDataBuilder,
        get_registry: Callable[[int], SourceRegistry],
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._source_builder = source_builder
        self._get_registry = get_registry
        self._explorator: Explorator | None = None
        self._tabularius: Tabularius | None = None
        self._probator: Probator | None = None

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
    def probator(self) -> Probator:
        if not self._probator:
            self._probator = Probator()
        return self._probator

    def _select_agents(self, directive: str) -> tuple[bool, bool]:
        """Decide which research agents should handle a directive."""
        use_explorator = self.explorator.should_handle(directive)
        use_tabularius = self.tabularius.should_handle(directive)
        if not use_explorator and not use_tabularius:
            use_explorator = True
        return use_explorator, use_tabularius

    async def _run_single_agent(self, agent_runner, label, directive, deps, chat_id) -> Dict[str, str] | None:
        """Run a single research agent and return its results.

        Returns dict with label, tool_data, output keys — or None on failure.
        """
        result = await agent_runner.run(directive, deps)
        if result is None:
            self.logger.warning(f"{label} returned no result")
            return None
        self.logger.debug(f"[{label}] output:\n{result.output}")
        tool_data = self._source_builder.build(
            result.all_messages(), self._get_registry(chat_id)
        )
        training_logger.record_agent(
            deps.interaction_id, label, directive,
            result.all_messages(), result.output or "",
        )
        return {
            "label": label,
            "tool_data": tool_data,
            "output": result.output or "",
        }

    def _accumulate_findings(self, deps: AgentDeps, label: str, output: str) -> None:
        """Append findings to deps and truncate if over limit."""
        deps.research_findings += f"\n\n{label}:\n{output}"
        if len(deps.research_findings) > MAX_FINDINGS_CHARS:
            deps.research_findings = (
                "...[earlier findings truncated]\n"
                + deps.research_findings[-MAX_FINDINGS_CHARS:]
            )

    async def _dispatch_agents(
        self, directive: str, deps: AgentDeps, label_suffix: str = "",
    ) -> List[Dict[str, str]]:
        """Select and run research agents in parallel, returning their results."""
        use_explorator, use_tabularius = self._select_agents(directive)
        tasks = []
        if use_explorator:
            tasks.append(self._run_single_agent(
                self.explorator, f"Web/Social{label_suffix}", directive, deps, deps.chat_id
            ))
        if use_tabularius:
            tasks.append(self._run_single_agent(
                self.tabularius, f"Data/Events{label_suffix}", directive, deps, deps.chat_id
            ))
        return [r for r in await asyncio.gather(*tasks) if r is not None]

    def update_plan():
        """Update research plan progress based on completed objectives."""
        

    async def run_research(self, directive: str, deps: AgentDeps) -> str:
        """Dispatch Explorator/Tabularius in parallel and return a summary."""
        self.logger.debug(f"[Research] directive:\n{directive}")
        results = await self._dispatch_agents(directive, deps)

        summaries: list[str] = []
        for r in results:
            self._accumulate_findings(deps, f"Research Findings ({r['label']})", r["output"])
            summaries.append(f"[{r['label']}]:\n{r['output']}")

        # Update research plan progress if one exists
        plan_status = ""
        if deps.research_plan:
            for obj in deps.research_plan.objectives:
                if not obj.completed and (
                    not obj.tool_names
                    or any(tool in directive for tool in obj.tool_names)
                ):
                    obj.completed = True
                    obj.findings_summary = (
                        "\n\n".join(summaries)[:150] if summaries else ""
                    )
            pending = deps.research_plan.pending_objectives()
            if pending:
                pending_list = "\n".join(
                    f"- {o.description}" for o in pending
                )
                plan_status = (
                    f"\n\nPlan progress: {len(pending)} objectives remaining:\n"
                    f"{pending_list}"
                )

        summary = "\n\n".join(summaries) if summaries else "No findings returned."
        return (
            f"Research complete.\n\n{summary}{plan_status}\n\n"
            "If the above reveals important leads (key people, orgs, breaking events) "
            "not yet fully investigated, call run_research again with a targeted follow-up "
            "directive. Otherwise stop."
        )

    async def run_followup(self, gap_description: str, deps: AgentDeps) -> None:
        """Run targeted follow-up research based on identified gaps."""
        directive = (
            f"OBJECTIVE: Fill identified gaps in prior research.\n"
            f"{gap_description}\n"
            f"CONTEXT: The user originally asked: {deps.user_input}\n"
            f"Search for NEW information — do NOT call get_registered_sources."
        )
        self.logger.debug(f"[Followup] directive:\n{directive}")
        for r in await self._dispatch_agents(directive, deps, label_suffix="-Followup"):
            self._accumulate_findings(deps, f"Follow-up Findings ({r['label']})", r["output"])

    async def run_gap_analysis(self, deps: AgentDeps) -> None:
        """Probator loop: analyze research for gaps and fill them."""
        registry = self._get_registry(deps.chat_id)
        while not deps.gap_research_counter.calls_exhausted():
            await deps.update_chat("_Analyzing research coverage..._")
            gaps = await self.probator.analyze(
                deps.user_input, deps.research_findings
            )
            if gaps is None:
                break
            self.logger.debug(f"[Probator] gaps found:\n{gaps}")
            await deps.update_chat("_Filling research gaps..._")
            sources_before = len(registry._sources)
            await self.run_followup(gaps, deps)
            if len(registry._sources) == sources_before:
                self.logger.warning("Followup research added no new sources — stopping gap loop")
                break


class WritingPipeline:
    """Iterative write/review cycle with source substitution."""

    def __init__(
        self,
        get_registry: Callable[[int], SourceRegistry],
        research: ResearchOrchestrator,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._get_registry = get_registry
        self._research = research
        self._nuntius: Nuntius | None = None
        self._cogitator: Cogitator | None = None

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

    async def _build_writing_context(self, deps: AgentDeps, query_embedding: list | None = None) -> str:
        """Build Nuntius input: semantically ranked sources + research summaries."""
        registry = self._get_registry(deps.chat_id)
        await registry.embed_sources()
        source_data = registry.format_for_agent_semantic(query_embedding)
        return f"{source_data}\n\n{deps.research_findings}"

    async def write_and_review(self, deps: AgentDeps) -> str:
        """Write, review, and return the final response text."""
        await deps.update_chat("_Writing response..._")
        try:
            query_embedding = await get_query_embedding(deps.user_input)
        except Exception as e:
            self.logger.warning(f"Query embedding failed, using recency order: {e}")
            query_embedding = None
        writing_context = await self._build_writing_context(deps, query_embedding)
        draft = await self.nuntius.write(deps.user_input, writing_context)
        self.logger.debug(f"[Nuntius] draft:\n{draft}")
        for _ in range(MAX_REVIEW_CALLS):
            await deps.update_chat("_Reviewing..._")
            feedback = await self.cogitator.review(deps.user_input, draft)
            self.logger.debug(f"[Cogitator] feedback:\n{feedback}")
            training_logger.record_nuntius(deps.interaction_id, draft, feedback)
            if "APPROVED" in feedback.upper():
                break
            if (
                "SEARCH:" in feedback
                and not deps.review_research_counter.calls_exhausted()
                    ):
                self.logger.debug(f"[Cogitator] needs research:\n{feedback}")
                await deps.update_chat("_Gathering additional information..._")
                registry = self._get_registry(deps.chat_id)
                sources_before = len(registry._sources)
                await self._research.run_followup(feedback, deps)
                found_new_sources = len(registry._sources) > sources_before
                if found_new_sources:
                    await deps.update_chat("_Revising with new information..._")
                else:
                    self.logger.warning("Review followup added no new sources — revising with existing sources")
                    await deps.update_chat("_Revising..._")
                writing_context = await self._build_writing_context(deps, query_embedding)
                nuntius_input = (
                    writing_context if found_new_sources
                    else f"{writing_context}\n\nPrevious Draft:\n{draft}\n\nReviewer Feedback:\n{feedback}"
                )
                draft = await self.nuntius.write(deps.user_input, nuntius_input)
            else:
                await deps.update_chat("_Revising..._")
                writing_context = await self._build_writing_context(deps, query_embedding)
                draft = await self.nuntius.write(
                    deps.user_input,
                    f"{writing_context}\n\nPrevious Draft:\n{draft}\n\nReviewer Feedback:\n{feedback}",
                )
            self.logger.debug(f"[Nuntius] revision:\n{draft}")
        return self._get_registry(deps.chat_id).substitute(strip_think_tags(draft))


class Praetor:
    """Coordination Agent — owns the pydantic-ai Agent and delegates to
    ResearchOrchestrator and WritingPipeline for the heavy lifting.
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.history = ChatHistoryManager()
        self._registries: dict[int, SourceRegistry] = {}
        self._research = ResearchOrchestrator(SourceDataBuilder(), self._get_registry)
        self._writing = WritingPipeline(self._get_registry, self._research)
        from .tools import ALL_RESEARCH_TOOLSET
        self.agent = Agent(
            model=AgentsConfiguration.PRAETOR.make_model(),
            instructions=Prompts.COORDINATOR,
            deps_type=AgentDeps,
            output_type=str,
            tools=[
                self.run_research,
                self.get_sources,
                self.create_research_plan,
                self.fetch_webpage,
            ],
            retries=4,
        )
        inject_date(self.agent)
        inject_tool_list(self.agent, ALL_RESEARCH_TOOLSET, extra_tools=[self.fetch_webpage])

    def _get_registry(self, chat_id: int) -> SourceRegistry:
        if chat_id not in self._registries:
            self._registries[chat_id] = SourceRegistry()
        return self._registries[chat_id]

    def get_sources_by_tg_command(self, chat_id: int) -> str:
        """Return formatted source list for a chat."""
        return self._get_registry(chat_id).format_for_user()

    def clear(self, chat_id: int) -> None:
        """Clear conversation history and source registry for a chat."""
        self.history.clear(chat_id)
        self._registries.pop(chat_id, None)

    async def run_research(self, ctx: RunContext[AgentDeps], directive: str) -> str:
        """Researches the directive using OSINT tools.

        Args:
            directive (str): What to investigate and which tools to use.

        Returns:
            str: Research findings and source data for the writer.
        """
        await ctx.deps.update_chat("_Researching..._")
        return await self._research.run_research(directive, ctx.deps)

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
        return self._get_registry(ctx.deps.chat_id).format_for_agent_semantic()

    async def create_research_plan(self, ctx: RunContext[AgentDeps], objectives: str) -> str:
        """Creates a structured research plan before executing research.

        Use this for complex queries with multiple angles to investigate.
        For simple queries, skip this and call run_research directly.

        Args:
            objectives (str): Newline-separated list of research objectives.
                Each line can optionally include tool names in parentheses.
                Example: "1. (search_news, search_web) Investigate recent protests\\n2. (search_polymarket) Check prediction markets"

        Returns:
            str: The research plan summary showing all objectives.
        """
        plan = ResearchPlan(query=ctx.deps.user_input)
        for line in objectives.strip().split("\n"):
            line = line.strip().lstrip("0123456789.-) ")
            if not line:
                continue
            tools: List[str] = []
            if "(" in line and ")" in line:
                tools_str = line[line.index("(") + 1:line.index(")")]
                tools = [t.strip() for t in tools_str.split(",")]
                line = line[line.index(")") + 1:].strip()
            plan.objectives.append(ResearchObjective(description=line, tool_names=tools))
        ctx.deps.research_plan = plan
        logger.debug(f"[Praetor] Research plan created:\n{plan.summary()}")
        return plan.summary()

    async def fetch_webpage(self, ctx: RunContext[AgentDeps], url: str) -> str:
        """Fetch a user-provided webpage directly for context and citation."""
        from .tools.fetch_url import fetch_webpage

        page = await fetch_webpage(ctx, url)
        if page is None:
            return "Unable to fetch the webpage."

        title = page.title or page.url
        tag = f" {page.tag}" if page.tag else ""
        return (
            f"Fetched webpage:{tag}\n"
            f"Title: {title}\n"
            f"URL: {page.url}\n"
            f"Body:\n{page.body}"
        )

    async def handle_query(
                self,
                user_input: str,
                chat_id: int,
                update_chat: Callable[[str], Awaitable[None]]
            ) -> tuple[str, str]:
        await update_chat("_Thinking..._")
        iid = training_logger.start(chat_id, user_input)
        deps = AgentDeps(
            update_chat=update_chat,
            user_input=user_input,
            chat_id=chat_id,
            source_registry=self._get_registry(chat_id),
            interaction_id=iid,
        )
        final = ""
        for attempt in range(MAX_SPECIAL_RETRIES + 1):
            try:
                result = await self.agent.run(
                    user_prompt=user_input,
                    message_history=self.history.get(chat_id),
                    deps=deps,
                    model_settings=AgentsConfiguration.PRAETOR.model_settings,
                )
                logger.debug(f"[Praetor] output:\n{result.output}")
                self.history.update(chat_id, result.all_messages())
                if deps.research_findings:
                    training_logger.set_path(iid, "osint")
                    await self._research.run_gap_analysis(deps)
                    final = await self._writing.write_and_review(deps)
                    await update_chat(final)
                elif result.output:
                    final = self._get_registry(chat_id).substitute(strip_think_tags(result.output))
                    await update_chat(final)
                path = training_logger.finalize(iid, result.all_messages(), final)
                return iid, path
            except (UnexpectedModelBehavior, ModelHTTPError) as e:
                self.logger.warning(f"Praetor failure (attempt {attempt + 1}): {e}", exc_info=True)
        path = training_logger.finalize(iid, [], final)
        await update_chat("_Something went wrong. Please try again._")
        return iid, path
