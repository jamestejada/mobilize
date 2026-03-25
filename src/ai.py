import re
import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
from typing import Callable, Awaitable, List

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
    """
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
        lines = [f"Research Plan for: {self.query}"]
        for i, obj in enumerate(self.objectives, 1):
            status = "DONE" if obj.completed else "PENDING"
            lines.append(f"{i}. [{status}] {obj.description}")
            if obj.findings_summary:
                lines.append(f"   Findings: {obj.findings_summary[:150]}")
        return "\n".join(lines)

    def pending_objectives(self) -> List[ResearchObjective]:
        return [o for o in self.objectives if not o.completed]


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
        self._probator: Probator | None = None
        self.source_builder = SourceDataBuilder()
        self._registries: dict[int, SourceRegistry] = {}
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
            ],
            retries=4,
        )
        inject_date(self.agent)
        inject_tool_list(self.agent, ALL_RESEARCH_TOOLSET)

    def _get_registry(self, chat_id: int) -> SourceRegistry:
        if chat_id not in self._registries:
            self._registries[chat_id] = SourceRegistry()
        return self._registries[chat_id]

    async def _build_writing_context(self, deps: AgentDeps, chat_id: int, query_embedding: list | None = None) -> str:
        """Build Nuntius input: semantically ranked sources + research summaries."""
        registry = self._get_registry(chat_id)
        await registry.embed_sources()
        source_data = registry.format_for_agent_semantic(query_embedding)
        return f"{source_data}\n\n{deps.research_findings}"

    def get_sources_by_tg_command(self, chat_id: int) -> str:
        """Return formatted source list for a chat."""
        return self._get_registry(chat_id).format_for_user()

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

    @property
    def probator(self) -> Probator:
        if not self._probator:
            self._probator = Probator()
        return self._probator

    async def _run_followup_research(self, gap_description: str, deps: AgentDeps, chat_id: int) -> None:
        """Run targeted follow-up research based on identified gaps."""
        directive = (
            f"OBJECTIVE: Fill identified gaps in prior research.\n"
            f"{gap_description}\n"
            f"CONTEXT: The user originally asked: {deps.user_input}\n"
            f"Search for NEW information — do NOT call get_registered_sources."
        )
        logger.debug(f"[Praetor→Followup] directive:\n{directive}")

        use_explorator = self.explorator.should_handle(directive)
        use_tabularius = self.tabularius.should_handle(directive)
        if not use_explorator and not use_tabularius:
            use_explorator = True

        tasks = []
        if use_explorator:
            tasks.append(self._run_single_agent(
                self.explorator, 
                "Web/Social-Followup",
                directive,
                deps,
                chat_id
            ))
        if use_tabularius:
            tasks.append(self._run_single_agent(
                self.tabularius, 
                "Data/Events-Followup",
                directive,
                deps,
                chat_id
            ))

        results = await asyncio.gather(*tasks)
        for r in results:
            if r is None:
                continue
            label, _, output = r
            deps.research_findings += f"\n\nFollow-up Findings ({label}):\n{output}"
            if len(deps.research_findings) > MAX_FINDINGS_CHARS:
                deps.research_findings = (
                    "...[earlier findings truncated]\n"
                    + deps.research_findings[-MAX_FINDINGS_CHARS:]
                )

    async def _run_single_agent(self, agent_runner, label, directive, deps, chat_id):
        """Run a single research agent and return its results.

        Returns (label, tool_data, output) or None if the agent failed.
        """
        result = await agent_runner.run(directive, deps)
        if result is None:
            logger.warning(f"{label} returned no result")
            return None
        logger.debug(f"[{label}] output:\n{result.output}")
        tool_data, _ = self.source_builder.build(
            result.all_messages(), self._get_registry(chat_id)
        )
        training_logger.record_agent(
            deps.interaction_id, label, directive,
            result.all_messages(), result.output or "",
        )
        return (label, tool_data, result.output)

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

        tasks = []
        if use_explorator:
            tasks.append(self._run_single_agent(
                self.explorator, "Web/Social", directive, ctx.deps, ctx.deps.chat_id
            ))
        if use_tabularius:
            tasks.append(self._run_single_agent(
                self.tabularius, "Data/Events", directive, ctx.deps, ctx.deps.chat_id
            ))

        summaries: list[str] = []
        results = await asyncio.gather(*tasks)
        for r in results:
            if r is None:
                continue
            label, _, output = r
            ctx.deps.research_findings += f"\n\nResearch Findings ({label}):\n{output}"
            if len(ctx.deps.research_findings) > MAX_FINDINGS_CHARS:
                ctx.deps.research_findings = (
                    "...[earlier findings truncated]\n"
                    + ctx.deps.research_findings[-MAX_FINDINGS_CHARS:]
                )
            summaries.append(f"[{label}]:\n{output}")

        # Update research plan progress if one exists
        plan_status = ""
        if ctx.deps.research_plan:
            for obj in ctx.deps.research_plan.objectives:
                if not obj.completed and (
                    not obj.tool_names
                    or any(tool in directive for tool in obj.tool_names)
                ):
                    obj.completed = True
                    obj.findings_summary = (
                        "\n\n".join(summaries)[:150] if summaries else ""
                    )
            pending = ctx.deps.research_plan.pending_objectives()
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

    async def _write_and_review(self, deps: AgentDeps) -> str:
        """Write, review, and return the final response text."""
        await deps.update_chat("_Writing response..._")
        try:
            query_embedding = await get_query_embedding(deps.user_input)
        except Exception as e:
            logger.warning(f"Query embedding failed, using recency order: {e}")
            query_embedding = None
        writing_context = await self._build_writing_context(deps, deps.chat_id, query_embedding)
        draft = await self.nuntius.write(deps.user_input, writing_context)
        logger.debug(f"[Nuntius] draft:\n{draft}")
        for _ in range(MAX_REVIEW_CALLS):
            await deps.update_chat("_Reviewing..._")
            feedback = await self.cogitator.review(deps.user_input, draft)
            logger.debug(f"[Cogitator] feedback:\n{feedback}")
            training_logger.record_nuntius(deps.interaction_id, draft, feedback)
            if "APPROVED" in feedback.upper():
                break
            if (
                "SEARCH:" in feedback
                and not deps.review_research_counter.calls_exhausted()
                    ):
                # IMPROVE feedback includes a SEARCH: section with tool calls —
                # pass the full feedback as the followup directive.
                logger.debug(f"[Cogitator] needs research:\n{feedback}")
                await deps.update_chat("_Gathering additional information..._")
                # Track whether followup found new sources; if not,
                # skip the rewrite — no point rewriting with same data.
                registry = self._get_registry(deps.chat_id)
                sources_before = len(registry._sources)
                await self._run_followup_research(
                    feedback, deps, deps.chat_id
                )
                found_new_sources = len(registry._sources) > sources_before
                if found_new_sources:
                    await deps.update_chat("_Revising with new information..._")
                else:
                    logger.warning("Review followup added no new sources — revising with existing sources")
                    await deps.update_chat("_Revising..._")
                writing_context = await self._build_writing_context(deps, deps.chat_id, query_embedding)
                nuntius_input = (
                    writing_context if found_new_sources
                    else f"{writing_context}\n\nPrevious Draft:\n{draft}\n\nReviewer Feedback:\n{feedback}"
                )
                draft = await self.nuntius.write(deps.user_input, nuntius_input)
            else:
                await deps.update_chat("_Revising..._")
                writing_context = await self._build_writing_context(deps, deps.chat_id, query_embedding)
                draft = await self.nuntius.write(
                    deps.user_input,
                    f"{writing_context}\n\nPrevious Draft:\n{draft}\n\nReviewer Feedback:\n{feedback}",
                )
            logger.debug(f"[Nuntius] revision:\n{draft}")
        return self._get_registry(deps.chat_id).substitute(strip_think_tags(draft))

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
                    # OSINT path — always goes through Nuntius+Cogitator
                    training_logger.set_path(iid, "osint")
                    # Gap analysis: check for weak claims before writing
                    registry = self._get_registry(chat_id)
                    while not deps.gap_research_counter.calls_exhausted():
                        await update_chat("_Analyzing research coverage..._")
                        gaps = await self.probator.analyze(
                            user_input, deps.research_findings
                        )
                        if gaps is None:
                            break
                        logger.debug(f"[Probator] gaps found:\n{gaps}")
                        await update_chat("_Filling research gaps..._")
                        # Track whether followup actually found new sources;
                        # if not, stop looping to avoid wasting cycles on
                        # agents that return nothing useful.
                        sources_before = len(registry._sources)
                        await self._run_followup_research(
                            gaps, deps, chat_id
                        )
                        if len(registry._sources) == sources_before:
                            logger.warning("Followup research added no new sources — stopping gap loop")
                            break
                    final = await self._write_and_review(deps)
                    await update_chat(final)
                elif result.output:
                    # Direct path — clarifications, out-of-scope, simple answers
                    final = self._get_registry(chat_id).substitute(strip_think_tags(result.output))
                    await update_chat(final)
                path = training_logger.finalize(iid, result.all_messages(), final)
                return iid, path
            except (UnexpectedModelBehavior, ModelHTTPError) as e:
                self.logger.warning(f"Praetor failure (attempt {attempt + 1}): {e}", exc_info=True)
        path = training_logger.finalize(iid, [], final)
        await update_chat("_Something went wrong. Please try again._")
        return iid, path
