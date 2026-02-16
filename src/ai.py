import re
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from typing import Callable, Dict, Awaitable, List, Tuple

from pydantic_ai import Agent, AgentRunResult, ModelResponse
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart

from .settings import (
        Models,
        Prompts,
        OllamaEndpoints,
        MAX_HISTORY,
        OLLAMA_NUM_CTX
    )

logger = logging.getLogger(__name__)



def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output.
    This is needed for qwen models.
    """
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)



#               ┌──────────────────────┐
#               │   System Orchestrator│
#               └─────────┬────────────┘
#                         │
#            ┌────────────┴──────────────┐
#            │                           │
#       ┌────────────────┐          ┌───────────────────┐
#       │ Coordination   │◄────────►│  Reflection       │
#       │ Agent (Praetor)│          │  Agent (Cogitator)│
#       └──────┬─────────┘          └───────────────────┘
#              │
#        ┌─────┴───────────────┐
#        │                     │
#       ┌────────────┐   ┌────────────┐
#       │ Researcher │   │ Report Gen │
#       │ (Quaesitor)│   │ (Nuntius)  │
#       └────────────┘   └────────────┘


research_coordinate_model = OpenAIChatModel(
    model_name=Models.Research_Coordinate,
    provider=OllamaProvider(
        base_url=str(OllamaEndpoints.API_ROOT)
    )
)
reflect_write_model = OpenAIChatModel(
    model_name=Models.Reflect_Write,
    provider=OllamaProvider(
        base_url=str(OllamaEndpoints.API_ROOT)
    )
)

OLLAMA_MODEL_SETTINGS = {"extra_body": {"options": {"num_ctx": OLLAMA_NUM_CTX}}}

def inject_date(agent: Agent) -> None:
    # Is this actually the best way to do it?
    # Maybe there should be a base class
    # that contains the agent instructions
    # for the current date.
    @agent.instructions
    def add_date() -> str:
        current_date = datetime.now(
            tz=ZoneInfo("UTC")
        ).strftime("%B %d, %Y")
        return f"Today's date is {current_date}"


@dataclass
class AgentDeps:
    update_chat: Callable[[str], Awaitable[None]]


class Quaesitor:
    """Research Agent
    """
    def __init__(self):
        # Import research tools
        # Lazy import to avoid circular dependency
        from .tools import TOOLS
        self.agent = Agent(
            model=research_coordinate_model,
            deps_type=AgentDeps,
            instructions=Prompts.RESEARCHER,
            tools=TOOLS
        )
        inject_date(agent=self.agent)

    async def research(
                self,
                directive: str,
                update_chat: Callable[[str], Awaitable[None]]
            ) -> AgentRunResult[str]:
        deps = AgentDeps(update_chat=update_chat)
        await update_chat("_Researching..._")
        result = await self.agent.run(
            user_prompt=f"Research Directives:\n{directive}",
            deps=deps,
            model_settings=OLLAMA_MODEL_SETTINGS
        )
        return result


class Nuntius:
    """Report/Response generator
    """
    def __init__(self):
        self.agent = Agent(
            model=reflect_write_model,
            instructions=Prompts.WRITER
        )
        inject_date(agent=self.agent)
    
    async def generate(
            self,
            user_input: str,
            research_findings: str,
            tool_data: str,
            update_chat: Callable[[str], Awaitable[None]]
        ) -> str:
        await update_chat("_Writing response..._")
        result = await self.agent.run(
            user_prompt=f"User Question: {user_input}\n\n"
                + f"{tool_data}\n\n"
                + f"Research Findings:\n{research_findings}",
            model_settings=OLLAMA_MODEL_SETTINGS
        )
        return strip_think_tags(result.output)


class Cogitator:
    """Reflection agent
    """
    def __init__(self):
        self.agent = Agent(
            model=reflect_write_model,
            instructions=Prompts.REFLECTION
        )
        inject_date(agent=self.agent)
    
    async def reflect(
            self,
            user_input: str,
            draft_response: str
        ) -> Tuple[bool, str]:

        result = await self.agent.run(
            user_prompt=f"Original Question: {user_input}\n\n"
                + f"Draft Response:\n{draft_response}",
            model_settings=OLLAMA_MODEL_SETTINGS
        )
        feedback = strip_think_tags(result.output)
        approved: bool = feedback.strip().upper().startswith("APPROVED")
        return approved, feedback


class Praetor:
    """Coordination Agent
    """
    MAX_HISTORY = MAX_HISTORY

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.chat_histories: Dict[int, List[ModelMessage]] = {}
        self._quaesitor: Quaesitor | None = None
        self._nuntius: Nuntius | None = None
        self._cogitator: Cogitator | None = None
        self.agent = Agent(
            model=research_coordinate_model,
            instructions=Prompts.COORDINATOR,
        )
        inject_date(self.agent)

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

    def get_history(self, chat_id: int) -> List[ModelMessage]:
        return self.chat_histories.get(chat_id, [])

    def trim_history(self, messages: List[ModelMessage]) -> List[ModelMessage]:
        if len(messages) <= self.MAX_HISTORY:
            return messages
        while messages and not isinstance(messages[0], ModelRequest):
            self.logger.info(f"Context Removed: {messages.pop(0)}")
        return messages[-self.MAX_HISTORY:]
    
    # Tools whose output is instructions to the researcher, not source data
    INTERMEDIATE_TOOLS = {
        "list_gov_rss_feeds",
        "list_us_news_rss_feeds",
        "list_world_news_rss_feeds"
    }

    def extract_tool_results(self, messages: List[ModelMessage]) -> str:
        tool_results = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if (isinstance(part, ToolReturnPart)
                            and part.content
                            and part.tool_name not in self.INTERMEDIATE_TOOLS):
                        tool_results.append(
                            f"[{part.tool_name}]:\n{part.content}"
                        )
        if not tool_results:
            return ""
        return "Source Data:\n" + "\n\n".join(tool_results)

    async def handle_query(
                self,
                user_input: str,
                chat_id: int,
                update_chat: Callable[[str], Awaitable[None]]
            ):
        # Step 1: Praetor generates research directive
        await update_chat(f"_Thinking..._")
        result = await self.agent.run(
            user_prompt=user_input,
            message_history=self.get_history(chat_id=chat_id),
            model_settings=OLLAMA_MODEL_SETTINGS
        )
        directive = strip_think_tags(result.output)
        print(f"Generated directive:\n{directive}")
        self.chat_histories[chat_id] = self.trim_history(
                result.all_messages()
            )

        # Step 2: Quaesitor researches
        research_result = await self.quaesitor.research(
            directive=directive,
            update_chat=update_chat
        )
        findings = research_result.output
        tool_data = self.extract_tool_results(
                research_result.all_messages()
            )

        # Step 3: Nuntius writes report
        draft = await self.nuntius.generate(
            user_input=user_input,
            research_findings=findings,
            tool_data=tool_data,
            update_chat=update_chat
        )
        print(f"Initial draft:\n{draft}")

        # Step 4: Cogitator reflects (up to 3 iterations)
        for i in range(3):
            await update_chat("_Reviewing response quality..._")
            approved, feedback = await self.cogitator.reflect(
                user_input=user_input,
                draft_response=draft
            )
            print(f"Reflection feedback: {feedback}")
            if approved:
                break
            await update_chat(f"_Revision {i + 1}..._")
            draft = await self.nuntius.generate(
                user_input=user_input,
                research_findings=f"{findings}\n\nReview Feedback:\n{feedback}",
                tool_data=tool_data,
                update_chat=update_chat
            )
            print(f"Revised draft:\n{draft}")

        # Step 5: Deliver
        final_response = draft.replace("**", "*").strip()
        await update_chat(final_response)
