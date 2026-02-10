import re
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from typing import Callable, Dict, Awaitable, List, Tuple

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.messages import ModelMessage

from .settings import MODEL, Prompts, OllamaEndpoints

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


ollama_model = OpenAIChatModel(
    model_name=MODEL,
    provider=OllamaProvider(
        base_url=str(OllamaEndpoints.API_ROOT)
    )
)


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
        from .tools import TOOLS  # Lazy import to avoid circular dependency
        self.agent = Agent(
            model=ollama_model,
            deps_type=AgentDeps,
            instructions=Prompts.RESEARCHER,
            tools=TOOLS
        )
        inject_date(agent=self.agent)

    async def research(
                self,
                directive: str,
                update_chat: Callable[[str], Awaitable[None]]
            ) -> str:
        deps = AgentDeps(update_chat=update_chat)
        await update_chat("_Researching..._")
        result = await self.agent.run(
            user_prompt=f"Research Directives:\n{directive}",
            deps=deps
        )
        return result.output


class Nuntius:
    """Report/Response generator
    """
    def __init__(self):
        self.agent = Agent(
            model=ollama_model,
            instructions=Prompts.WRITER
        )
        inject_date(agent=self.agent)
    
    async def generate(
            self,
            user_input: str,
            research_findings: str,
            update_chat: Callable[[str], Awaitable[None]]
        ) -> str:
        await update_chat("_Writing response..._")
        result = await self.agent.run(
            user_prompt=f"User Question: {user_input}\n\n"
                + f"Research Findings:\n{research_findings}"
        )
        return strip_think_tags(result.output)


class Cogitator:
    """Reflection agent
    """
    def __init__(self):
        self.agent = Agent(
            model=ollama_model,
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
                + f"Draft Response:\n{draft_response}"
        )
        feedback = strip_think_tags(result.output)
        approved: bool = feedback.strip().upper().startswith("APPROVED")
        return approved, feedback


class Praetor:
    """Coordination Agent
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.chat_histories: Dict[int, List[ModelMessage]] = {}
        self._quaesitor: Quaesitor | None = None
        self._nuntius: Nuntius | None = None
        self._cogitator: Cogitator | None = None
        self.agent = Agent(
            model=ollama_model,
            deps_type=AgentDeps,
            instructions=Prompts.COORDINATOR,
            tools=[
                Tool(self._research, takes_ctx=True),
                Tool(self._write_report, takes_ctx=True),
                Tool(self._reflect, takes_ctx=True)
            ]
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
    
    async def _research(
            self,
            ctx: RunContext[AgentDeps],
            directive: str
        ) -> str:
        """Delegate research directive to the Research Agent.
        Call this with a clear, specific research plan.

        Args: 
            directive: What to research - specific queries 
                data sources.
        """
        await ctx.deps.update_chat("_Starting Research Agent [Quaesitor]_")
        return await self.quaesitor.research(
            directive=directive,
            update_chat=ctx.deps.update_chat
        )
    
    async def _write_report(
            self,
            ctx: RunContext[AgentDeps],
            user_question: str,
            research_findings: str,
        ) -> str:
        """Delegates report writing to the Report Generator
        
        Args:
            user_question: The original question from the user
            research_findings: Raw findings from the Research Agent
        """
        await ctx.deps.update_chat("_Generating Report [Nuntius]..._")
        self.logger.info(f'Nuntius called.\nUser question: {user_question}')
        self.logger.info(f'Nuntius called.\nResearch Finding: {research_findings}')
        return await self.nuntius.generate(
                user_input=user_question,
                research_findings=research_findings,
                update_chat=ctx.deps.update_chat
            )
    
    async def _reflect(
                self,
                ctx: RunContext[AgentDeps],
                user_question: str,
                draft_response: str
            ) -> str:
        """Send a draft to the Reflection Agent for quality review.
        Returns APPROVED or IMPROVE with feedback.
        
        Args:
            user_question: The original question from the user.
            draft_response: The draft response to review
        """
        await ctx.deps.update_chat("_Reviewing response quality [Cogitator]..._")
        self.logger.info(f'Cogitator called.\nUser Question: {user_question}')
        self.logger.info(f'Cogitator called.\nDraft Response:\n{draft_response}')
        approved, feedback = await self.cogitator.reflect(
            user_input=user_question,
            draft_response=draft_response
        )
        return f"{'APPROVED' if approved else 'IMPROVE'}:\n{feedback}"

    def get_history(self, chat_id: int) -> List[ModelMessage]:
        return self.chat_histories.get(chat_id, [])

    async def handle_query(
                self,
                user_input: str,
                chat_id: int,
                update_chat: Callable[[str], Awaitable[None]]
            ):
        deps = AgentDeps(update_chat=update_chat)
        result = await self.agent.run(
            user_prompt=user_input,
            deps=deps,
            message_history=self.get_history(chat_id=chat_id)
        )
        self.chat_histories[chat_id] = result.all_messages()
        final_response = result.output.replace("**", "*").strip()
        await update_chat(final_response)


# async def get_embedding(text: str) -> List[float]:
#     async with httpx.AsyncClient() as client:
#         resp = await client.post(
#             str(OllamaEndpoints.EMBEDDINGS),
#             json={"model": "nomic-embed-text", "prompt": text}
#         )
#         return resp.json()["embedding"]


# def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
#     """Calculate cosine similarity between two vectors."""
#     if vec1 is None or vec2 is None:
#         return 0.0
#     vec1 = np.array(vec1)
#     vec2 = np.array(vec2)
#     if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
#         return 0.0
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# current_date: str = datetime.now(
#         tz=ZoneInfo("America/Chicago")
#     ).strftime("%B %d, %Y")

# REGULAR_PROMPT = ChatPromptTemplate.from_messages([
#     SystemMessage(content=f"Today's date is {current_date}."),
#     SystemMessage(content=Prompts.GENERAL_OSINT),
#     MessagesPlaceholder("history", optional=True, n_messages=10),
#     HumanMessagePromptTemplate.from_template("User Input: {user_input}"),
# ])
# REFLECTION_PROMPT = ChatPromptTemplate.from_messages([
#     SystemMessage(content=f"Today's date is {current_date}."),
#     SystemMessage(content=Prompts.REFLECTION),
#     HumanMessagePromptTemplate.from_template(
#         "Original question: {user_input}\n\nDraft response: {draft_response}"
#     ),
# ])


# class LLM:
#     def __init__(self):
#         from .tools import TOOLS  # Lazy import to avoid circular dependency
#         self.tools = TOOLS
#         self.model = self.initialize_model()
#         self.memories: Dict[int, InMemoryChatMessageHistory] = {}
#         self.regular_prompt = REGULAR_PROMPT
#         self.reflection_prompt = REFLECTION_PROMPT

#     def initialize_model(self) -> ChatOllama:
#         return ChatOllama(
#                 model=MODEL,
#                 base_url=str(OllamaEndpoints.API_ROOT),
#                 temperature=0.8
#             ).bind_tools(self.tool_functions)

#     @property
#     def reflection_model(self) -> ChatOllama:
#         return ChatOllama(
#                 model=MODEL,
#                 base_url=str(OllamaEndpoints.API_ROOT),
#                 temperature=0.5
#             )
    
#     async def reflect(
#                 self,
#                 user_input: str,
#                 draft_response: str,
#                 tool_information: str
#             ) -> Tuple[bool, str]:
#         prompt = self.reflection_prompt.format_prompt(
#             user_input=user_input,
#             draft_response=draft_response,
#             tool_information=tool_information
#         ).to_messages()
#         response = await self.reflection_model.ainvoke(prompt)
#         feedback = strip_think_tags(response.content)
#         return feedback.strip().upper().startswith("APPROVED"), feedback
    
#     @property
#     def tool_functions(self) -> List[Callable]:
#         return [tool.func for tool in self.tools.values()]

#     def get_history(self, chat_id: int) -> InMemoryChatMessageHistory:
#         if chat_id not in self.memories:
#             self.memories[chat_id] = InMemoryChatMessageHistory()
#         return self.memories[chat_id]
    
#     def clear_history(self, chat_id: int):
#         if chat_id in self.memories:
#             self.memories[chat_id].clear()

#     async def _try_valid_tool_call(self, tool_call: ToolCall) -> str:
#         tool = self.tools.get(tool_call.name)
#         if not tool:
#             raise ValueError(f"Tool {tool_call.name} not found.")
#         try:
#             return await tool.func.ainvoke(tool_call.args)
#         except Exception as e:
#             return self.tool_call_failed(tool_call.name, error=str(e))

#     async def _execute_tool_call(self, tool_call: ToolCall) -> ToolMessage:
#         # Invalid tool call
#         if not tool_call.valid:
#             return ToolMessage(
#                 content=self.tool_call_failed(tool_call.name),
#                 tool_call_id=tool_call.id,
#             )

#         # Valid tool call
#         result = await self._try_valid_tool_call(tool_call)
#         return ToolMessage(
#                 content=result,
#                 tool_call_id=tool_call.id,
#             )

#     async def process_with_tools(
#                 self, 
#                 messages: list,
#                 update_chat: Callable[[str], Awaitable[None]],
#                 max_iterations: int = 10
#             ) -> Dict[str, str]:
#         tool_messages = messages.copy()
#         response = await self.model.ainvoke(messages)

#         # LLM calls tools until done
#         for _ in range(max_iterations):
#             if not response.tool_calls:
#                 break
#             tool_messages.append(response)
#             await update_chat(
#                 response.content
#                 )
#             saturated_tool_calls = [
#                     ToolCall(**call_dict)
#                     for call_dict in response.tool_calls
#                 ]
#             for tool_call in saturated_tool_calls:
#                 logger.info(
#                     f"LLM requested tool call: {tool_call.name}"
#                     f" with args {tool_call.args}"
#                     )
#                 tool_config = self.tools.get(tool_call.name)
#                 if tool_config:
#                     await update_chat(f"Calling Tool: _{tool_config.status_text}_")
#                 tool_message = await self._execute_tool_call(tool_call)
#                 tool_messages.append(tool_message)

#             await update_chat("_Thinking about tool results..._")
#             response = await self.model.ainvoke(tool_messages)

#         tool_message_str = "\n\n".join(
#             f"---\nTool Call: {msg.tool_call_id}\nResult: {msg.content}\n---"
#             for msg in tool_messages
#             if isinstance(msg, ToolMessage)
#         )
#         return {
#             "draft": response.content,
#             "tool_results": tool_message_str
#         }

#     async def simple_query(
#                 self,
#                 user_input: str,
#                 chat_id: int,
#                 update_chat: Callable[[str], Awaitable[None]]
#             ) -> str:
#         history = self.get_history(chat_id=chat_id)
#         messages = self.regular_prompt.format_prompt(
#             user_input=user_input,
#             history=history.messages
#         ).to_messages()

#         draft = await self.process_with_tools(
#                 messages=messages,
#                 update_chat=update_chat
#             )
#         final_response = strip_think_tags(draft["draft"])
#         logger.debug(f"[DRAFT RESPONSE]\n{final_response}")
#         for i in range(5):  # Limit number of reflections to prevent infinite loops
#             # Reflection step for self-evaluation and improvement
#             approved, feedback = await self.reflect(
#                     user_input=user_input,
#                     draft_response=final_response,
#                     tool_information=strip_think_tags(draft["tool_results"])
#                 )
#             logger.debug(f"_{i + 1} [REFLECTION]\n{feedback}_")
#             if approved:
#                 await update_chat("_Response APPROVED by reflection model._")
#                 final_response = strip_think_tags(draft["draft"])
#                 break
#             messages.append(
#                 SystemMessage(content=f"[REVIEW FEEDBACK] {strip_think_tags(feedback)}")
#                 )
#             draft = await self.process_with_tools(
#                     messages=messages,
#                     update_chat=update_chat
#                 )
#             await update_chat(f"*{i + 2} [DRAFT RESPONSE]*\n{draft['draft']}")
#             final_response = strip_think_tags(draft["draft"])

#         await update_chat(f"Final Response:\n{strip_think_tags(final_response)}")
#         history.add_user_message(user_input)
#         history.add_ai_message(final_response)

#     def tool_call_failed(self, tool_name: str, error: Optional[str] = "") -> str:
#         return (
#             f"Sorry, I was unable to complete the tool call for {tool_name}." 
#             f"\n{error}"
#             )