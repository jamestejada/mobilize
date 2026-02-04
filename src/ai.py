import re
import logging
from typing import Dict
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_ollama import ChatOllama
from langchain_core.messages import (
    SystemMessage,
    ToolMessage
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.chat_history import InMemoryChatMessageHistory

from .tools.mobilize import get_protests_for_llm, get_events
from .tools.web_search import search_web, search_news
from .tools.bsky import search_bluesky_posts
from .settings import MODEL, Prompts, BlueSkyCredentials
from .models import ToolCall


logger = logging.getLogger(__name__)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output.
    This is needed for qwen models.
    """
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)


class LLM:
    def __init__(self):
        self.tools: dict = {
            "get_protests_for_llm": get_protests_for_llm,
            "search_web": search_web,
            "search_news": search_news,
            "search_bluesky_posts": search_bluesky_posts,
        }
        self.model = self.initialize_model()
        self.memories: Dict[int, InMemoryChatMessageHistory] = {}
        current_date: str = datetime.now(
                tz=ZoneInfo("America/Chicago")
            ).strftime("%B %d, %Y")
        self.regular_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"Today's date is {current_date}."),
            SystemMessage(content=Prompts.GENERAL_OSINT),
            MessagesPlaceholder("history", optional=True, n_messages=10),
            HumanMessagePromptTemplate.from_template("User Input: {user_input}"),
        ])

    def initialize_model(self) -> ChatOllama:
        return ChatOllama(
                model=MODEL,
                base_url="http://192.168.4.100:11434",
                temperature=0.9
            ).bind_tools(list(self.tools.values()))
    
    def get_history(self, chat_id: int) -> InMemoryChatMessageHistory:
        if chat_id not in self.memories:
            self.memories[chat_id] = InMemoryChatMessageHistory()
        return self.memories[chat_id]
    
    def clear_history(self, chat_id: int):
        if chat_id in self.memories:
            self.memories[chat_id].clear()

    async def _execute_tool_call(self, tool_call: ToolCall) -> ToolMessage:
        # Invalid tool call
        if not tool_call.valid:
            return ToolMessage(
                content=self.tool_call_failed(tool_call.name),
                tool_call_id=tool_call.id,
            )
        
        # Valid tool call
        tool = self.tools.get(tool_call.name)
        if not tool:
            raise ValueError(f"Tool {tool_call.name} not found.")
        try:
            result = await tool.ainvoke(tool_call.args)
        except Exception as e:
            logger.error(f"Tool call {tool_call.name} failed: {e}")
            result = self.tool_call_failed(tool_call.name)
        return ToolMessage(
                content=result,
                tool_call_id=tool_call.id,
            )

    async def process_with_tools(
                self, 
                messages: list, 
                max_iterations: int = 10
            ) -> str:
        response = await self.model.ainvoke(messages)

        iterations = 0
        # LLM calls tools until done
        while response.tool_calls and iterations < max_iterations:
            messages.append(response)

            for tool_call_dict in response.tool_calls:
                tool_call = ToolCall(**tool_call_dict)
                tool_message = await self._execute_tool_call(tool_call)
                messages.append(tool_message)

            response = await self.model.ainvoke(messages)
            iterations += 1

        ai_text = strip_think_tags(response.content)
        return ai_text

    async def simple_query(self, user_input: str, chat_id: int) -> str:
        history = self.get_history(chat_id=chat_id)
        messages = self.regular_prompt.format_prompt(
            user_input=user_input,
            history=history.messages
        ).to_messages()

        ai_text = await self.process_with_tools(messages=messages)

        history.add_user_message(user_input)
        history.add_ai_message(ai_text)

        return ai_text
    
    def tool_call_failed(self, tool_name: str) -> str:
        return f"Sorry, I was unable to complete the tool call for {tool_name}."
