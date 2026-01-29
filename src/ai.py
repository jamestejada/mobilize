import re

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.chat_history import InMemoryChatMessageHistory

from typing import List, Dict

from .models import Event
from .settings import MODEL, Prompts


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)

class LLM:
    def __init__(self):
        self.model = self.initialize_model()
        self.memories: Dict[int, InMemoryChatMessageHistory] = {}
        self.events_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=Prompts.EVENT_ANALYSIS),
            MessagesPlaceholder("history", optional=True, n_messages=10),
            SystemMessagePromptTemplate.from_template("Events for analysis: {events}"),
            HumanMessagePromptTemplate.from_template("User Input: {user_input}"),
        ])
        self.regular_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=Prompts.GENERAL_OSINT),
            MessagesPlaceholder("history", optional=True, n_messages=10),
            HumanMessagePromptTemplate.from_template("User Input: {user_input}"),
        ])

    def initialize_model(self) -> ChatOllama:
        return ChatOllama(
                model=MODEL,
                base_url="http://192.168.4.100:11434"
            )
    
    def get_history(self, chat_id: int) -> InMemoryChatMessageHistory:
        if chat_id not in self.memories:
            self.memories[chat_id] = InMemoryChatMessageHistory()
        return self.memories[chat_id]
    
    def clear_history(self, chat_id: int):
        if chat_id in self.memories:
            self.memories[chat_id].clear()
    
    async def provide_analysis(
                self,
                user_input: str,
                events: List[Event],
                chat_id: int
            ) -> str:
        history = self.get_history(chat_id=chat_id)
        chat_prompt = self.events_prompt.format_prompt(
            user_input=user_input,
            events="\n\n".join([event.llm_context for event in events]),
            history=history.messages
        )
        response = await self.model.agenerate(
            [chat_prompt.to_messages()]
        )
        ai_text = strip_think_tags(response.generations[0][0].text)

        history.add_user_message(user_input)
        history.add_ai_message(ai_text)

        return ai_text

    async def simple_query(self, user_input: str, chat_id: int) -> str:
        history = self.get_history(chat_id=chat_id)
        chat_prompt = self.regular_prompt.format_prompt(
            user_input=user_input,
            history=history.messages
        )

        response = await self.model.agenerate(
            [chat_prompt.to_messages()]
        )
        ai_text = strip_think_tags(response.generations[0][0].text)

        history.add_user_message(user_input)
        history.add_ai_message(ai_text)

        return ai_text
