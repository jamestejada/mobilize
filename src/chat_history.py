import logging
from typing import Dict, List

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
)

from .settings import MAX_HISTORY


class ChatHistoryManager:
    """Manages per-chat conversation history with turn-based compression.

    Each pipeline run produces ~9 messages (tool calls + returns). Storing
    all of them verbatim means MAX_HISTORY=30 only covers ~3 real turns.

    Instead, completed turns are compressed to a [user_request, final_answer]
    pair (2 messages), while the most recent turn is kept intact for context.
    MAX_HISTORY=30 then covers ~13 real conversation turns.
    """

    MAX_HISTORY = MAX_HISTORY

    def __init__(self):
        self._histories: Dict[int, List[ModelMessage]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def get(self, chat_id: int) -> List[ModelMessage]:
        """Return stored message history for a chat."""
        return self._histories.get(chat_id, [])

    def update(self, chat_id: int, messages: List[ModelMessage]) -> None:
        """Compress and store history after a completed pipeline run."""
        self._histories[chat_id] = self._compress(messages)

    def clear(self, chat_id: int) -> None:
        """Discard all history for a chat."""
        self._histories.pop(chat_id, None)

    def _split_into_turns(self, messages: List[ModelMessage]) -> List[List[ModelMessage]]:
        """Group messages into pipeline runs, each starting with a UserPromptPart."""
        # XXX: There must be a way to refactor this
        # *Bangs on table* "There must be a better way"
        #               - Raymond Hettinger
        turns, current = [], []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                if any(
                    isinstance(
                        message_part, UserPromptPart
                        ) for message_part in msg.parts
                        ):
                    if current:
                        turns.append(current)
                    current = []
            current.append(msg)
        if current:
            turns.append(current)
        return turns

    def _extract_answer(self, turn: List[ModelMessage]) -> str | None:
        """Extract the final answer text from a completed turn.

        Praetor produces plain text output — find the last non-empty TextPart.
        """
        for msg in reversed(turn):
            if isinstance(msg, ModelResponse):
                for part in reversed(list(msg.parts)):
                    if isinstance(part, TextPart) and part.content.strip():
                        return part.content

        return None

    def _compress_turn(self, turn: List[ModelMessage]) -> List[ModelMessage]:
        """Reduce a completed turn to [user_request, final_answer_response].

        Falls back to keeping the full turn if no answer can be extracted.
        """
        answer = self._extract_answer(turn)
        if answer is None:
            self.logger.warning("Could not extract answer from turn — dropping turn")
            return []
        self.logger.info(f"Compressed turn: {len(turn)} msgs → 2")
        return [turn[0], ModelResponse(parts=[TextPart(content=answer)])]

    def _compress(self, messages: List[ModelMessage]) -> List[ModelMessage]:
        """Compress all but the most recent turn, then apply safety trim."""
        if not messages:
            return messages
        turns = self._split_into_turns(messages)
        if len(turns) <= 1:
            return messages[-self.MAX_HISTORY:]
        compressed = []
        for turn in turns[:-1]:
            compressed.extend(self._compress_turn(turn))
        compressed.extend(turns[-1])
        if len(compressed) > self.MAX_HISTORY:
            compressed = compressed[-self.MAX_HISTORY:]
        return compressed
