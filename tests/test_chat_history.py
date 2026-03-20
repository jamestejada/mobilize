"""Unit tests: ChatHistoryManager multi-turn round-trip."""
import pytest

pytestmark = pytest.mark.unit

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
    SystemPromptPart,
)

from src.chat_history import ChatHistoryManager


def user_msg(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def assistant_msg(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def tool_msg(text: str = "tool result") -> ModelResponse:
    """Simulates intermediate tool-call messages within a turn."""
    return ModelResponse(parts=[TextPart(content=text)])


def make_turn(user_text: str, answer_text: str, n_intermediate: int = 3):
    """Build a realistic pipeline turn: user msg + intermediate + final answer."""
    msgs = [user_msg(user_text)]
    for i in range(n_intermediate):
        msgs.append(tool_msg(f"[tool call {i}]"))
    msgs.append(assistant_msg(answer_text))
    return msgs


# ---------------------------------------------------------------------------
# Basic round-trip
# ---------------------------------------------------------------------------

def test_single_turn_stored_and_retrieved():
    mgr = ChatHistoryManager()
    turn = make_turn("Hello?", "Hi there!")
    mgr.update(chat_id=1, messages=turn)
    history = mgr.get(chat_id=1)
    assert len(history) > 0


def test_get_unknown_chat_returns_empty():
    mgr = ChatHistoryManager()
    assert mgr.get(chat_id=999) == []


def test_clear_removes_history():
    mgr = ChatHistoryManager()
    mgr.update(chat_id=1, messages=make_turn("Q", "A"))
    mgr.clear(chat_id=1)
    assert mgr.get(chat_id=1) == []


def test_different_chats_are_independent():
    mgr = ChatHistoryManager()
    mgr.update(chat_id=1, messages=make_turn("Chat 1 Q", "Chat 1 A"))
    mgr.update(chat_id=2, messages=make_turn("Chat 2 Q", "Chat 2 A"))
    mgr.clear(chat_id=1)
    assert mgr.get(chat_id=1) == []
    assert len(mgr.get(chat_id=2)) > 0


# ---------------------------------------------------------------------------
# Compression: length ≤ MAX_HISTORY
# ---------------------------------------------------------------------------

def test_compressed_length_within_max_history():
    mgr = ChatHistoryManager()
    MAX = mgr.MAX_HISTORY

    # Build enough turns to exceed MAX_HISTORY if uncompressed
    all_msgs = []
    for i in range(20):
        all_msgs.extend(make_turn(f"Question {i}", f"Answer {i}", n_intermediate=5))

    mgr.update(chat_id=1, messages=all_msgs)
    history = mgr.get(chat_id=1)

    assert len(history) <= MAX


def test_two_turns_compresses_first_keeps_second_full():
    mgr = ChatHistoryManager()

    turn1 = make_turn("First question", "First answer", n_intermediate=4)
    turn2 = make_turn("Second question", "Second answer", n_intermediate=4)
    all_msgs = turn1 + turn2

    mgr.update(chat_id=1, messages=all_msgs)
    history = mgr.get(chat_id=1)

    # turn1 compressed to 2 msgs; turn2 kept full (1 + 4 + 1 = 6)
    # total = 2 + 6 = 8
    assert len(history) == 8


def test_final_answer_preserved_after_compression():
    mgr = ChatHistoryManager()

    turn1 = make_turn("Old question", "Old final answer", n_intermediate=5)
    turn2 = make_turn("New question", "New final answer", n_intermediate=2)

    mgr.update(chat_id=1, messages=turn1 + turn2)
    history = mgr.get(chat_id=1)

    # The most recent answer should appear in history
    all_text = " ".join(
        part.content
        for msg in history
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, TextPart)
    )
    assert "New final answer" in all_text


def test_many_turns_answer_content_preserved():
    mgr = ChatHistoryManager()

    turns = []
    for i in range(10):
        turns.extend(make_turn(f"Q{i}", f"Answer{i}", n_intermediate=3))

    mgr.update(chat_id=1, messages=turns)
    history = mgr.get(chat_id=1)

    # Last answer must survive
    all_text = " ".join(
        part.content
        for msg in history
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, TextPart)
    )
    assert "Answer9" in all_text
    assert len(history) <= mgr.MAX_HISTORY


# ---------------------------------------------------------------------------
# Update is idempotent / accumulates correctly
# ---------------------------------------------------------------------------

def test_update_overwrites_previous():
    mgr = ChatHistoryManager()
    mgr.update(chat_id=1, messages=make_turn("First", "First answer"))
    mgr.update(chat_id=1, messages=make_turn("Second", "Second answer"))
    history = mgr.get(chat_id=1)
    all_text = " ".join(
        part.content
        for msg in history
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, TextPart)
    )
    assert "Second answer" in all_text
