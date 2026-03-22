"""
Utilities for inspecting pydantic-ai tool call history in eval tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart


@dataclass
class ToolInvocation:
    tool_name: str
    args: dict[str, Any]  # always a dict, regardless of how model emitted it
    return_content: Any   # str, list, dict, or None depending on tool return type
    tool_call_id: str


def _parse_args(raw: str | dict | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def get_tool_invocations(result) -> list[ToolInvocation]:
    """
    Extract all tool calls and pair each with its return value.
    Returns a list of ToolInvocation in call order.
    """
    calls: list[tuple[str, ToolCallPart]] = []
    returns: dict[str, str] = {}  # tool_call_id → content

    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    calls.append((part.tool_call_id, part))
        elif isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    returns[part.tool_call_id] = part.content

    return [
        ToolInvocation(
            tool_name=part.tool_name,
            args=_parse_args(part.args),
            return_content=returns.get(call_id),
            tool_call_id=call_id,
        )
        for call_id, part in calls
    ]


def get_calls_for_tool(result, tool_name: str) -> list[ToolInvocation]:
    return [t for t in get_tool_invocations(result) if t.tool_name == tool_name]


def tool_return_looks_successful(invocation: ToolInvocation) -> bool:
    """
    Heuristic: a tool return is considered successful if:
    - content is not None
    - content is not an empty list/dict/string
    - content does not start with common error prefixes
    """
    c = invocation.return_content
    if c is None:
        return False
    if isinstance(c, list):
        return len(c) > 0
    if isinstance(c, dict):
        return bool(c)
    stripped = str(c).strip()
    if stripped in ("[]", "{}", "null", "None", ""):
        return False
    lower = stripped.lower()
    if lower.startswith("error") or lower.startswith("exception"):
        return False
    return True
