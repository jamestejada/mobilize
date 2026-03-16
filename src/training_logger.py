import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessagesTypeAdapter

logger = logging.getLogger(__name__)

GOOD_EMOJI = {"🔥", "💯", "🥰", "🤩", "❤\u200d🔥", "❤"}
MEDIOCRE_EMOJI = {"👍"}
BAD_EMOJI = {"🤬", "👎"}


def emoji_to_rating(emoji: str) -> str | None:
    if emoji in GOOD_EMOJI:
        return "good"
    if emoji in MEDIOCRE_EMOJI:
        return "mediocre"
    if emoji in BAD_EMOJI:
        return "bad"
    return "mediocre" # default


class TrainingLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir / "training"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._pending: dict[str, dict] = {}
        # Maps (chat_id, message_id) → (interaction_id, path)
        self._message_index: dict[tuple[int, int], tuple[str, str]] = {}

    def start(self, chat_id: int, user_query: str) -> str:
        iid = str(uuid.uuid4())
        self._pending[iid] = {
            "id": iid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chat_id": chat_id,
            "user_query": user_query,
            "path": "direct",
            "praetor_messages": [],
            "agents": [],
            "nuntius_iterations": [],
            "final_response": None,
            "message_ids": [],
            "ratings": {
                "overall": "mediocre",
                "praetor_directive": "mediocre",
                "explorator_tools": None,
                "tabularius_tools": None,
                "nuntius_output": None,
                "notes": None,
            },
        }
        return iid

    def set_path(self, iid: str, path: str) -> None:
        """Set to 'osint' when research findings exist."""
        if iid in self._pending:
            self._pending[iid]["path"] = path
            if path == "osint":
                self._pending[iid]["ratings"]["nuntius_output"] = "mediocre"

    def record_agent(self, iid: str, label: str, directive: str,
                     messages: list[Any], findings: str) -> None:
        if iid not in self._pending:
            return
        try:
            serialized = ModelMessagesTypeAdapter.dump_python(messages, mode="json")
        except Exception:
            serialized = []
        self._pending[iid]["agents"].append({
            "label": label,
            "directive": directive,
            "messages": serialized,
            "findings": findings,
        })

    def record_nuntius(self, iid: str, draft: str, feedback: str) -> None:
        if iid not in self._pending:
            return
        self._pending[iid]["nuntius_iterations"].append({
            "draft": draft,
            "cogitator_feedback": feedback,
        })

    def finalize(self, iid: str, praetor_messages: list[Any],
                 final_response: str) -> str:
        """Finalize and write to JSONL. Returns the path for message association."""
        if iid not in self._pending:
            return "direct"
        entry = self._pending.pop(iid)
        path = entry["path"]
        try:
            entry["praetor_messages"] = ModelMessagesTypeAdapter.dump_python(
                praetor_messages, mode="json"
            )
        except Exception:
            entry["praetor_messages"] = []
        entry["final_response"] = final_response

        log_path = self.log_dir / f"{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write training log: {e}")

        return path

    def associate_messages(self, iid: str, chat_id: int,
                           message_ids: list[int], path: str) -> None:
        """Map Telegram message IDs to this interaction for reaction lookup."""
        for mid in message_ids:
            self._message_index[(chat_id, mid)] = (iid, path)

    def rate_by_message(self, chat_id: int, message_id: int,
                        rating: str) -> None:
        """Called from reaction handler. Writes rating to ratings.jsonl."""
        key = (chat_id, message_id)
        if key not in self._message_index:
            return
        iid, path = self._message_index[key]
        field = "nuntius_output" if path == "osint" else "praetor_directive"
        entry = {
            "interaction_id": iid,
            "rating": rating,
            "field": field,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        ratings_path = self.log_dir / "ratings.jsonl"
        try:
            with open(ratings_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            logger.info(f"Rated interaction {iid[:8]}... as '{rating}' ({field})")
        except Exception as e:
            logger.warning(f"Failed to write rating: {e}")
