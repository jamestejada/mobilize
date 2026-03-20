"""Tests for TrainingLogger and emoji_to_rating in src/training_logger.py."""
import json
import pytest
from pathlib import Path
from datetime import datetime

from src.training_logger import TrainingLogger, emoji_to_rating

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# emoji_to_rating
# ---------------------------------------------------------------------------

class TestEmojiToRating:
    def test_fire_is_good(self):
        assert emoji_to_rating("🔥") == "good"

    def test_hundred_is_good(self):
        assert emoji_to_rating("💯") == "good"

    def test_smiling_hearts_is_good(self):
        assert emoji_to_rating("🥰") == "good"

    def test_star_struck_is_good(self):
        assert emoji_to_rating("🤩") == "good"

    def test_heart_on_fire_is_good(self):
        # ❤‍🔥 with ZWJ
        assert emoji_to_rating("❤\u200d🔥") == "good"

    def test_red_heart_is_good(self):
        assert emoji_to_rating("❤") == "good"

    def test_thumbs_up_is_mediocre(self):
        assert emoji_to_rating("👍") == "mediocre"

    def test_angry_is_bad(self):
        assert emoji_to_rating("🤬") == "bad"

    def test_thumbs_down_is_bad(self):
        assert emoji_to_rating("👎") == "bad"

    def test_unknown_defaults_to_mediocre(self):
        assert emoji_to_rating("🦄") == "mediocre"

    def test_empty_string_defaults_to_mediocre(self):
        assert emoji_to_rating("") == "mediocre"


# ---------------------------------------------------------------------------
# TrainingLogger.start()
# ---------------------------------------------------------------------------

class TestTrainingLoggerStart:
    def test_returns_uuid_string(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        iid = logger.start(chat_id=1, user_query="test")
        # UUID4 format: 8-4-4-4-12 hex chars
        assert len(iid) == 36
        assert iid.count("-") == 4

    def test_two_starts_return_different_uuids(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        iid1 = logger.start(1, "query one")
        iid2 = logger.start(1, "query two")
        assert iid1 != iid2

    def test_pending_has_correct_chat_id(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        iid = logger.start(chat_id=42, user_query="hello")
        assert logger._pending[iid]["chat_id"] == 42

    def test_pending_has_correct_query(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        iid = logger.start(chat_id=1, user_query="what is happening")
        assert logger._pending[iid]["user_query"] == "what is happening"

    def test_pending_initial_path_is_direct(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        iid = logger.start(1, "q")
        assert logger._pending[iid]["path"] == "direct"


# ---------------------------------------------------------------------------
# TrainingLogger.finalize()
# ---------------------------------------------------------------------------

class TestTrainingLoggerFinalize:
    def test_writes_jsonl_file(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        iid = logger.start(chat_id=1, user_query="test query")
        logger.finalize(iid, praetor_messages=[], final_response="The answer.")
        log_dir = tmp_path / "training"
        jsonl_files = list(log_dir.glob("*.jsonl"))
        assert len(jsonl_files) == 1

    def test_written_jsonl_is_valid_json(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        iid = logger.start(chat_id=1, user_query="test")
        logger.finalize(iid, praetor_messages=[], final_response="done")
        log_dir = tmp_path / "training"
        jsonl_file = next(log_dir.glob("*.jsonl"))
        with open(jsonl_file) as f:
            line = f.readline()
        entry = json.loads(line)
        assert entry["id"] == iid
        assert entry["final_response"] == "done"
        assert entry["chat_id"] == 1

    def test_finalize_clears_pending_entry(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        iid = logger.start(1, "q")
        logger.finalize(iid, [], "response")
        assert iid not in logger._pending

    def test_finalize_returns_path_string(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        iid = logger.start(1, "q")
        result = logger.finalize(iid, [], "response")
        assert isinstance(result, str)

    def test_finalize_unknown_iid_returns_direct(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        result = logger.finalize("nonexistent-id", [], "x")
        assert result == "direct"

    def test_filename_is_yyyymmdd(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        iid = logger.start(1, "q")
        logger.finalize(iid, [], "r")
        today = datetime.now().strftime("%Y%m%d")
        log_dir = tmp_path / "training"
        assert (log_dir / f"{today}.jsonl").exists()

    def test_multiple_entries_appended(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        iid1 = logger.start(1, "q1")
        logger.finalize(iid1, [], "r1")
        iid2 = logger.start(2, "q2")
        logger.finalize(iid2, [], "r2")
        log_dir = tmp_path / "training"
        jsonl_file = next(log_dir.glob("*.jsonl"))
        lines = jsonl_file.read_text().strip().split("\n")
        assert len(lines) == 2
        ids = {json.loads(l)["id"] for l in lines}
        assert ids == {iid1, iid2}


# ---------------------------------------------------------------------------
# TrainingLogger.rate_by_message()
# ---------------------------------------------------------------------------

class TestRateByMessage:
    def test_rate_writes_to_ratings_jsonl(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        iid = logger.start(1, "q")
        path = logger.finalize(iid, [], "response")
        logger.associate_messages(iid, chat_id=1, message_ids=[101], path=path)
        logger.rate_by_message(chat_id=1, message_id=101, rating="good")
        ratings_file = tmp_path / "training" / "ratings.jsonl"
        assert ratings_file.exists()
        entry = json.loads(ratings_file.read_text().strip())
        assert entry["interaction_id"] == iid
        assert entry["rating"] == "good"

    def test_rate_unknown_message_does_nothing(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        logger.rate_by_message(chat_id=99, message_id=999, rating="bad")
        ratings_file = tmp_path / "training" / "ratings.jsonl"
        assert not ratings_file.exists()

    def test_rate_osint_path_field_is_nuntius_output(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        iid = logger.start(1, "q")
        logger.set_path(iid, "osint")
        logger.finalize(iid, [], "response")
        logger.associate_messages(iid, chat_id=1, message_ids=[200], path="osint")
        logger.rate_by_message(chat_id=1, message_id=200, rating="good")
        ratings_file = tmp_path / "training" / "ratings.jsonl"
        entry = json.loads(ratings_file.read_text().strip())
        assert entry["field"] == "nuntius_output"

    def test_rate_direct_path_field_is_praetor_directive(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        iid = logger.start(1, "q")
        logger.finalize(iid, [], "response")
        logger.associate_messages(iid, chat_id=1, message_ids=[300], path="direct")
        logger.rate_by_message(chat_id=1, message_id=300, rating="bad")
        ratings_file = tmp_path / "training" / "ratings.jsonl"
        entry = json.loads(ratings_file.read_text().strip())
        assert entry["field"] == "praetor_directive"
