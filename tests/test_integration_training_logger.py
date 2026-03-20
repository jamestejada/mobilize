"""Integration test: TrainingLogger full write/read cycle using tmp_path."""
import json
import pytest

pytestmark = pytest.mark.integration
from datetime import datetime
from pathlib import Path

from src.training_logger import TrainingLogger


@pytest.fixture
def logger(tmp_path):
    return TrainingLogger(log_dir=tmp_path)


# ---------------------------------------------------------------------------
# start → finalize → valid JSONL
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_start_returns_uuid(logger):
    iid = logger.start(chat_id=1, user_query="What is happening?")
    assert isinstance(iid, str)
    assert len(iid) == 36  # UUID format


@pytest.mark.unit
def test_two_starts_return_different_uuids(logger):
    iid1 = logger.start(chat_id=1, user_query="Query A")
    iid2 = logger.start(chat_id=1, user_query="Query B")
    assert iid1 != iid2


@pytest.mark.integration
def test_finalize_writes_jsonl_file(logger, tmp_path):
    iid = logger.start(chat_id=42, user_query="Test query")
    logger.finalize(iid, praetor_messages=[], final_response="Final answer here.")

    log_dir = tmp_path / "training"
    jsonl_files = list(log_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 1


@pytest.mark.integration
def test_finalize_file_named_by_date(logger, tmp_path):
    iid = logger.start(chat_id=1, user_query="Q")
    logger.finalize(iid, praetor_messages=[], final_response="R")

    expected_name = datetime.now().strftime("%Y%m%d") + ".jsonl"
    log_path = tmp_path / "training" / expected_name
    assert log_path.exists()


@pytest.mark.integration
def test_finalize_writes_valid_json_line(logger, tmp_path):
    iid = logger.start(chat_id=99, user_query="Integration test query")
    logger.finalize(iid, praetor_messages=[], final_response="The answer.")

    log_path = tmp_path / "training" / f"{datetime.now().strftime('%Y%m%d')}.jsonl"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["chat_id"] == 99
    assert entry["user_query"] == "Integration test query"
    assert entry["final_response"] == "The answer."
    assert "id" in entry
    assert "timestamp" in entry


@pytest.mark.unit
def test_finalize_clears_pending_entry(logger):
    iid = logger.start(chat_id=1, user_query="Q")
    assert iid in logger._pending
    logger.finalize(iid, praetor_messages=[], final_response="R")
    assert iid not in logger._pending


@pytest.mark.integration
def test_multiple_interactions_appended_to_same_file(logger, tmp_path):
    for i in range(3):
        iid = logger.start(chat_id=i, user_query=f"Query {i}")
        logger.finalize(iid, praetor_messages=[], final_response=f"Answer {i}")

    log_path = tmp_path / "training" / f"{datetime.now().strftime('%Y%m%d')}.jsonl"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        json.loads(line)  # each line must be valid JSON


# ---------------------------------------------------------------------------
# record_agent and record_nuntius appear in finalized entry
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_record_agent_appears_in_finalized_entry(logger, tmp_path):
    iid = logger.start(chat_id=1, user_query="Research query")
    logger.record_agent(
        iid,
        label="explorator",
        directive="Search for X",
        messages=[],
        findings="Found some things about X."
    )
    logger.finalize(iid, praetor_messages=[], final_response="Summary of X.")

    log_path = tmp_path / "training" / f"{datetime.now().strftime('%Y%m%d')}.jsonl"
    entry = json.loads(log_path.read_text().strip())
    assert len(entry["agents"]) == 1
    assert entry["agents"][0]["label"] == "explorator"
    assert entry["agents"][0]["findings"] == "Found some things about X."


@pytest.mark.integration
def test_record_nuntius_appears_in_finalized_entry(logger, tmp_path):
    iid = logger.start(chat_id=1, user_query="Write query")
    logger.record_nuntius(iid, draft="Draft response text.", feedback="APPROVED")
    logger.finalize(iid, praetor_messages=[], final_response="Final.")

    log_path = tmp_path / "training" / f"{datetime.now().strftime('%Y%m%d')}.jsonl"
    entry = json.loads(log_path.read_text().strip())
    assert len(entry["nuntius_iterations"]) == 1
    assert entry["nuntius_iterations"][0]["draft"] == "Draft response text."
    assert entry["nuntius_iterations"][0]["cogitator_feedback"] == "APPROVED"


# ---------------------------------------------------------------------------
# rate_by_message → ratings.jsonl
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_rate_by_message_writes_ratings_file(logger, tmp_path):
    iid = logger.start(chat_id=10, user_query="Q")
    logger.finalize(iid, praetor_messages=[], final_response="R")
    logger.associate_messages(iid, chat_id=10, message_ids=[101, 102], path="direct")

    logger.rate_by_message(chat_id=10, message_id=101, rating="good")

    ratings_path = tmp_path / "training" / "ratings.jsonl"
    assert ratings_path.exists()
    entry = json.loads(ratings_path.read_text().strip())
    assert entry["rating"] == "good"
    assert entry["interaction_id"] == iid


@pytest.mark.unit
def test_rate_by_message_unknown_message_id_is_noop(logger, tmp_path):
    logger.rate_by_message(chat_id=10, message_id=999, rating="bad")
    ratings_path = tmp_path / "training" / "ratings.jsonl"
    assert not ratings_path.exists()


@pytest.mark.integration
def test_rate_by_message_osint_path_sets_correct_field(logger, tmp_path):
    iid = logger.start(chat_id=5, user_query="OSINT Q")
    logger.finalize(iid, praetor_messages=[], final_response="R")
    logger.associate_messages(iid, chat_id=5, message_ids=[200], path="osint")

    logger.rate_by_message(chat_id=5, message_id=200, rating="mediocre")

    ratings_path = tmp_path / "training" / "ratings.jsonl"
    entry = json.loads(ratings_path.read_text().strip())
    assert entry["field"] == "nuntius_output"
