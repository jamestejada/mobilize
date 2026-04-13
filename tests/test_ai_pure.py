"""Tests for pure logic in src/ai.py — no network, no agents."""
import pytest
from unittest.mock import AsyncMock

from src.ai import strip_think_tags, CallCounter, ResearchPlan, ResearchObjective

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# strip_think_tags
# ---------------------------------------------------------------------------

class TestStripThinkTags:
    def test_removes_think_block(self):
        result = strip_think_tags("<think>internal monologue</think>Answer here")
        assert "internal monologue" not in result
        assert "Answer here" in result

    def test_removes_multiline_think_block(self):
        text = "<think>\nline one\nline two\n</think>\nResponse"
        result = strip_think_tags(text)
        assert "line one" not in result
        assert "Response" in result

    def test_removes_thought_block(self):
        result = strip_think_tags("<thought>hidden</thought>visible")
        assert "hidden" not in result
        assert "visible" in result

    def test_unwraps_model_block(self):
        result = strip_think_tags("<model>APPROVED: looks good</model>")
        assert "<model>" not in result
        assert "APPROVED: looks good" in result

    def test_removes_multiple_blocks(self):
        text = "<think>A</think>middle<thought>B</thought>end"
        result = strip_think_tags(text)
        assert "A" not in result
        assert "B" not in result
        assert "middle" in result
        assert "end" in result

    def test_passthrough_plain_text(self):
        text = "Nothing to strip here."
        assert strip_think_tags(text) == text

    def test_orphaned_close_tag(self):
        # Text before closing tag should be dropped
        text = "leaked thought</think>clean output"
        result = strip_think_tags(text)
        assert "leaked thought" not in result
        assert "clean output" in result

    def test_strips_endoftext(self):
        result = strip_think_tags("Answer<|endoftext|>garbage")
        assert "garbage" not in result
        assert "Answer" in result

    def test_strips_im_start(self):
        result = strip_think_tags("Answer<|im_start|>garbage")
        assert "garbage" not in result
        assert "Answer" in result

    def test_removes_gemma_visible_thinking_block(self):
        text = (
            "Thinking...\n"
            "Here is a thinking process.\n"
            "1. Analyze the question.\n"
            "...done thinking.\n\n"
            "Final answer."
        )
        result = strip_think_tags(text)
        assert "thinking process" not in result
        assert "Analyze the question" not in result
        assert result == "Final answer."

    def test_preserves_plain_text_when_done_marker_missing(self):
        text = "Thinking... out loud about truth."
        assert strip_think_tags(text) == text


# ---------------------------------------------------------------------------
# CallCounter
# ---------------------------------------------------------------------------

class TestCallCounter:
    def test_not_exhausted_before_limit(self):
        c = CallCounter(max_calls=3)
        assert c.calls_exhausted() is False  # count=1
        assert c.calls_exhausted() is False  # count=2
        assert c.calls_exhausted() is False  # count=3

    def test_exhausted_at_limit_plus_one(self):
        c = CallCounter(max_calls=3)
        for _ in range(3):
            c.calls_exhausted()
        assert c.calls_exhausted() is True   # count=4 > 3

    def test_max_calls_one(self):
        # count=1, 1 > max_calls=1 is False — limit met but not exceeded yet
        c = CallCounter(max_calls=1)
        assert c.calls_exhausted() is False

    def test_max_calls_one_second_call_exhausted(self):
        c = CallCounter(max_calls=1)
        c.calls_exhausted()  # count=1, not yet exceeded
        assert c.calls_exhausted() is True   # count=2 > 1

    def test_max_calls_zero(self):
        # max_calls=0: first call count=1, 1>0=True
        c = CallCounter(max_calls=0)
        assert c.calls_exhausted() is True

    def test_stays_exhausted(self):
        c = CallCounter(max_calls=2)
        for _ in range(5):
            c.calls_exhausted()
        assert c.calls_exhausted() is True


# ---------------------------------------------------------------------------
# ResearchObjective
# ---------------------------------------------------------------------------

class TestResearchObjective:
    def test_defaults(self):
        obj = ResearchObjective(description="find news", tool_names=["search_news"])
        assert obj.completed is False
        assert obj.findings_summary == ""

    def test_can_be_marked_complete(self):
        obj = ResearchObjective(description="check", tool_names=[])
        obj.completed = True
        assert obj.completed is True


# ---------------------------------------------------------------------------
# ResearchPlan
# ---------------------------------------------------------------------------

class TestResearchPlan:
    def _make_plan(self):
        plan = ResearchPlan(query="test query")
        plan.objectives = [
            ResearchObjective("Find protests", ["search_web"], completed=False),
            ResearchObjective("Check legislation", ["search_legislation"], completed=True),
            ResearchObjective("Look up FEC data", ["search_candidate_finance"], completed=False),
        ]
        return plan

    def test_summary_contains_query(self):
        plan = self._make_plan()
        assert "test query" in plan.summary()

    def test_summary_shows_pending_status(self):
        plan = self._make_plan()
        summary = plan.summary()
        assert "PENDING" in summary

    def test_summary_shows_done_status(self):
        plan = self._make_plan()
        summary = plan.summary()
        assert "DONE" in summary

    def test_summary_all_complete(self):
        plan = ResearchPlan(query="q")
        plan.objectives = [
            ResearchObjective("A", [], completed=True),
            ResearchObjective("B", [], completed=True),
        ]
        summary = plan.summary()
        assert "PENDING" not in summary
        assert summary.count("DONE") == 2

    def test_summary_all_incomplete(self):
        plan = ResearchPlan(query="q")
        plan.objectives = [
            ResearchObjective("A", [], completed=False),
            ResearchObjective("B", [], completed=False),
        ]
        summary = plan.summary()
        assert "DONE" not in summary
        assert summary.count("PENDING") == 2

    def test_pending_objectives_returns_only_incomplete(self):
        plan = self._make_plan()
        pending = plan.pending_objectives()
        assert len(pending) == 2
        assert all(not o.completed for o in pending)

    def test_pending_objectives_empty_when_all_done(self):
        plan = ResearchPlan(query="q")
        plan.objectives = [
            ResearchObjective("A", [], completed=True),
        ]
        assert plan.pending_objectives() == []

    def test_pending_objectives_all_when_none_done(self):
        plan = ResearchPlan(query="q")
        plan.objectives = [
            ResearchObjective("A", [], completed=False),
            ResearchObjective("B", [], completed=False),
        ]
        assert len(plan.pending_objectives()) == 2

    def test_findings_summary_shown_in_summary(self):
        plan = ResearchPlan(query="q")
        obj = ResearchObjective("A", [], completed=True, findings_summary="found x")
        plan.objectives = [obj]
        assert "found x" in plan.summary()

    def test_empty_plan(self):
        plan = ResearchPlan(query="empty")
        assert plan.pending_objectives() == []
        assert "empty" in plan.summary()
