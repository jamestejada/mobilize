import pytest
"""Tests for CandidateFinance and CommitteeFinance in src/tools/fec.py."""
from src.tools.fec import CandidateFinance, CommitteeFinance

pytestmark = pytest.mark.unit


class TestCandidateFinanceSourceUrl:
    def test_source_url_format(self):
        cf = CandidateFinance(candidate_id="H0TX00213", name="Test Candidate")
        assert cf.source_url == "https://www.fec.gov/data/candidate/H0TX00213/"

    def test_source_url_ends_with_slash(self):
        cf = CandidateFinance(candidate_id="S6AZ00019", name="N")
        assert cf.source_url.endswith("/")

    def test_source_url_different_ids(self):
        cf = CandidateFinance(candidate_id="P80000722", name="N")
        assert "P80000722" in cf.source_url

    def test_defaults(self):
        cf = CandidateFinance(candidate_id="X", name="N")
        assert cf.party == ""
        assert cf.receipts == 0.0
        assert cf.tag == ""


class TestCommitteeFinanceSourceUrl:
    def test_source_url_format(self):
        cf = CommitteeFinance(committee_id="C00431445", name="ActBlue")
        assert cf.source_url == "https://www.fec.gov/data/committee/C00431445/"

    def test_source_url_ends_with_slash(self):
        cf = CommitteeFinance(committee_id="C00000935", name="N")
        assert cf.source_url.endswith("/")

    def test_source_url_different_ids(self):
        cf = CommitteeFinance(committee_id="C99999999", name="N")
        assert "C99999999" in cf.source_url

    def test_defaults(self):
        cf = CommitteeFinance(committee_id="X", name="N")
        assert cf.state == ""
        assert cf.receipts == 0.0
        assert cf.tag == ""
