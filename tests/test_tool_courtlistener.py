import pytest
"""Tests for CourtCase model in src/tools/courtlistener.py."""
from src.tools.courtlistener import CourtCase

pytestmark = pytest.mark.unit


class TestCourtCase:
    def test_source_url_equals_url(self):
        case = CourtCase(
            case_name="Roe v. Wade",
            url="https://www.courtlistener.com/opinion/123/roe-v-wade/"
        )
        assert case.source_url == "https://www.courtlistener.com/opinion/123/roe-v-wade/"

    def test_source_url_format_with_absolute_url(self):
        """URL is pre-built by caller as https://www.courtlistener.com{absolute_url}"""
        case = CourtCase(case_name="T", url="https://www.courtlistener.com/opinion/456/case/")
        assert case.source_url.startswith("https://www.courtlistener.com")

    def test_summary_stored(self):
        case = CourtCase(case_name="T", url="https://www.courtlistener.com/x", summary="Case summary text")
        assert case.summary == "Case summary text"

    def test_summary_truncation_enforced_by_caller(self):
        # The model itself doesn't truncate — caller calls [:500] during _fetch_opinion_text
        long_summary = "s" * 600
        case = CourtCase(case_name="T", url="https://x", summary=long_summary)
        assert len(case.summary) == 600  # model stores whatever is given

    def test_defaults(self):
        case = CourtCase(case_name="Test v. Case")
        assert case.court == ""
        assert case.date_filed == ""
        assert case.summary == ""
        assert case.url == ""
        assert case.tag == ""

    def test_case_name_stored(self):
        case = CourtCase(case_name="Marbury v. Madison", url="https://www.courtlistener.com/x")
        assert case.case_name == "Marbury v. Madison"

    def test_url_construction_from_api(self):
        """Simulate how search_court_cases builds the URL from API response."""
        item = {"absolute_url": "/opinion/789/test-case/", "caseName": "Test v. Case"}
        case_url = f"https://www.courtlistener.com{item['absolute_url']}"
        case = CourtCase(case_name=item["caseName"], url=case_url)
        assert case.source_url == "https://www.courtlistener.com/opinion/789/test-case/"
