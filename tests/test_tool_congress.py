import pytest
"""Tests for Bill model and construction logic in src/tools/congress.py."""
from src.tools.congress import Bill

pytestmark = pytest.mark.unit


class TestBill:
    def test_source_url_equals_url(self):
        bill = Bill(bill_id="HR123", title="T", url="https://www.congress.gov/bill/119th-congress/house-bill/123")
        assert bill.source_url == "https://www.congress.gov/bill/119th-congress/house-bill/123"

    def test_source_url_has_no_format_json(self):
        # The caller strips ?format=json before constructing Bill
        bill = Bill(bill_id="HR1", title="T", url="https://www.congress.gov/bill/119/hr/1")
        assert "format=json" not in bill.source_url

    def test_bill_id_type_plus_number(self):
        # Bill ID is constructed by caller as f"{type}{number}"
        bill = Bill(bill_id="HR123", title="T")
        assert bill.bill_id == "HR123"

    def test_bill_id_senate(self):
        bill = Bill(bill_id="S45", title="T")
        assert bill.bill_id == "S45"

    def test_sponsor_stored(self):
        # Sponsor is extracted from first element of sponsors list by caller
        bill = Bill(bill_id="HR1", title="T", sponsor="Rep. Jane Doe [D-CA]")
        assert bill.sponsor == "Rep. Jane Doe [D-CA]"

    def test_latest_action_stored(self):
        bill = Bill(bill_id="HR1", title="T", latest_action="Passed Senate")
        assert bill.latest_action == "Passed Senate"

    def test_defaults(self):
        bill = Bill(bill_id="HR1", title="T")
        assert bill.congress == 0
        assert bill.sponsor == ""
        assert bill.latest_action == ""
        assert bill.tag == ""

    def test_bill_construction_from_api_dict(self):
        """Simulate how search_legislation constructs a Bill from API response."""
        item = {
            "type": "HR",
            "number": "456",
            "title": "A Bill to Do Things",
            "congress": 119,
            "introducedDate": "2025-01-15",
            "latestAction": {"text": "Referred to Committee"},
            "sponsors": [{"fullName": "Rep. Alice Smith [D-NY]"}],
            "url": "https://api.congress.gov/v3/bill/119/hr/456?format=json",
        }
        latest = (item.get("latestAction") or {})
        sponsor = (item.get("sponsors") or [{}])[0]
        bill_url = (item.get("url") or "").replace("?format=json", "")
        bill = Bill(
            bill_id=f"{item.get('type', '')}{item.get('number', '')}",
            title=item.get("title", ""),
            congress=item.get("congress", 0),
            bill_type=item.get("type", ""),
            introduced_date=item.get("introducedDate", ""),
            latest_action=latest.get("text", ""),
            sponsor=sponsor.get("fullName", ""),
            url=bill_url,
        )
        assert bill.bill_id == "HR456"
        assert bill.sponsor == "Rep. Alice Smith [D-NY]"
        assert bill.latest_action == "Referred to Committee"
        assert "format=json" not in bill.url
        assert bill.source_url == "https://api.congress.gov/v3/bill/119/hr/456"
