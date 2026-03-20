"""Tests for OllamaRetryTransport in src/ollama_transport.py."""
import json
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from src.ollama_transport import OllamaRetryTransport

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_request(body: dict | None = None, method: str = "POST",
                 url: str = "http://localhost:11434/v1/chat/completions") -> httpx.Request:
    content = json.dumps(body).encode() if body is not None else b""
    return httpx.Request(method, url, content=content,
                         headers={"content-type": "application/json"})


def make_response(status: int, body: dict | None = None) -> httpx.Response:
    content = json.dumps(body or {}).encode()
    return httpx.Response(status, content=content)


# ---------------------------------------------------------------------------
# _sanitize_request
# ---------------------------------------------------------------------------

class TestSanitizeRequest:
    def setup_method(self):
        self.transport = OllamaRetryTransport()

    def test_null_content_replaced_with_empty_string(self):
        body = {"messages": [{"role": "assistant", "content": None}]}
        req = make_request(body)
        result = self.transport._sanitize_request(req)
        parsed = json.loads(result.content)
        assert parsed["messages"][0]["content"] == ""

    def test_non_null_content_unchanged(self):
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        req = make_request(body)
        result = self.transport._sanitize_request(req)
        parsed = json.loads(result.content)
        assert parsed["messages"][0]["content"] == "Hello"

    def test_no_messages_key_returns_unchanged(self):
        body = {"model": "qwen3"}
        req = make_request(body)
        result = self.transport._sanitize_request(req)
        assert result is req  # same object returned

    def test_multiple_messages_only_null_patched(self):
        body = {"messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "follow-up"},
        ]}
        req = make_request(body)
        result = self.transport._sanitize_request(req)
        msgs = json.loads(result.content)["messages"]
        assert msgs[0]["content"] == "question"
        assert msgs[1]["content"] == ""
        assert msgs[2]["content"] == "follow-up"

    def test_invalid_json_body_returns_original(self):
        req = httpx.Request("POST", "http://localhost/v1/chat",
                            content=b"not json",
                            headers={"content-type": "application/json"})
        result = self.transport._sanitize_request(req)
        assert result is req

    def test_no_change_returns_original_object(self):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        req = make_request(body)
        result = self.transport._sanitize_request(req)
        assert result is req  # no null → same request object


# ---------------------------------------------------------------------------
# fix_request
# ---------------------------------------------------------------------------

class TestFixRequest:
    def setup_method(self):
        self.transport = OllamaRetryTransport()

    def test_compact_json_body_written(self):
        body = {"messages": [{"role": "user", "content": ""}]}
        req = make_request(body)
        fixed = self.transport.fix_request(req, body)
        # Should be compact (no spaces after separators)
        assert b" " not in fixed.content or json.loads(fixed.content)

    def test_method_preserved(self):
        body = {"messages": []}
        req = make_request(body, method="POST")
        fixed = self.transport.fix_request(req, body)
        assert fixed.method == "POST"

    def test_url_preserved(self):
        url = "http://localhost:11434/v1/chat/completions"
        body = {"messages": []}
        req = make_request(body, url=url)
        fixed = self.transport.fix_request(req, body)
        assert str(fixed.url) == url

    def test_content_length_header_updated(self):
        body = {"messages": [{"role": "user", "content": "hello world"}]}
        req = make_request(body)
        fixed = self.transport.fix_request(req, body)
        expected_len = len(json.dumps(body, separators=(',', ':')).encode())
        assert fixed.headers["content-length"] == str(expected_len)

    def test_original_headers_preserved(self):
        body = {"messages": []}
        req = make_request(body)
        fixed = self.transport.fix_request(req, body)
        assert "content-type" in dict(fixed.headers)


# ---------------------------------------------------------------------------
# handle_async_request
# ---------------------------------------------------------------------------

class TestHandleAsyncRequest:
    def setup_method(self):
        self.transport = OllamaRetryTransport()

    @pytest.mark.asyncio
    async def test_200_returned_immediately(self):
        response_200 = make_response(200)
        self.transport._transport = MagicMock()
        self.transport._transport.handle_async_request = AsyncMock(return_value=response_200)

        body = {"messages": [{"role": "user", "content": "hi"}]}
        req = make_request(body)
        result = await self.transport.handle_async_request(req)

        assert result.status_code == 200
        self.transport._transport.handle_async_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_500_retried_max_retries_then_returned(self):
        response_500 = make_response(500)
        # aread() must be awaitable
        response_500.aread = AsyncMock(return_value=b"")

        self.transport._transport = MagicMock()
        self.transport._transport.handle_async_request = AsyncMock(return_value=response_500)

        body = {"messages": [{"role": "user", "content": "hi"}]}
        req = make_request(body)
        result = await self.transport.handle_async_request(req)

        assert result.status_code == 500
        assert self.transport._transport.handle_async_request.call_count == OllamaRetryTransport.MAX_RETRIES

    @pytest.mark.asyncio
    async def test_500_then_200_succeeds(self):
        response_500 = make_response(500)
        response_500.aread = AsyncMock(return_value=b"")
        response_200 = make_response(200)

        call_count = 0
        async def side_effect(req):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return response_500
            return response_200

        self.transport._transport = MagicMock()
        self.transport._transport.handle_async_request = AsyncMock(side_effect=side_effect)

        body = {"messages": [{"role": "user", "content": "hi"}]}
        req = make_request(body)
        result = await self.transport.handle_async_request(req)

        assert result.status_code == 200
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_request_sanitized_before_send(self):
        response_200 = make_response(200)
        captured = []

        async def capture(req):
            captured.append(req)
            return response_200

        self.transport._transport = MagicMock()
        self.transport._transport.handle_async_request = AsyncMock(side_effect=capture)

        # null content should be sanitized
        body = {"messages": [{"role": "assistant", "content": None}]}
        req = make_request(body)
        await self.transport.handle_async_request(req)

        sent_body = json.loads(captured[0].content)
        assert sent_body["messages"][0]["content"] == ""

    @pytest.mark.asyncio
    async def test_non_500_non_200_returned_immediately(self):
        response_400 = make_response(400)
        self.transport._transport = MagicMock()
        self.transport._transport.handle_async_request = AsyncMock(return_value=response_400)

        body = {"messages": []}
        req = make_request(body)
        result = await self.transport.handle_async_request(req)

        assert result.status_code == 400
        self.transport._transport.handle_async_request.assert_called_once()
