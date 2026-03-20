"""Tests for AsyncHTTPClient in src/tools/http_client.py."""
import pytest
from unittest.mock import AsyncMock, patch

import aiohttp

from src.tools.http_client import AsyncHTTPClient

pytestmark = pytest.mark.unit


class ConcreteClient(AsyncHTTPClient):
    BASE_URL = "https://api.example.com"


class TestBuildUrl:
    def test_basic_endpoint(self):
        client = ConcreteClient()
        assert client.build_url("search") == "https://api.example.com/search"

    def test_nested_endpoint(self):
        client = ConcreteClient()
        assert client.build_url("v1/candidates") == "https://api.example.com/v1/candidates"

    def test_empty_base_url(self):
        class NoBase(AsyncHTTPClient):
            BASE_URL = ""
        client = NoBase()
        assert client.build_url("ep") == "/ep"

    def test_empty_endpoint(self):
        client = ConcreteClient()
        assert client.build_url("") == "https://api.example.com/"


class TestRetry:
    @pytest.fixture
    def client(self):
        return ConcreteClient()

    async def test_success_on_first_try(self, client):
        expected = {"result": "ok"}
        client.http_request = AsyncMock(return_value=(200, expected))
        client.rate_limit_delay = AsyncMock()

        result = await client.retry(url="https://x", params={})

        assert result == expected
        client.http_request.assert_called_once()
        client.rate_limit_delay.assert_not_called()

    async def test_429_retries_and_returns_none_after_max(self, client):
        client.http_request = AsyncMock(return_value=(429, {}))
        client.rate_limit_delay = AsyncMock()

        result = await client.retry(url="https://x", params={})

        assert result is None
        assert client.http_request.call_count == AsyncHTTPClient.MAX_RETRIES
        assert client.rate_limit_delay.call_count == AsyncHTTPClient.MAX_RETRIES

    async def test_429_then_200_returns_data(self, client):
        data = {"key": "value"}
        client.http_request = AsyncMock(side_effect=[(429, {}), (200, data)])
        client.rate_limit_delay = AsyncMock()

        result = await client.retry(url="https://x", params={})

        assert result == data
        assert client.http_request.call_count == 2
        assert client.rate_limit_delay.call_count == 1

    async def test_client_error_returns_none(self, client):
        client.http_request = AsyncMock(side_effect=aiohttp.ClientError("connection failed"))
        client.rate_limit_delay = AsyncMock()

        result = await client.retry(url="https://x", params={})

        assert result is None

    async def test_client_error_calls_rate_limit_delay(self, client):
        client.http_request = AsyncMock(side_effect=aiohttp.ClientError("err"))
        client.rate_limit_delay = AsyncMock()

        await client.retry(url="https://x", params={})

        assert client.rate_limit_delay.call_count == AsyncHTTPClient.MAX_RETRIES

    async def test_200_with_params_passed_through(self, client):
        client.http_request = AsyncMock(return_value=(200, {}))
        client.rate_limit_delay = AsyncMock()

        params = {"api_key": "abc", "q": "test"}
        await client.retry(url="https://x", params=params)

        client.http_request.assert_called_once_with(url="https://x", params=params)

    async def test_non_429_non_200_status_returned(self, client):
        """Non-retry status codes (e.g. 404) are returned as data on first attempt."""
        client.http_request = AsyncMock(return_value=(404, {"error": "not found"}))
        client.rate_limit_delay = AsyncMock()

        result = await client.retry(url="https://x", params={})

        assert result == {"error": "not found"}
        client.http_request.assert_called_once()


class TestMaxRetries:
    def test_max_retries_constant(self):
        assert AsyncHTTPClient.MAX_RETRIES == 3

    def test_retry_delay_constant(self):
        assert AsyncHTTPClient.RETRY_DELAY == 7
