import asyncio
import logging
from typing import Optional

import aiohttp


class AsyncHTTPClient:
    """Base async HTTP client with session management and retry logic."""

    BASE_URL: str = ""
    MAX_RETRIES = 3
    RETRY_DELAY = 7  # seconds

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(self.__class__.__name__)

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def build_url(self, endpoint: str) -> str:
        return f"{self.BASE_URL}/{endpoint}"

    async def request(self, endpoint: str, params: dict) -> Optional[dict]:
        """Makes a request with retry on rate limit."""
        url = self.build_url(endpoint)
        return await self.retry(url=url, params=params)

    async def retry(self, url: str, params: dict) -> Optional[dict]:
        """Retries requests with exponential backoff on rate limit or error."""
        for attempt in range(self.MAX_RETRIES):
            try:
                status, data = await self.http_request(url=url, params=params)
                if status == 429:
                    await self.rate_limit_delay(attempt=attempt)
                    continue
                return data
            except aiohttp.ClientError as e:
                await self.rate_limit_delay(attempt=attempt, e=e)
        self.logger.error("Failed after all retries")
        return None

    async def rate_limit_delay(self, attempt: int, e: Optional[Exception] = None):
        """Delays based on number of attempts for rate limiting."""
        delay = self.RETRY_DELAY * (2 ** attempt)
        if e:
            self.logger.error(f"Request error: {e}")
        else:
            self.logger.warning(
                f"Rate limited, retrying in {delay}s "
                f"(attempt {attempt + 1}/{self.MAX_RETRIES})"
            )
        await asyncio.sleep(delay)

    async def http_request(self, url: str, params: dict) -> tuple[int, dict]:
        """Makes a single HTTP GET request and returns (status, json_body)."""
        async with self.session.get(url, params=params) as response:
            data = await response.json()
            return response.status, data
