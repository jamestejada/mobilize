import asyncio
import logging
from typing import Optional

import trafilatura
from pydantic import BaseModel
from pydantic_ai import RunContext
from playwright.async_api import async_playwright, Browser

from ..ai import AgentDeps

logger = logging.getLogger(__name__)

MAX_BODY_CHARS = 3000  # ~750 tokens


class BrowserManager:
    """Manages a headless Chromium instance with idle-timeout auto-shutdown.

    The browser is launched lazily on first use and automatically closed after
    IDLE_TIMEOUT seconds of inactivity, preventing resource accumulation during
    long bot uptime. It relaunches transparently on the next fetch call.
    """

    IDLE_TIMEOUT = 10 * 60  # seconds before closing idle browser
    CHECK_INTERVAL = 60     # how often the watchdog checks for idleness

    def __init__(self):
        self._playwright = None
        self._browser: Browser | None = None
        self._last_used: float = 0.0
        self._idle_task: asyncio.Task | None = None

    async def _get_browser(self) -> Browser:
        if self._browser is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
            self._idle_task = asyncio.create_task(self._idle_watchdog())
            logger.info("Chromium launched")
        self._last_used = asyncio.get_event_loop().time()
        return self._browser

    async def _idle_watchdog(self):
        """Background task: close the browser after IDLE_TIMEOUT of inactivity."""
        while True:
            await asyncio.sleep(self.CHECK_INTERVAL)
            if self._browser is None:
                break
            idle = asyncio.get_event_loop().time() - self._last_used
            if idle >= self.IDLE_TIMEOUT:
                logger.info(f"Chromium idle for {idle:.0f}s — closing")
                await self.close()
                break

    async def fetch(self, url: str) -> str | None:
        """Render a URL with headless Chromium and return the full HTML."""
        try:
            browser = await self._get_browser()
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=15000)
            html = await page.content()
            await page.close()
            return html
        except Exception as e:
            logger.warning(f"Chromium fetch failed for {url}: {e}")
            await self.close()   # reset so next call gets a fresh browser
            return None

    async def close(self):
        """Explicitly close the browser and stop Playwright."""
        if self._idle_task:
            self._idle_task.cancel()
            self._idle_task = None
        if self._browser:
            await self._browser.close()
            self._browser = None
            logger.info("Chromium closed")
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None


_browser_manager = BrowserManager()


async def close_browser():
    """Call from main.py on shutdown to ensure clean Chromium teardown."""
    await _browser_manager.close()


class FetchedPage(BaseModel):
    url: str
    title: str
    body: str

    @property
    def source_url(self) -> str:
        return self.url


def _extract(html: str, url: str) -> Optional[FetchedPage]:
    """Run trafilatura extraction on an HTML string from any source."""
    result = trafilatura.bare_extraction(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        with_metadata=True,
    )
    if not result:
        return None
    body = (result.get("text", "") or "")[:MAX_BODY_CHARS]
    if not body.strip():
        return None
    return FetchedPage(
        url=url,
        title=result.get("title", "") or "",
        body=body,
    )


async def fetch_url(ctx: RunContext[AgentDeps], url: str) -> Optional[FetchedPage]:
    """Fetches a URL and extracts its main article text.

    Tries a fast HTTP fetch first; falls back to headless browser rendering
    for JavaScript-heavy pages. Returns None if the page cannot be read.

    Use on the 1-3 most relevant URLs from search results — not every result.

    Args:
        url (str): Full URL of the article or page to fetch.
                   Example: "https://apnews.com/article/some-article-slug"

    Returns:
        FetchedPage with title and body text, or None if unreachable/unreadable.

    Example:
        fetch_url(url="https://reuters.com/world/us/some-article")
    """
    await ctx.deps.update_chat(f"_Reading: {url}_")
    try:
        # Tier 1: trafilatura fetches and extracts directly (~0.5s)
        html = trafilatura.fetch_url(url)
        if html:
            page = _extract(html, url)
            if page:
                return page

        # Tier 2: JS fallback — Chromium renders, trafilatura extracts (~3-8s)
        logger.info(f"Trafilatura failed for {url}, trying Chromium")
        html = await _browser_manager.fetch(url)
        if html:
            return _extract(html, url)

    except Exception as e:
        logger.warning(f"fetch_url failed for {url}: {e}")

    return None
