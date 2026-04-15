import asyncio
import logging
from typing import Optional
from urllib.parse import urlparse

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
    CHECK_INTERVAL = 60     # how often the watchman checks for idleness

    def __init__(self):
        self._playwright = None
        self._browser: Browser | None = None
        self._last_used: float = 0.0
        self._idle_task: asyncio.Task | None = None

    async def _get_browser(self) -> Browser:
        if self._browser is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
            self._idle_task = asyncio.create_task(self._idle_watchman())
            logger.info("Chromium launched")
        self._last_used = asyncio.get_event_loop().time()
        return self._browser

    async def _idle_watchman(self):
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
    tag: str = ""

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
    body = (getattr(result, "text", "") or "")[:MAX_BODY_CHARS]
    if not body.strip():
        return None
    return FetchedPage(
        url=url,
        title=getattr(result, "title", "") or "",
        body=body,
    )


def _is_supported_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


async def _fetch_page_from_url(url: str) -> Optional[FetchedPage]:
    """Fetch and extract a page from a direct URL."""
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


async def fetch_url(ctx: RunContext[AgentDeps], source_key: str) -> Optional[FetchedPage]:
    """Fetches full article text for a source already collected in this session.

    Use the [SOURCE_N] tag shown in search results, or just the number.
    Do NOT pass raw URLs — use the source tag instead.

    Use on the 1-3 most relevant results after search_news or search_web.

    Args:
        source_key (str): The source tag from a search result.
                          Accepts "SOURCE_3", "[SOURCE_3]", or just "3".

    Returns:
        FetchedPage with title and body text, or None if unreachable/unreadable.

    Example:
        fetch_url(source_key="SOURCE_3")
    """
    registry = ctx.deps.source_registry
    if registry is None:
        logger.warning("fetch_url: no source registry available")
        return None

    source_item = registry.lookup_by_key(source_key)
    if source_item is None:
        logger.warning(f"fetch_url: source key {source_key!r} not in registry")
        return None

    url = source_item.url
    await ctx.deps.update_chat(f"_Reading: {source_item.title or url}_")
    page = await _fetch_page_from_url(url)
    if page:
        page.tag = source_key.strip()
    return page


async def fetch_webpage(ctx: RunContext[AgentDeps], url: str) -> Optional[FetchedPage]:
    """Fetch a user-provided webpage URL directly and register it as a source.

    Use this when the user includes a full webpage link in their message and
    you need page text for context. Pass the raw URL exactly as given.

    Args:
        url (str): A full http(s) URL to fetch.

    Returns:
        FetchedPage with title/body text and a registered [SOURCE_N] tag, or
        None if the URL is invalid or the page is unreachable/unreadable.
    """
    if not _is_supported_url(url):
        logger.warning(f"fetch_webpage: invalid URL {url!r}")
        return None

    await ctx.deps.update_chat(f"_Reading: {url}_")
    page = await _fetch_page_from_url(url.strip())
    if page is None:
        return None

    registry = ctx.deps.source_registry
    if registry is not None:
        page.tag = registry.register(
            url=page.url,
            title=page.title or page.url,
            description=page.body[:200],
        )
    return page
