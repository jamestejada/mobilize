import logging
import trafilatura
import aiohttp
from typing import Optional
from pydantic import BaseModel
from pydantic_ai import RunContext
from ..ai import AgentDeps
from ..source_registry import SourceRegistry

logger = logging.getLogger(__name__)
CDX_API = "https://web.archive.org/cdx/search/cdx"
MAX_BODY_CHARS = 3000


class ArchivedPage(BaseModel):
    original_url: str
    snapshot_url: str
    timestamp: str
    title: str = ""
    body: str = ""
    tag: str = ""

    @property
    def source_url(self) -> str:
        return self.snapshot_url


async def fetch_archived_page(
    ctx: RunContext[AgentDeps], url: str, date: str = ""
) -> Optional[ArchivedPage]:
    """Fetch a historical version of a webpage from the Wayback Machine.

    Use to retrieve content that has been deleted, edited, or scrubbed from the
    live web. If no date is given, returns the most recent available snapshot.

    Args:
        url (str): The original URL to look up. Example: "https://example.com/article"
        date (str): Optional target date in YYYYMMDD format. Example: "20230115"
                    Leave empty for most recent snapshot.

    Returns:
        ArchivedPage with title, body text, and snapshot URL. None if no snapshot found.

    Example:
        fetch_archived_page(url="https://somesite.com/deleted-page")
        fetch_archived_page(url="https://somesite.com/page", date="20220601")
    """
    await ctx.deps.update_chat(f"_Wayback Machine: looking up {url}_")
    try:
        params = {
            "url": url,
            "output": "json",
            "limit": 1,
            "fl": "timestamp,statuscode",
            "filter": "statuscode:200",
            "sort": "desc",
        }
        if date:
            params["from"] = date
            params["to"] = date + "235959"

        async with aiohttp.ClientSession() as session:
            async with session.get(CDX_API, params=params) as resp:
                data = await resp.json(content_type=None)

        if not data or len(data) < 2:
            return None

        timestamp = data[1][0]
        snapshot_url = f"https://web.archive.org/web/{timestamp}/{url}"

        html = trafilatura.fetch_url(snapshot_url)
        if not html:
            return None

        result = trafilatura.bare_extraction(
            html,
            url=snapshot_url,
            include_comments=False,
            include_tables=False,
            with_metadata=True,
        )
        if not result:
            return None

        body = (result.text or "")[:MAX_BODY_CHARS]
        if not body.strip():
            return None

        page = ArchivedPage(
            original_url=url,
            snapshot_url=snapshot_url,
            timestamp=timestamp,
            title=result.title or "",
            body=body,
        )
        SourceRegistry.register_one(ctx.deps.source_registry, page)
        return page

    except Exception as e:
        logger.warning(f"Wayback fetch failed for '{url}': {e}")
        return None
