from datetime import datetime, timedelta
from typing import Optional, List
import logging

from pydantic import BaseModel
from dateutil.parser import parse as parse_date

from src import settings

logger = logging.getLogger(__name__)


DATE_KEYS: List[str] = [
    "published",
    "updated",
    "created",
    "date",
    "pubDate",
    "datetimewritten"
]


class RSSFeedItem(BaseModel):
    """Represents a single RSS feed entry."""
    title: str
    link: str
    summary: str
    source_name: str
    published: Optional[datetime] = None
    id: Optional[str] = None
    author: Optional[str] = None
    thumbnail_url: Optional[str] = None
    tags: Optional[List[str]] = None
    relevance_score: Optional[float] = 0.0
    tag: str = ""

    @property
    def source_url(self) -> str:
        return self.link

    def __str__(self):
        return "\n".join([
                f"Title: {self.title}",
                f"Source: {self.source_name}",
                f"Published: {self.published}",
                f"Relevance Score: {self.relevance_score}",
                f"Link: {self.link}",
                f"Summary: {self.summary}",
                "-------------"
            ])

    def time_check(self, days: int) -> bool:
        if not self.published:
            return False
        try:
            return self.age < timedelta(days=days)
        except Exception as e:
            logger.error(
                f"Error in time_check for entry '{self.title}': {e}\n"
                f"Published date: {self.published}"
                )
            return False

    @property
    def age(self) -> Optional[timedelta]:
        if not self.published:
            return None
        try:
            return datetime.now(
                    tz=settings.TZ_INFOS["UTC"]
                ) - self.published
        except Exception as e:
            logger.error(
                f"Error calculating age for entry '{self.title}': {e}\n"
                f"Published date: {self.published}"
                )
            return None

    @property
    def freshness(self) -> Optional[float]:
        """Returns a freshness score between 0 and 1 based on age,
        where 1 is brand new and 0 is very old. We are using 48 hours
        as the threshold for freshness, but this can be adjusted as needed.
        """
        HOURS_THRESHOLD = 48
        if self.age:
            return max(0.0, 1.0 - (self.age.total_seconds()/(HOURS_THRESHOLD*3600)))
        return 0.0


    @property
    def current(self) -> bool:
        """Determines if the feed is current (not outdated)."""
        return self.time_check(days=2)

    @property
    def outdated(self) -> bool:
        """Determines if the feed is outdated"""
        return not self.time_check(days=300)

    @staticmethod
    def _resolve_published_date(entry: dict) -> Optional[datetime]:
        """Attempts to parse the published date from various possible keys.
        """
        for key in DATE_KEYS:
            if entry.get(key):
                try:
                    parsed_date = parse_date(
                        entry[key],
                        tzinfos=settings.TZ_INFOS
                        )
                    if not parsed_date.tzinfo:
                        parsed_date = parsed_date.replace(
                            tzinfo=settings.TZ_INFOS["UTC"]
                        )
                    return parsed_date
                except Exception:
                    continue
        logger.debug(
            f"Could not parse published date for entry: {entry.get('title')}"
            )
        return None

    @staticmethod
    def _resolve_thumbnail_url(entry: dict) -> Optional[str]:
        """Attempts to extract a thumbnail URL from media content."""
        if entry.get("media_thumbnail"):
            return entry["media_thumbnail"][0].get("url")
        elif entry.get("media_content"):
            for media in entry["media_content"]:
                if media.get("medium") == "image":
                    return media.get("url")
        logger.debug(
            f"Could not find thumbnail URL for entry: {entry.get('title')}"
            )
        return None

    @staticmethod
    def _resolve_tags(entry: dict) -> Optional[List[str]]:
        """Extracts tags from the entry if available."""
        if entry.get("tags"):
            return [t.get("term") for t in entry["tags"] if t.get("term")]
        logger.debug(
            f"Could not find tags for entry: {entry.get('title')}"
            )
        return None

    @classmethod
    def from_feedparser_entry(cls, entry: dict, source: str) -> "RSSFeedItem":
        """Create RSSFeedItem from a feedparser entry dict."""
        return cls(
            title=entry.get("title", ""),
            link=entry.get("link", ""),
            summary=entry.get("summary", ""),
            source_name=source,
            published=cls._resolve_published_date(entry),
            id=entry.get("id"),
            author=entry.get("author"),
            thumbnail_url=cls._resolve_thumbnail_url(entry),
            tags=cls._resolve_tags(entry),
        )
