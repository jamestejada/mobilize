from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List
import logging

from dateutil.parser import parse as parse_date

from src import settings

logger = logging.getLogger(__name__)


class EventType(Enum):
    CANVASS = "CANVASS"
    PHONE_BANK = "PHONE_BANK"
    TEXT_BANK = "TEXT_BANK"
    MEETING = "MEETING"
    COMMUNITY = "COMMUNITY"
    FUNDRAISER = "FUNDRAISER"
    MEET_GREET = "MEET_GREET"
    HOUSE_PARTY = "HOUSE_PARTY"
    VOTER_REG = "VOTER_REG"
    TRAINING = "TRAINING"
    FRIEND_TO_FRIEND_OUTREACH = "FRIEND_TO_FRIEND_OUTREACH"
    DEBATE_WATCH_PARTY = "DEBATE_WATCH_PARTY"
    ADVOCACY_CALL = "ADVOCACY_CALL"
    RALLY = "RALLY"
    TOWN_HALL = "TOWN_HALL"
    OFFICE_OPENING = "OFFICE_OPENING"
    BARNSTORM = "BARNSTORM"
    SOLIDARITY_EVENT = "SOLIDARITY_EVENT"
    COMMUNITY_CANVASS = "COMMUNITY_CANVASS"
    SIGNATURE_GATHERING = "SIGNATURE_GATHERING"
    CARPOOL = "CARPOOL"
    WORKSHOP = "WORKSHOP"
    PETITION = "PETITION"
    AUTOMATED_PHONE_BANK = "AUTOMATED_PHONE_BANK"
    LETTER_WRITING = "LETTER_WRITING"
    LITERATURE_DROP_OFF = "LITERATURE_DROP_OFF"
    VISIBILITY_EVENT = "VISIBILITY_EVENT"
    PLEDGE = "PLEDGE"
    INTEREST_FORM = "INTEREST_FORM"
    DONATION_CAMPAIGN = "DONATION_CAMPAIGN"
    SOCIAL_MEDIA_CAMPAIGN = "SOCIAL_MEDIA_CAMPAIGN"
    POSTCARD_WRITING = "POSTCARD_WRITING"
    GROUP = "GROUP"
    VOLUNTEER_SHIFT = "VOLUNTEER_SHIFT"
    OTHER = "OTHER"


@dataclass
class Sponsor:
    id: int
    name: str
    slug: str
    org_type: str
    is_coordinated: bool
    is_independent: bool
    is_nonelectoral: bool
    is_primary_campaign: bool
    state: str
    district: str
    candidate_name: str
    event_feed_url: str
    created_date: int
    modified_date: int
    logo_url: str
    race_type: Optional[str] = None


@dataclass
class Coordinates:
    latitude: float
    longitude: float


@dataclass
class Location:
    venue: str
    address_lines: List[str]
    locality: str
    region: str
    country: str
    postal_code: str
    location: Optional[Coordinates] = None
    congressional_district: Optional[str] = None
    state_leg_district: Optional[str] = None
    state_senate_district: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.location, dict):
            self.location = Coordinates(**self.location)


@dataclass
class Timeslot:
    id: int
    start_date: datetime
    end_date: datetime
    is_full: bool
    instructions: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.start_date, int):
            self.start_date = datetime.fromtimestamp(self.start_date)
        if isinstance(self.end_date, int):
            self.end_date = datetime.fromtimestamp(self.end_date)

@dataclass
class Event:
    id: int
    title: str
    summary: str
    description: str
    event_type: EventType
    timezone: str
    browser_url: str
    created_date: int
    modified_date: int
    visibility: str
    address_visibility: str
    accessibility_status: str
    approval_status: str
    is_virtual: bool
    created_by_volunteer_host: bool
    sponsor: Sponsor
    location: Location
    timeslots: List[Timeslot]
    tags: List[str]
    featured_image_url: Optional[str] = None
    contact: Optional[str] = None
    event_campaign: Optional[str] = None
    instructions: Optional[str] = None
    high_priority: Optional[bool] = None
    virtual_action_url: Optional[str] = None
    accessibility_notes: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.event_type, str):
            self.event_type = EventType(self.event_type)
        if isinstance(self.sponsor, dict):
            self.sponsor = Sponsor(**self.sponsor)
        if isinstance(self.location, dict):
            self.location = Location(**self.location)
        if self.timeslots and isinstance(self.timeslots[0], dict):
            self.timeslots = [Timeslot(**t) for t in self.timeslots]
    
    @property
    def location_str(self) -> str:
        return ", ".join([
                self.location.venue,
                self.location.locality,
                self.location.region
            ])

    @property
    def telegram_message(self) -> str:
        return '\n'.join([
            f"_*{self.title}*_",
            f"Date: {self.timeslots[0].start_date.strftime('%Y-%m-%d %H:%M')}",
            f"Location: {self.location_str}",
            f"Coordinates: {self.coordinates}",
            f"Organizer: {self.sponsor.name}",
            f"[More Info]({self.browser_url})"
        ])

    @property
    def coordinates(self) -> Optional[str]:
        if not self.location.location:
            return "N/A"
        return f"{self.location.location.latitude}, {self.location.location.longitude}"

    @property
    def llm_context(self) -> str:
        """Returns a formatted string with of a single event for LLM context.
        FORMAT:
            Title: {title}
            Type: {event_type}
            Date: {start_date}
            Location: {location_str}
            Coordinates: {coordinates}
            Organizer: {organizer}
            URL: {browser_url}
            Summary: {summary}
            Description: {description}
        """
        return '\n'.join([
            f"Title: {self.title}",
            f"Type: {self.event_type.value}",
            f"Date: {self.timeslots[0].start_date.strftime('%A, %B %d, %Y at %I:%M %p')}",
            f"Location: {self.location_str}",
            f"Coordinates: {self.coordinates}",
            f"Organizer: {self.sponsor.name}",
            f"URL: {self.browser_url}",
            f"Summary: {self.summary}",
            f"Description: {self.description}",
        ])


DATE_KEYS: List[str] = [
    "published",
    "updated",
    "created",
    "date",
    "pubDate",
    "datetimewritten"
]


@dataclass
class RSSFeedItem:
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