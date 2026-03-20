from datetime import datetime
from enum import Enum
from typing import Optional, List
import logging

from pydantic import BaseModel, field_validator
from dateutil.parser import parse as parse_date

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


class Sponsor(BaseModel):
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


class Coordinates(BaseModel):
    latitude: float
    longitude: float


class Location(BaseModel):
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

    @field_validator('location', mode='before')
    @classmethod
    def coerce_location(cls, v):
        if isinstance(v, dict):
            return Coordinates(**v)
        return v


class Timeslot(BaseModel):
    id: int
    start_date: datetime
    end_date: datetime
    is_full: bool
    instructions: Optional[str] = None

    @field_validator('start_date', 'end_date', mode='before')
    @classmethod
    def coerce_timestamps(cls, v):
        if isinstance(v, int):
            return datetime.fromtimestamp(v)
        return v

class Event(BaseModel):
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
    tag: str = ""

    @field_validator('event_type', mode='before')
    @classmethod
    def coerce_event_type(cls, v):
        if isinstance(v, str):
            return EventType(v)
        return v

    @field_validator('sponsor', mode='before')
    @classmethod
    def coerce_sponsor(cls, v):
        if isinstance(v, dict):
            return Sponsor(**v)
        return v

    @field_validator('location', mode='before')
    @classmethod
    def coerce_location(cls, v):
        if isinstance(v, dict):
            return Location(**v)
        return v

    @field_validator('timeslots', mode='before')
    @classmethod
    def coerce_timeslots(cls, v):
        if v and isinstance(v[0], dict):
            return [Timeslot(**t) for t in v]
        return v

    @field_validator('tags', mode='before')
    @classmethod
    def coerce_tags(cls, v):
        if v and isinstance(v[0], dict):
            return [tag.get('name', '') for tag in v]
        return v

    @property
    def source_url(self) -> str:
        return self.browser_url

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
