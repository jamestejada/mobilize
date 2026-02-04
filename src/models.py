from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List


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
    end_date: int
    is_full: bool
    instructions: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.start_date, int):
            self.start_date = datetime.fromtimestamp(self.start_date)

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


UNKNOWN_TOOL_NAME = "Unknown Tool"
UNKNOWN_TOOL_ID = "Unknown Tool ID"

@dataclass
class ToolCall:
    # NOTE: This is a mirror of LangChain's ToolCall(TypedDict) class
    # with added validation property.
    """Represents a tool call made by the LLM."""
    name: Optional[str] = UNKNOWN_TOOL_NAME
    args: Optional[dict] = field(default_factory=dict)
    id: Optional[str] = UNKNOWN_TOOL_ID
    type: Optional[str] = None

    @property
    def valid(self) -> bool:
        return bool(
                self.name
                and self.id
                and self.name != UNKNOWN_TOOL_NAME 
                and self.id != UNKNOWN_TOOL_ID
            )
