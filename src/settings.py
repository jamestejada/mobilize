import os
import json
from pathlib import Path
from dotenv import load_dotenv

import yarl
from dateutil.tz import gettz

load_dotenv()


PROMPT_PATH = Path("prompts/")
PROMPT_PATH.mkdir(exist_ok=True)


TZ_INFOS = {
    "UTC": gettz("UTC"),
    "EST": gettz("America/New_York"),
    "EDT": gettz("America/New_York"),
    "CST": gettz("America/Chicago"),
    "CDT": gettz("America/Chicago"),
    "MST": gettz("America/Denver"),
    "MDT": gettz("America/Denver"),
    "PST": gettz("America/Los_Angeles"),
    "PDT": gettz("America/Los_Angeles"),
}


class RSS:
    FEED_DIR = Path("rss_feeds/")
    FEED_DIR.mkdir(exist_ok=True)

    US_GOV = FEED_DIR.joinpath("gov_feeds.json")
    US_NEWS = FEED_DIR.joinpath("usa_news_feeds.json")
    WORLD_NEWS = FEED_DIR.joinpath("world_news_feeds.json")

    US_GOV_JSON = json.loads(US_GOV.read_text())
    US_NEWS_JSON = json.loads(US_NEWS.read_text())
    WORLD_NEWS_JSON = json.loads(WORLD_NEWS.read_text())


class Prompts:
    COORDINATOR = PROMPT_PATH.joinpath("coordinator.md").read_text()
    REFLECTION = PROMPT_PATH.joinpath("reflection.md").read_text()
    RESEARCHER = PROMPT_PATH.joinpath("researcher.md").read_text()
    WRITER = PROMPT_PATH.joinpath("writer.md").read_text()


class OllamaEndpoints:
    API_ROOT = yarl.URL(os.getenv("OLLAMA_ROOT_URL"))
    EMBEDDINGS = API_ROOT.joinpath("api/embeddings")


class MobilizeEndpoints:
    API_ROOT = yarl.URL(os.getenv("MOBILIZE_US_ROOT_URL"))
    EVENTS = API_ROOT.joinpath("events")
    ORGANIZATIONS = API_ROOT.joinpath("organizations/")


class BlueSkyCredentials:
    HANDLE = os.getenv("BSKY_HANDLE")
    APP_PASSWORD = os.getenv("BSKY_APP_PASSWORD")

class TelegramBotCredentials:
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    CHANNEL_ID = int(os.getenv("BOT_CHANNEL_ID"))

class Models:
    Research_Coordinate = os.getenv("RESEARCH_COORDINATE_MODEL", "qwen3:8b-q4_K_M")
    Reflect_Write = os.getenv("REFLECT_WRITE_MODEL", "qwen2.5:14b-instruct-q4_k_m")

MAX_HISTORY = int(os.getenv("MAX_HISTORY", 30))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_CTX_SIZE", 8192))