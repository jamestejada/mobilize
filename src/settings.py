from dotenv import load_dotenv
import os
from pathlib import Path

import yarl

load_dotenv()


PROMPT_PATH = Path("prompts/")
PROMPT_PATH.mkdir(exist_ok=True)

class RSSPaths:
    FEED_DIR = Path("rss_feeds/")
    FEED_DIR.mkdir(exist_ok=True)

    US_GOV = FEED_DIR.joinpath("gov_feeds.json")
    US_NEWS = FEED_DIR.joinpath("usa_news_feeds.json")
    WORLD_NEWS = FEED_DIR.joinpath("world_news_feeds.json")


class Prompts:
    GENERAL_OSINT = PROMPT_PATH.joinpath("osint_prompt.txt").read_text()


class MobilizeEndpoints:
    API_ROOT = yarl.URL("https://api.mobilize.us/v1")
    EVENTS = API_ROOT.joinpath("events")
    ORGANIZATIONS = API_ROOT.joinpath("organizations/")

class BlueSkyCredentials:
    HANDLE = os.getenv("BSKY_HANDLE")
    APP_PASSWORD = os.getenv("BSKY_APP_PASSWORD")


TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
