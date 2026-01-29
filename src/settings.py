from dotenv import load_dotenv
import os
from pathlib import Path

import yarl

load_dotenv()


PROMPT_PATH = Path("prompts/")
PROMPT_PATH.mkdir(exist_ok=True)

class Prompts:
    EVENT_ANALYSIS = PROMPT_PATH.joinpath("event_prompt.txt").read_text()
    GENERAL_OSINT = PROMPT_PATH.joinpath("osint_prompt.txt").read_text()


class MobilizeEndpoints:
    API_ROOT = yarl.URL("https://api.mobilize.us/v1")
    EVENTS = API_ROOT.joinpath("events")
    ORGANIZATIONS = API_ROOT.joinpath("organizations/")


TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
