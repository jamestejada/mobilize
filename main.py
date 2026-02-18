from src.logging_config import setup_logging
from src.telegram_bot import run_bot
from src.tools.fetch_url import close_browser
import asyncio


async def main():
    try:
        await run_bot()
    finally:
        await close_browser()


if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
