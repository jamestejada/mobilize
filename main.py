# from src.settings import RSS
from src.telegram_bot import run_bot
import asyncio


# async def main():
#     from src.tools.rss_feeds import get_feeds
    
#     feeds = await get_feeds(
#             RSS.US_GOV_JSON,
#             title_match="Immigration"
#         )
#     print(feeds)



if __name__ == "__main__":
    asyncio.run(run_bot())
    # asyncio.run(main())
