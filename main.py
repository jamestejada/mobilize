from src.telegram_bot import run_bot
import asyncio






if __name__ == "__main__":
    # from src.mobilize import get_events

    # events = asyncio.run(get_events(zipcode="55401"))
    # for event in events[:5]:
    #     location = event.location.venue\
    #         if not event.location.location\
    #             or not event.location.location.latitude\
    #     else f"{event.location.location.latitude}, {event.location.location.longitude}"
    #     print(event.title, location)
    asyncio.run(run_bot())
