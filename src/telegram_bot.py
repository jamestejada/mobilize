import logging
from typing import Awaitable

from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode, ChatAction, ChatType
from aiogram.filters import CommandStart, Command, CommandObject

from .settings import TelegramBotCredentials
from .tools.mobilize import get_events
from .tools.bsky import trending_topics
from .ai import Praetor


logger = logging.getLogger(__name__)


bot = Bot(
    token=TelegramBotCredentials.TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN)
    )
dp = Dispatcher()
llm = Praetor()


@dp.message(CommandStart())
@dp.message(Command("help"))
async def send_welcome(message: types.Message):
    await message.reply(
        "Hi! I'm your AI Powered OSINT Bot.\n\n"
        "\t- Use /protests <location> to get upcoming protests.\n"
        "\t- Use /trending to see current trending topics on Bluesky.\n"
        "\t- Use /clear to reset conversation context."
        " Use this when switching topics or if I'm acting weird.\n"
        "\nOtherwise, @mention me in the group chat"
        " and I'll try to help you with your questions!"
    )


@dp.message(Command("clear"))
async def clear_context(message: types.Message):
    llm.clear(message.chat.id)
    await message.reply("Context cleared.")


def clean_markdown(text: str) -> str:
    """Cleans markdown formatting from text."""
    REPLACEMENTS = [
        ("`", ''),
        ('***', '*'),
        ('**', '*'),
        ('#', ''),
    ]
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    return text


async def send_long_message(
            message: types.Message,
            text: str,
            disable_web_page_preview: bool = False,
            chunk_size: int = 4000
        ):
    """Split long messages into chunks."""
    for i in range(0, len(text), chunk_size):
        chunk = clean_markdown(text[i:i + chunk_size])
        try:
            await message.answer(
                chunk,
                parse_mode=ParseMode.MARKDOWN,
                link_preview_options=types.LinkPreviewOptions(
                    is_disabled=disable_web_page_preview
                    )
                )
        except Exception:
            await message.answer(
                chunk,
                parse_mode=None,
                link_preview_options=types.LinkPreviewOptions(
                    is_disabled=disable_web_page_preview
                )
            )


@dp.message(Command("trending"))
async def send_trending_topics(message: types.Message):
    await message.answer("Fetching trending topics on Bluesky...")
    topics = await trending_topics()
    if not topics:
        await message.answer("No trending topics found on Bluesky.")
        return
    # convert topics to strings
    topics = "\n\n".join([str(topic) for topic in topics])
    await send_long_message(
        message=message,
        text=topics,
        disable_web_page_preview=True
    )


@dp.message(Command("protests"))
async def send_events(message: types.Message, command: CommandObject):
    location = command.args
    if not location:
        await message.reply(
            "Please provide a location (zipcode, address, or city name)."
            " Usage: /events <location>"
            )
        return

    await message.answer(
        f"Fetching protest activity for: *{location}*..."
        )
    events = await get_events(location)
    if len(events) == 0:
        await message.answer("No upcoming protests found in your area.")
        return

    # Batch events into a single message to avoid rate limits
    events_text = f"Found {len(events)} protests around {location}:\n\n"
    events_text += "\n\n------------------\n\n".join(
            [event.telegram_message for event in events]
        )
    await send_long_message(
            message=message,
            text=events_text,
            disable_web_page_preview=True
        )

@dp.message()
async def regular_message(message: types.Message):
    if not message.text:
        # Ignore status updates or non-text messages
        return
    
    if message.chat.type == ChatType.PRIVATE:
        await message.reply("Sorry, I only work in group chats to help reduce spam.")
        return
    
    # only respond when @mentioned in group chats
    bot_info = await bot.get_me()
    bot_username = f"@{bot_info.username}"
    if bot_username.lower() not in message.text.lower():
        return

    await bot.send_chat_action(
            chat_id=message.chat.id,
            action=ChatAction.TYPING,
            request_timeout=10
        )
    user_text= message.text.replace(bot_username, "").strip()

    async def update_callback(text: str) -> Awaitable[None]:
         await send_long_message(
                message=message,
                text=text
                )
    # NOTE: Messaging will be handled from within the LLM
    # class itself to provide real-time updates.
    await llm.handle_query(
        user_input=user_text,
        chat_id=message.chat.id,
        update_chat=update_callback
    )


@dp.startup()
async def on_startup():
    await independent_message(
        "_Oculis Apertis Bot is now online!_"
        )


@dp.shutdown()
async def on_shutdown():
    await independent_message(
        "_Oculis Apertis Bot is shutting down._"
        )


async def independent_message(message: str):
    await bot.send_message(
        chat_id=TelegramBotCredentials.CHANNEL_ID,
        text=message,
    )

async def run_bot():
    await dp.start_polling(bot)
