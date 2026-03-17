import logging
import re
from typing import Awaitable

import asyncio

from aiogram import Bot, Dispatcher, types, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode, ChatAction, ChatType
from aiogram.exceptions import TelegramNetworkError
from aiogram.filters import CommandStart, Command, CommandObject

from .settings import TelegramBotCredentials
from .tools.mobilize import get_events
from .tools.bsky import trending_topics
from .ai import Praetor, training_logger
from .training_logger import emoji_to_rating


logger = logging.getLogger(__name__)


def is_authorized(message: types.Message) -> bool:
    if not TelegramBotCredentials.ALLOWED_USER_IDS:
        return True     # Defaulting to True just for now
                        # until I can find a good way to get user ids
        # return False
    return message.from_user is not None and message.from_user.id in TelegramBotCredentials.ALLOWED_USER_IDS

bot = Bot(
    token=TelegramBotCredentials.TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
dp = Dispatcher()
llm = Praetor()

# Track running query tasks per chat so they can be cancelled
_running_tasks: dict[int, asyncio.Task] = {}



@dp.message(CommandStart())
@dp.message(Command("help"))
async def send_welcome(message: types.Message):
    if not is_authorized(message):
        return
    await message.reply(
        "Hi! I'm your AI Powered OSINT Bot.\n\n"
        "\t- Use /protests &lt;location&gt; to get upcoming protests.\n"
        "\t- Use /trending to see current trending topics on Bluesky.\n\n"
        "\nOtherwise, @mention me in the group chat"
        " and I'll try to help you with your questions!\n\n"
        "Use these commands after your query to manage context and sources:\n"
        "\t- Use /sources to see collected sources from the last query.\n"
        "\t- Use /stop to cancel a running query.\n"
        "\t- Use /clear to reset conversation context."
        " Use this when switching topics or if I'm acting weird.\n\n"
        "<b>Rate my responses</b> with emoji reactions:\n"
        "\t🔥 💯 🥰 🤩 ❤ = Great response\n"
        "\t👍 = Okay / mediocre\n"
        "\t🤬 👎 = Bad response\n"
        "Your ratings help improve future responses!"
    )


@dp.message(Command("clear"))
async def clear_context(message: types.Message):
    if not is_authorized(message):
        return
    llm.clear(message.chat.id)
    await message.reply("Context cleared.")


@dp.message(Command("sources"))
async def send_sources(message: types.Message):
    if not is_authorized(message):
        return
    text = llm.get_sources_by_tg_command(message.chat.id)
    if not text:
        await message.reply("No sources available. Run a query first.")
        return
    await send_long_message(message=message, text=text, disable_web_page_preview=True)


@dp.message(Command("stop"))
async def stop_query(message: types.Message):
    if not is_authorized(message):
        return
    task = _running_tasks.get(message.chat.id)
    if task and not task.done():
        task.cancel()
        await message.reply("_Stopping current query..._", parse_mode=ParseMode.MARKDOWN)
    else:
        await message.reply("No query is running.")


def markdown_to_html(text: str) -> str:
    """Convert markdown formatting to Telegram HTML."""
    def escape(s: str) -> str:
        return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    # Split on markdown links so URLs are handled separately from prose
    text = text.replace('`', '') # Remove backticks
    parts = re.split(r'(\[[^\]]+\]\(https?://[^\)]+\))', text)
    result = []
    for part in parts:
        m = re.fullmatch(r'\[([^\]]+)\]\((https?://[^\)]+)\)', part)
        if m:
            link_text = escape(m.group(1))
            url = m.group(2).replace('&', '&amp;')
            result.append(f'<a href="{url}">{link_text}</a>')
        else:
            part = escape(part)
            part = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b>\1</b>', part)
            part = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', part)
            part = re.sub(r'\*([^*\n]+)\*', r'<b>\1</b>', part)
            part = re.sub(r'_([^_\n]+)_', r'<i>\1</i>', part)
            part = re.sub(r'`([^`]+)`', r'<code>\1</code>', part)
            part = re.sub(r'^#{1,6}\s+', '', part, flags=re.MULTILINE)
            result.append(part)
    return ''.join(result)


async def send_long_message(
            message: types.Message,
            text: str,
            disable_web_page_preview: bool = False,
            chunk_size: int = 4000
        ) -> list[int]:
    """Split long messages into chunks. Returns list of sent message IDs."""
    sent_ids: list[int] = []

    async def send_chunk(chunk: str):
        for attempt in range(3):
            try:
                msg = await message.answer(
                    markdown_to_html(chunk),
                    parse_mode=ParseMode.HTML,
                    link_preview_options=types.LinkPreviewOptions(
                        is_disabled=disable_web_page_preview
                        )
                    )
                sent_ids.append(msg.message_id)
                break
            except TelegramNetworkError:
                if attempt < 2:
                    logger.warning("Telegram send failed, retrying in 5s (attempt %d/3)", attempt + 1)
                    await asyncio.sleep(5)
                else:
                    logger.error("Failed to send message chunk after 3 attempts", exc_info=True)

    paragraphs_split = text.split('\n\n')
    chunk = ""
    for paragraph in paragraphs_split:
        if len(paragraph) > chunk_size:
            # send the current chunk because we can't add anymore to it.
            # then split the long paragraph directly into more chunks
            if chunk:
                await send_chunk(chunk)
                chunk = ""
            # If a single paragraph exceeds the chunk size, split it directly
            for i in range(0, len(paragraph), chunk_size):
                await send_chunk(paragraph[i:i + chunk_size])
        elif len(chunk) + len(paragraph) > chunk_size:
            await send_chunk(chunk)
            chunk = paragraph + "\n\n"
        else:
            chunk += paragraph + "\n\n"
    if chunk:
        await send_chunk(chunk)
    return sent_ids


@dp.message(Command("trending"))
async def send_trending_topics(message: types.Message):
    if not is_authorized(message):
        return
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
    if not is_authorized(message):
        return
    location = command.args
    if not location:
        await message.reply(
            "Please provide a location (zipcode, address, or city name)."
            " Usage: /events <location>"
            )
        return

    await message.answer(
        f"Fetching protest activity for: <b>{location}</b>..."
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
        return
    if not is_authorized(message):
        return
    if message.chat.type == ChatType.PRIVATE:
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

    # Include replied-to message content for context
    reply = message.reply_to_message
    if reply and reply.text:
        reply_text = reply.text[:500]
        user_text = f'[Replying to: "{reply_text}"]\n\n{user_text}'

    last_message_ids: list[int] = []

    async def update_callback(text: str) -> None:
        nonlocal last_message_ids
        last_message_ids = await send_long_message(
                message=message,
                text=text
                )

    async def run_query():
        nonlocal last_message_ids
        iid, path = await llm.handle_query(
            user_input=user_text,
            chat_id=message.chat.id,
            update_chat=update_callback
        )
        if iid and last_message_ids:
            training_logger.associate_messages(
                iid, message.chat.id, last_message_ids, path=path,
            )

    chat_id = message.chat.id
    # Cancel any existing query in this chat
    existing = _running_tasks.get(chat_id)
    if existing and not existing.done():
        existing.cancel()

    task = asyncio.create_task(run_query())
    _running_tasks[chat_id] = task
    try:
        await task
    except asyncio.CancelledError:
        await message.reply("_Query stopped._", parse_mode=ParseMode.MARKDOWN)
    finally:
        _running_tasks.pop(chat_id, None)


@dp.startup()
async def on_startup():
    await independent_message(
        "<i>Oculis Apertis Bot is now online!</i>"
        )


@dp.shutdown()
async def on_shutdown():
    await independent_message(
        "<i>Oculis Apertis Bot is shutting down.</i>"
        )


async def independent_message(message: str):
    await bot.send_message(
        chat_id=TelegramBotCredentials.CHANNEL_ID,
        text=message,
    )


# Handle message reactions for training data ratings
@dp.message_reaction()
async def handle_reaction_update(event: types.MessageReactionUpdated):
    for reaction in event.new_reaction:
        emoji = getattr(reaction, "emoji", None)
        if not emoji:
            continue
        rating = emoji_to_rating(emoji)
        if rating:
            training_logger.rate_by_message(
                event.chat.id, event.message_id, rating
            )
            break


async def run_bot():
    await dp.start_polling(bot)
