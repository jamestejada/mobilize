import asyncio
import logging

from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command, CommandObject

from .settings import TOKEN
from .mobilize import get_events
from .ai import LLM

logging.basicConfig(level=logging.INFO)

bot = Bot(
    token=TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN)
    )
dp = Dispatcher()
llm = LLM()


@dp.message(CommandStart())
@dp.message(Command("help"))
async def send_welcome(message: types.Message):
    await message.reply(
        "Hi! I'm your AI Powered OSINT Bot.\n"
        "\t- Use /events <zipcode> to get upcoming events.\n"
        "\t- Use /analyze <zipcode> <question> to get analysis on events."
    )

async def send_long_message(message: types.Message, text: str, chunk_size: int = 4000):
    """Split long messages into chunks."""
    for i in range(0, len(text), chunk_size):
        await message.answer(text[i:i + chunk_size], parse_mode=ParseMode.MARKDOWN)

@dp.message(Command("analyze"))
async def analyze_events(message: types.Message, command: CommandObject):
    await message.answer("Analyzing events...")
    args = command.args
    if not args or len(args.split()) < 2:
        await message.reply("Please provide a zipcode and your analysis question. Usage: /analyze <zipcode> <question>")
        return
    
    parts = args.split(maxsplit=1)
    zipcode = parts[0]
    user_input = parts[1]

    if not zipcode.isdigit() or not len(zipcode) == 5:
        await message.reply("Please provide a valid 5-digit zipcode. Usage: /analyze <zipcode> <question>")
        return

    await message.answer(
        f"Fetching events for zipcode: *{zipcode}* and analyzing your question..."
        )
    events = await get_events(zipcode)
    if len(events) == 0:
        await message.answer("No upcoming events found in your area.")
        return
    analysis = await llm.provide_analysis(
            user_input=args,
            events=events,
            chat_id=message.chat.id
        )
    print(analysis)
    await send_long_message(message, f"Analysis:\n{analysis}")


@dp.message(Command("events"))
async def send_events(message: types.Message, command: CommandObject):
    args = command.args
    if not args or not args.isdigit():
        await message.reply("Please provide a valid zipcode. Usage: /events <zipcode>")
        return
    
    if not len(args) == 5:
        await message.reply("Please provide a 5-digit zipcode. Usage: /events <zipcode>")
        return

    zipcode = args

    await message.answer(
        f"Fetching protest activity for zipcode: *{zipcode}*..."
        )
    events = await get_events(zipcode)
    if len(events) == 0:
        await message.answer("No upcoming events found in your area.")
        return
    for event in events:
        logging.info(f"Sending event: {event.title}")
        try:
            await message.answer(
                event.telegram_message,
                link_preview_options=types.LinkPreviewOptions(is_disabled=True)
                )
        except Exception as e:
            logging.error(f"Failed to send event message: {e}")

@dp.message()
async def regular_message(message: types.Message):
    await message.answer("Thinking...")
    response = await llm.simple_query(
            user_input=message.text,
            chat_id=message.chat.id
        )
    await send_long_message(message, response)


async def run_bot():
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(run_bot())