import aiohttp
from pydantic_ai import RunContext

from ..ai import AgentDeps


async def get_polymarket_sentiment(
        ctx: RunContext[AgentDeps],
        query: str
        ) -> str:
    """Gets the sentiment of a given query on Polymarket.

    Args:
        query (str): The query to get sentiment for.

    Returns:
        str: The sentiment of the query on Polymarket.
    """
    await ctx.deps.update_chat(f"_Searching polymarket: {query}_")
    url = f"https://api.polymarket.com/v1/markets/search?q={query}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()
            if not data["markets"]:
                return f"No markets found for query: {query}"
            market = data["markets"][0]
            sentiment = market["probability"] * 100
            return f"The sentiment for '{query}' on Polymarket is approximately {sentiment:.2f}%."