from pydantic_ai import RunContext

from ..ai import AgentDeps


async def get_registered_sources(ctx: RunContext[AgentDeps]) -> str:
    """Returns [SOURCE_N] tags and descriptions already collected in this conversation.

    Call this at the start of a research task to see what has already been
    found. Use the results to avoid duplicate searches and to get URLs you
    can pass directly to fetch_url. Even when prior sources exist, still
    search for recent developments — the situation may have changed since
    they were collected.

    Args:
        (no parameters)

    Returns:
        str: List of [SOURCE_N] tags with titles, URLs, and short descriptions,
             or a message indicating no sources have been collected yet.

    Example:
        get_registered_sources()
    """
    if ctx.deps.source_registry is None:
        return "No sources have been collected yet."
    return ctx.deps.source_registry.format_for_agent_semantic()
