import logging

from atproto import AsyncClient
from atproto_client.exceptions import RequestErrorBase
from pydantic_ai import RunContext

from dataclasses import dataclass
from typing import Optional

from ..settings import BlueSkyCredentials
from ..ai import AgentDeps

logger = logging.getLogger(__name__)


@dataclass
class BlueskyPost:
    author_handle: str
    text: str
    post_id: str | int

    def __str__(self) -> str:
        return "\n".join([
                f"Author: {self.author_handle}",
                f"Content: {self.text}",
                f"URL: {self.url}",
                "---------"
            ])

    @property
    def url(self) -> str:
        return f"https://bsky.app/profile/{self.author_handle}/post/{self.post_id}"

    @classmethod
    def from_atproto(cls, post) -> "BlueskyPost":
        post_id = post.uri.split("/")[-1]
        return cls(
            author_handle=post.author.handle,
            text=post.record.text,
            post_id=post_id
        )

async def bluesky_login() -> AsyncClient:
    client = AsyncClient()
    await client.login(
        login=BlueSkyCredentials.HANDLE,
        password=BlueSkyCredentials.APP_PASSWORD
    )
    return client


def sanitize_handle(handle: str) -> str:
    """Sanitizes a Bluesky handle by removing '@' if present."""
    if 'did:plc:' in handle:
        return handle  # Assume it's already a DID and return as is
    handle = handle.strip().lstrip('@')
    if not '.bsky.social' in handle:
        handle += '.bsky.social'
    return handle



async def search_bluesky_posts(ctx: RunContext[AgentDeps], query: str, limit: int = 30) -> str:
    """Searches for posts on Bluesky containing the given query.

    Args:
        query (str): The search query.
        limit (int, optional): The maximum number of posts
            to return. Defaults to 30.

    Returns:
        str: A formatted string of the found posts.
    """
    await ctx.deps.update_chat("_Searching Bluesky Posts_")
    try:
        client = await bluesky_login()
        results = await client.app.bsky.feed.search_posts(
            params={
                "q": query,
                "limit": limit
            }
        )
        if not results:
            logger.warning(f"No Bluesky posts found for query: '{query}'")
            return ""
    except RequestErrorBase as e:
        logger.error(f"Bluesky search failed for '{query}': {e}")
        return ""
    formatted_posts = [
        "Here are the Bluesky posts found:",
        "---------"
        ] + [
            str(BlueskyPost.from_atproto(post))
            for post in results.posts
        ]

    return "\n".join(formatted_posts)



async def get_bluesky_profile(ctx: RunContext[AgentDeps], handle: str) -> str:
    """Fetches and returns information about a Bluesky profile.

    Args:
        handle (str): The handle of the Bluesky profile to fetch.

    Returns:
        str: A formatted string of the profile information."""
    handle = sanitize_handle(handle)
    await ctx.deps.update_chat(f"_Checking {handle} profile_")
    try:
        client = await bluesky_login()
        profile = await client.app.bsky.actor.get_profile(
                params={"actor": handle}
            )
    except RequestErrorBase as e:
        logger.error(f"Bluesky profile lookup failed for '{handle}': {e}")
        return ""
    if not profile:
        logger.warning(f"No Bluesky profile found for handle: {handle}")
        return ""
    return "\n".join([
            f"Profile Information for @{handle}:",
            f"Display Name: {profile.display_name}",
            f"Description: {profile.description}",
            f"Followers: {profile.followers_count}",
            f"Following: {profile.follows_count}",
            f"Posts: {profile.posts_count}"
        ])



async def get_author_feed(ctx: RunContext[AgentDeps], handle: str, limit: int = 30) -> str:
    """Fetches and returns recent posts from a Bluesky profile.

    Args:
        handle (str): The handle of the Bluesky profile to fetch posts from.
        limit (int, optional): The maximum number of posts
            to return. Defaults to 30.

    Returns:
        str: A formatted string of the recent posts from the profile.
    """
    handle = sanitize_handle(handle)
    await ctx.deps.update_chat(f"_Checking {handle}'s feed_")
    try:
        client = await bluesky_login()
        results = await client.app.bsky.feed.get_author_feed(
            params={"actor": handle, "limit": limit}
        )
        if not results:
            logger.warning(f"No posts found for Bluesky profile: @{handle}")
            return ""
    except RequestErrorBase as e:
        logger.error(f"Bluesky author feed failed for '{handle}': {e}")
        return ""
    formatted_posts = [
        f"Recent posts from @{handle}:",
        "---------"
        ] + [
            str(BlueskyPost.from_atproto(item.post))
            for item in results.feed
        ]

    return "\n".join(formatted_posts)


async def trending_topics() -> str:
    """Fetches and returns the current trending topics on Bluesky.

    Returns:
        str: A formatted string of the current trending topics on Bluesky.
    """
    try:
        client = await bluesky_login()
        results = await client.app.bsky.unspecced.get_trending_topics()
        if not results:
            logger.warning("No trending topics found on Bluesky")
            return ""
    except RequestErrorBase as e:
        logger.error(f"Bluesky trending topics failed: {e}")
        return ""
    formatted_topics = [
        "Current trending topics on Bluesky:",
        "---------"
        ] + [
            f"{idx + 1}. {topic.topic} (Posts: {topic.link})\n"
            f"{'No description available.' if not topic.description else topic.description}\n\n"
            "---------"
            for idx, topic in enumerate(results.topics)
        ]

    return "\n".join(formatted_topics)



async def get_trending_topics(ctx: RunContext[AgentDeps]) -> str:
    """Fetches and returns the current trending topics on Bluesky.

    Returns:
        str: A formatted string of the current trending topics on Bluesky.
    """
    await ctx.deps.update_chat("_Checking Bluesky trending topics_")
    return await trending_topics()