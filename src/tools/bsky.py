from atproto import AsyncClient
# from langchain_core.tools import tool
from pydantic_ai import RunContext

from dataclasses import dataclass
from typing import Optional

from ..settings import BlueSkyCredentials
from ..ai import AgentDeps


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
    client = await bluesky_login()
    results = await client.app.bsky.feed.search_posts(
        params={
            "q": query,
            "limit": limit
        }
    )
    if not results:
        return "No posts found matching the query."
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
    await ctx.deps.update_chat(f"_Checking {handle} profile_")
    client = await bluesky_login()
    profile = await client.app.bsky.actor.get_profile(params={"actor": handle})
    if not profile:
        return f"No profile found for handle: {handle}"
    return "\n".join([
            f"Profile Information for @{handle}:",
            f"Display Name: {profile.displayName}",
            f"Description: {profile.description}",
            f"Followers: {profile.followersCount}",
            f"Following: {profile.followsCount}",
            f"Posts: {profile.postsCount}"
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
    await ctx.deps.update_chat(f"_Checking {handle}'s feed_")
    client = await bluesky_login()
    results = await client.app.bsky.feed.get_author_feed(
        params={"actor": handle, "limit": limit}
    )
    if not results:
        return f"No posts found for profile: @{handle}"
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
    client = await bluesky_login()
    results = await client.app.bsky.unspecced.get_trending_topics()
    if not results:
        return "No trending topics found."
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