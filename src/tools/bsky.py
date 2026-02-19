import logging

from atproto import AsyncClient
from atproto_client.exceptions import RequestErrorBase
from pydantic_ai import RunContext
from pydantic import BaseModel

from typing import Optional, List

from ..settings import BlueSkyCredentials
from ..ai import AgentDeps

logger = logging.getLogger(__name__)


class BlueskyPost(BaseModel):
    author_handle: str
    text: str
    post_id: str | int

    @property
    def source_url(self) -> str:
        return f"https://bsky.app/profile/{self.author_handle}/post/{self.post_id}"

    @property
    def url(self) -> str:
        # For protocol compatibility
        return self.source_url

    def __str__(self) -> str:
        return "\n".join([
                f"Author: {self.author_handle}",
                f"Content: {self.text}",
                f"URL: {self.url}",
                "---------"
            ])

    @classmethod
    def from_atproto(cls, post) -> "BlueskyPost":
        post_id = post.uri.split("/")[-1]
        return cls(
            author_handle=post.author.handle,
            text=post.record.text,
            post_id=post_id
        )


class BlueskyProfile(BaseModel):
    handle: str
    display_name: str
    description: Optional[str] = None
    followers_count: int = 0
    follows_count: int = 0
    posts_count: int = 0

    @property
    def source_url(self) -> str:
        return f"https://bsky.app/profile/{self.handle}"

    def __str__(self) -> str:
        return "\n".join([
                f"Profile Information for @{self.handle}:",
                f"Display Name: {self.display_name}",
                f"Description: {self.description or 'N/A'}",
                f"Followers: {self.followers_count}",
                f"Following: {self.follows_count}",
                f"Posts: {self.posts_count}"
            ])


class BlueskyTrendingTopic(BaseModel):
    topic: str
    link: str
    display_name: Optional[str] = None
    description: Optional[str] = None

    @property
    def source_url(self) -> str:
        # Link to search results for this trending topic
        from urllib.parse import quote
        return f"https://bsky.app/search?q={quote(self.topic)}"

    @property
    def title(self) -> str:
        """Returns display_name if available, otherwise topic."""
        return self.display_name if self.display_name else self.topic

    @property
    def summary(self) -> str:
        """Returns description or empty string for SourceDataBuilder."""
        return self.description if self.description else ""

    @property
    def source_name(self) -> str:
        """Returns 'Bluesky Trending' as the source."""
        return "Bluesky Trending"

    @property
    def feed_url(self) -> str:
        """Returns full URL for the trending topic feed."""
        return f"https://bsky.app{self.link}" if self.link.startswith('/') else self.link

    def __str__(self) -> str:
        # Use display_name if available, otherwise fall back to topic
        name = self.title
        lines = [f"Topic: {name}", f"Feed: {self.feed_url}"]
        if self.description:
            lines.append(f"Description: {self.description}")
        return "\n".join(lines)

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



async def search_bluesky_posts(
        ctx: RunContext[AgentDeps], query: str, limit: int = 30
        ) -> List[BlueskyPost]:
    """Searches for posts on Bluesky containing the given query.

    Args:
        query (str): The search query text. Example: "climate change" or "tariffs economy"
        limit (int, optional): Maximum number of posts to return. Defaults to 30.

    Returns:
        List[BlueskyPost]: List of posts matching the query.

    Example:
        search_bluesky_posts(query="artificial intelligence", limit=20)
    """
    await ctx.deps.update_chat(f"_Searching Bluesky Posts: {query}_")
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
            return []
    except RequestErrorBase as e:
        logger.error(f"Bluesky search failed for '{query}': {e}")
        return []

    posts = [BlueskyPost.from_atproto(post) for post in results.posts]
    return posts



async def get_bluesky_profile(
            ctx: RunContext[AgentDeps],
            handle: str
        ) -> BlueskyProfile | None:
    """Fetches and returns information about a Bluesky profile.

    Args:
        handle (str): The profile handle. Formats: "@username.bsky.social", "username.bsky.social", or "username"

    Returns:
        BlueskyProfile | None: Profile info or None if not found.

    Example:
        get_bluesky_profile(handle="journalist.bsky.social")
    """
    handle = sanitize_handle(handle)
    await ctx.deps.update_chat(f"_Checking {handle} profile_")
    try:
        client = await bluesky_login()
        profile = await client.app.bsky.actor.get_profile(
                params={"actor": handle}
            )
    except RequestErrorBase as e:
        logger.error(f"Bluesky profile lookup failed for '{handle}': {e}")
        return None
    if not profile:
        logger.warning(f"No Bluesky profile found for handle: {handle}")
        return None

    return BlueskyProfile(
        handle=handle,
        display_name=profile.display_name or "",
        description=profile.description,
        followers_count=profile.followers_count or 0,
        follows_count=profile.follows_count or 0,
        posts_count=profile.posts_count or 0
    )



async def get_author_feed(ctx: RunContext[AgentDeps], handle: str, limit: int = 30) -> List[BlueskyPost]:
    """Fetches and returns recent posts from a Bluesky profile.

    Args:
        handle (str): The profile handle (same formats as get_bluesky_profile)
        limit (int, optional): Maximum posts to return. Defaults to 30.

    Returns:
        List[BlueskyPost]: List of recent posts.

    Example:
        get_author_feed(handle="newsaccount.bsky.social", limit=20)
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
            return []
    except RequestErrorBase as e:
        logger.error(f"Bluesky author feed failed for '{handle}': {e}")
        return []

    posts = [BlueskyPost.from_atproto(item.post) for item in results.feed]
    return posts


async def trending_topics() -> List[BlueskyTrendingTopic]:
    """Fetches and returns the current trending topics on Bluesky.

    Returns:
        List[BlueskyTrendingTopic]: List of trending topics.
    """
    try:
        client = await bluesky_login()
        results = await client.app.bsky.unspecced.get_trending_topics()

        # NOTE: Could add fallback to results.suggested if results.topics is empty
        if not results.topics:
            logger.info("No trending topics available on Bluesky")
            return []

    except RequestErrorBase as e:
        logger.error(f"Bluesky trending topics API call failed: {e}")
        return []

    return [
        BlueskyTrendingTopic(
            topic=topic.topic,
            link=topic.link,
            display_name=getattr(topic, 'display_name', None),
            description=getattr(topic, 'description', None)
        )
        for topic in results.topics
    ]



async def get_trending_topics(ctx: RunContext[AgentDeps]) -> List[BlueskyTrendingTopic]:
    """Fetches and returns the current trending topics on Bluesky.

    No parameters required.

    Returns:
        List[BlueskyTrendingTopic]: List of trending topics with search URLs.

    Example:
        get_trending_topics()
    """
    await ctx.deps.update_chat("_Checking Bluesky trending topics_")
    return await trending_topics()