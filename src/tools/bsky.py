from atproto import AsyncClient
from langchain_core.tools import tool

from dataclasses import dataclass

from ..settings import BlueSkyCredentials


@dataclass
class BlueskyPost:
    author_handle: str
    text: str
    id: str

    def __str__(self) -> str:
        return "\n".join([
                f"Author: {self.author_handle}",
                f"Content: {self.text}",
                f"URL: {self.url}",
                # f"Post ID: {self.id}",
                "---------"
            ])

    @property
    def url(self) -> str:
        return f"https://bsky.app/profile/{self.author_handle}/post/{self.id}"

    @classmethod
    def from_atproto(cls, post) -> "BlueskyPost":
        id = post.uri.split("/")[-1]
        return cls(
            author_handle=post.author.handle,
            text=post.record.text,
            id=id
        )


@tool
async def search_bluesky_posts(query: str, limit: int = 30) -> str:
    """Searches for posts on Bluesky containing the given query.

    Args:
        query (str): The search query.
        limit (int, optional): The maximum number of posts to return. Defaults to 30.

    Returns:
        str: A formatted string of the found posts.
    """
    # client = AsyncClient(base_url="https://public.api.bsky.app")
    client = AsyncClient()
    await client.login(
        login=BlueSkyCredentials.HANDLE,
        password=BlueSkyCredentials.APP_PASSWORD
    )
    results = await client.app.bsky.feed.search_posts(
        params={
            "q": query,
            "limit": limit
        }
    )
    if not results:
        return "No posts found matching the query."

    formatted_posts = [
            str(BlueskyPost.from_atproto(post))
            for post in results.posts
        ]

    return "\n".join(formatted_posts)
