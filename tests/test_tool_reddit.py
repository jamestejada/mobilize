"""Tests for RedditPost model in src/tools/reddit.py."""
import pytest
from src.tools.reddit import RedditPost

pytestmark = pytest.mark.unit


class TestRedditPostSourceUrl:
    def test_source_url_prefixes_reddit_com(self):
        post = RedditPost(title="t", permalink="/r/politics/comments/abc/title/")
        assert post.source_url == "https://www.reddit.com/r/politics/comments/abc/title/"

    def test_source_url_empty_permalink(self):
        post = RedditPost(title="t", permalink="")
        assert post.source_url == "https://www.reddit.com"

    def test_source_url_various_subreddits(self):
        post = RedditPost(title="t", permalink="/r/news/comments/xyz/something/")
        assert post.source_url.startswith("https://www.reddit.com")
        assert "/r/news/comments/xyz/" in post.source_url


class TestRedditPostSelftext:
    def test_selftext_stored_as_is_under_500(self):
        post = RedditPost(title="t", selftext="short text")
        assert post.selftext == "short text"

    def test_selftext_can_be_500_chars(self):
        text = "x" * 500
        post = RedditPost(title="t", selftext=text)
        assert len(post.selftext) == 500

    def test_selftext_truncation_is_enforced_by_caller(self):
        # The model itself does not truncate — truncation happens in the tool function.
        # A RedditPost created directly with 600 chars keeps all 600.
        long_text = "y" * 600
        post = RedditPost(title="t", selftext=long_text)
        # Model stores whatever is given — caller slices to [:500]
        assert len(post.selftext) == 600

    def test_defaults(self):
        post = RedditPost(title="headline")
        assert post.subreddit == ""
        assert post.score == 0
        assert post.selftext == ""
        assert post.tag == ""
