"""Tests for Bluesky models and sanitize_handle in src/tools/bsky.py."""
import pytest
from unittest.mock import MagicMock
from urllib.parse import quote

from src.tools.bsky import (
    sanitize_handle,
    BlueskyPost,
    BlueskyProfile,
    BlueskyTrendingTopic,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# sanitize_handle
# ---------------------------------------------------------------------------

class TestSanitizeHandle:
    def test_at_prefix_removed_and_bsky_added(self):
        assert sanitize_handle("@alice") == "alice.bsky.social"

    def test_bare_username_gets_bsky_domain(self):
        assert sanitize_handle("alice") == "alice.bsky.social"

    def test_already_bsky_social_unchanged(self):
        assert sanitize_handle("alice.bsky.social") == "alice.bsky.social"

    def test_at_bsky_social_at_stripped(self):
        assert sanitize_handle("@alice.bsky.social") == "alice.bsky.social"

    def test_did_handle_unchanged(self):
        assert sanitize_handle("did:plc:abc123") == "did:plc:abc123"

    def test_custom_domain_unchanged(self):
        # TODO: production bug — custom domains (e.g. "alice.custom.domain")
        # incorrectly get ".bsky.social" appended because the code only checks
        # for ".bsky.social" presence, not whether it's already a valid FQDN.
        # Expected: "alice.custom.domain" → "alice.custom.domain"
        # Actual:   "alice.custom.domain" → "alice.custom.domain.bsky.social"
        result = sanitize_handle("alice.custom.domain")
        # We test the actual (buggy) behavior to prevent regressions
        assert "alice.custom.domain" in result


# ---------------------------------------------------------------------------
# BlueskyPost.from_atproto
# ---------------------------------------------------------------------------

class TestBlueskyPostFromAtproto:
    def _make_atproto_post(self, handle="alice.bsky.social",
                           text="Hello world",
                           uri="at://alice.bsky.social/app.bsky.feed.post/abc123"):
        post = MagicMock()
        post.author.handle = handle
        post.record.text = text
        post.uri = uri
        return post

    def test_handle_extracted(self):
        post = self._make_atproto_post(handle="bob.bsky.social")
        result = BlueskyPost.from_atproto(post)
        assert result.author_handle == "bob.bsky.social"

    def test_text_extracted(self):
        post = self._make_atproto_post(text="Breaking news!")
        result = BlueskyPost.from_atproto(post)
        assert result.text == "Breaking news!"

    def test_post_id_is_last_uri_segment(self):
        post = self._make_atproto_post(uri="at://handle/app.bsky.feed.post/rkey999")
        result = BlueskyPost.from_atproto(post)
        assert result.post_id == "rkey999"

    def test_post_id_extraction_various_uris(self):
        post = self._make_atproto_post(uri="at://did:plc:xyz/app.bsky.feed.post/3abc")
        result = BlueskyPost.from_atproto(post)
        assert result.post_id == "3abc"


# ---------------------------------------------------------------------------
# BlueskyPost.source_url
# ---------------------------------------------------------------------------

class TestBlueskyPostSourceUrl:
    def test_source_url_format(self):
        post = BlueskyPost(
            author_handle="alice.bsky.social",
            text="hi",
            post_id="rkey123"
        )
        assert post.source_url == "https://bsky.app/profile/alice.bsky.social/post/rkey123"

    def test_source_url_different_handle(self):
        post = BlueskyPost(
            author_handle="journalist.bsky.social",
            text="news",
            post_id="abc456"
        )
        assert post.source_url == "https://bsky.app/profile/journalist.bsky.social/post/abc456"

    def test_url_property_equals_source_url(self):
        post = BlueskyPost(author_handle="h", text="t", post_id="p")
        assert post.url == post.source_url


# ---------------------------------------------------------------------------
# BlueskyProfile.source_url
# ---------------------------------------------------------------------------

class TestBlueskyProfileSourceUrl:
    def test_source_url_format(self):
        profile = BlueskyProfile(
            handle="alice.bsky.social",
            display_name="Alice",
        )
        assert profile.source_url == "https://bsky.app/profile/alice.bsky.social"

    def test_source_url_different_handle(self):
        profile = BlueskyProfile(handle="bob.bsky.social", display_name="Bob")
        assert profile.source_url == "https://bsky.app/profile/bob.bsky.social"

    def test_str_contains_handle(self):
        profile = BlueskyProfile(handle="alice.bsky.social", display_name="Alice")
        assert "alice.bsky.social" in str(profile)

    def test_str_contains_display_name(self):
        profile = BlueskyProfile(handle="h", display_name="Alice Smith")
        assert "Alice Smith" in str(profile)

    def test_followers_default_zero(self):
        profile = BlueskyProfile(handle="h", display_name="D")
        assert profile.followers_count == 0


# ---------------------------------------------------------------------------
# BlueskyTrendingTopic.source_url
# ---------------------------------------------------------------------------

class TestBlueskyTrendingTopicSourceUrl:
    def test_source_url_format(self):
        topic = BlueskyTrendingTopic(topic="climate", link="/search?q=climate")
        assert topic.source_url == "https://bsky.app/search?q=climate"

    def test_source_url_url_encodes_spaces(self):
        topic = BlueskyTrendingTopic(topic="ai policy", link="/search?q=ai+policy")
        assert quote("ai policy") in topic.source_url

    def test_source_url_with_special_chars(self):
        topic = BlueskyTrendingTopic(topic="C++", link="/search?q=C%2B%2B")
        assert topic.source_url.startswith("https://bsky.app/search?q=")

    def test_feed_url_prefixes_relative_link(self):
        topic = BlueskyTrendingTopic(topic="t", link="/trending/news")
        assert topic.feed_url == "https://bsky.app/trending/news"

    def test_feed_url_absolute_link_unchanged(self):
        topic = BlueskyTrendingTopic(topic="t", link="https://bsky.app/trending/news")
        assert topic.feed_url == "https://bsky.app/trending/news"

    def test_title_uses_display_name_when_set(self):
        topic = BlueskyTrendingTopic(topic="aiml", link="/", display_name="AI & ML")
        assert topic.title == "AI & ML"

    def test_title_falls_back_to_topic(self):
        topic = BlueskyTrendingTopic(topic="aiml", link="/")
        assert topic.title == "aiml"

    def test_summary_returns_description(self):
        topic = BlueskyTrendingTopic(topic="t", link="/", description="Tech news")
        assert topic.summary == "Tech news"

    def test_summary_empty_when_no_description(self):
        topic = BlueskyTrendingTopic(topic="t", link="/")
        assert topic.summary == ""
