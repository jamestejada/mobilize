import re
import logging
from dataclasses import dataclass
from typing import (
        Protocol,
        runtime_checkable,
        List,
        Tuple,
        Set,
        Any,
        Optional
    )


@dataclass
class SourceItem:
    """A registered source with URL and title."""
    url: str
    title: str = "Source"
    source_name: str = ""


@runtime_checkable
class Sourceable(Protocol):
    """Any model with a source_url property."""
    @property
    def source_url(self) -> str: ...


class SourceRegistry:
    """Registry that maps [SOURCE_N] placeholders to real URLs.

    Used to eliminate hallucinated URLs by having the LLM reference
    numbered placeholders instead of copying long URLs verbatim.
    Python then does deterministic substitution after generation.

    Bounded to MAX_SIZE entries; oldest entry is evicted on overflow.
    The counter never resets so evicted tag numbers are never reused.
    """

    MAX_SIZE = 500

    @staticmethod
    def _make_counter():
        n = 0
        while True:
            n += 1
            yield n

    def __init__(self):
        self._sources: dict[str, SourceItem] = {}  # tag -> SourceItem (insertion-ordered)
        self._url_to_tag: dict[str, str] = {}      # url -> tag (for deduplication)
        self._counter = self._make_counter()
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def counter(self) -> int:
        return next(self._counter)

    def register(self, url: str, title: str = "", source_name: str = "") -> str:
        """Register a URL and return its placeholder tag.

        Args:
            url: The URL to register
            title: Optional title for markdown link text

        Returns:
            The placeholder tag (e.g., "[SOURCE_1]")
        """
        if not url or not url.strip():
            return ""

        url = url.strip()

        # Deduplicate: return existing tag if URL already registered
        if url in self._url_to_tag:
            return self._url_to_tag[url]

        # Evict oldest entry if at capacity
        if len(self._sources) >= self.MAX_SIZE:
            oldest_tag, oldest_item = next(iter(self._sources.items()))
            del self._sources[oldest_tag]
            del self._url_to_tag[oldest_item.url]
            self.logger.debug(f"Registry full ({self.MAX_SIZE}): evicted {oldest_tag}")

        tag = f"[SOURCE_{self.counter}]"

        item = SourceItem(url=url, title=title if title else "Source", source_name=source_name)
        self._sources[tag] = item
        self._url_to_tag[url] = tag

        return tag

    @staticmethod
    def _expand_compound_tags(text: str) -> str:
        """Expand [SOURCE_X, SOURCE_Y] into [SOURCE_X] [SOURCE_Y]."""
        def expand(match: re.Match) -> str:
            tags = re.findall(r'SOURCE_\w+', match.group(1))
            return ' '.join(f'[{t}]' for t in tags)
        return re.sub(r'\[(SOURCE_\w+(?:\s*,\s*SOURCE_\w+)+)\]', expand, text)

    @staticmethod
    def _normalize_bare_tags(text: str) -> str:
        """Wrap bare SOURCE_N references in brackets.

        Catches patterns like SOURCE_3, *SOURCE_3*, (SOURCE_3, SOURCE_6) so
        they are handled by the primary substitution pass.
        Does not double-wrap already-bracketed tags.
        """
        # Match SOURCE_\w+ not already surrounded by [ ]
        return re.sub(r'(?<!\[)(SOURCE_\w+)(?!\])', r'[\1]', text)

    def substitute(self, text: str) -> str:
        """Replace all [SOURCE_N] placeholders with markdown links.

        Also does a fallback pass replacing (source_name) patterns for cases
        where the model cited by outlet name instead of tag.
        Also removes any hallucinated [SOURCE_N] tags not in the registry.

        Args:
            text: The text containing [SOURCE_N] placeholders

        Returns:
            Text with placeholders replaced by markdown [title](url) links
        """
        if not text:
            return text

        # Normalize compound tags like [SOURCE_1, SOURCE_3] → [SOURCE_1] [SOURCE_3]
        text = self._expand_compound_tags(text)

        # Normalize bare references like SOURCE_3 or *SOURCE_3* → [SOURCE_3]
        text = self._normalize_bare_tags(text)

        # Primary: replace registered [SOURCE_N] tags with markdown links
        for tag, item in self._sources.items():
            if tag in text:
                self.logger.debug(f"Substituting {tag} with URL: {item.url}")
            text = text.replace(tag, f"[{item.title}]({item.url})")

        # Fallback: replace (source_name) patterns the model wrote instead of tags
        seen: set[str] = set()
        for item in self._sources.values():
            name = item.source_name
            if not name or name in seen:
                continue
            seen.add(name)
            pattern = f"({name})"
            if pattern in text:
                text = text.replace(pattern, f"[{item.title}]({item.url})")

        # Strip any remaining [SOURCE_N] tags not in registry (hallucinated or evicted)
        text = re.sub(r'\[SOURCE_\w+\]', '', text)

        return text

    @property
    def source_map(self) -> dict[str, str]:
        """Returns a copy of the tag->URL mapping."""
        return {tag: item.url for tag, item in self._sources.items()}

    @property
    def count(self) -> int:
        """Returns the number of registered sources."""
        return self._counter


class SourceDataBuilder:
    """Builds formatted source data from tool results with URL registry."""

    # Tools whose output is instructions to the researcher, not source data
    INTERMEDIATE_TOOLS = {
        "list_gov_rss_feeds",
        "list_us_news_rss_feeds",
        "list_world_news_rss_feeds"
    }

    def build(self, messages: List[Any], registry: Optional[SourceRegistry] = None) -> Tuple[str, SourceRegistry]:
        """Extract tool results from messages and build source data with registry.

        Accumulates into an existing registry if provided, so SOURCE_N tags
        remain consistent across multiple research calls within one query.

        Args:
            messages: Message history from agent run
            registry: Existing registry to extend. Creates a fresh one if None.

        Returns:
            Tuple of (formatted source text, registry)
        """
        if registry is None:
            registry = SourceRegistry()

        tool_parts = self._extract_tool_parts(messages)
        sections = self._collect_sections(tool_parts, registry, None)

        if not sections:
            return "", registry

        return self._build_source_text(sections), registry

    def _collect_sections(
        self,
        tool_parts: List[Any],
        registry: SourceRegistry,
        filter_text: Optional[str],
    ) -> List[str]:
        """Build formatted sections from tool parts, optionally filtered."""
        sections = []
        for part in tool_parts:
            section = self._format_section(part.tool_name, part.content, registry, filter_text)
            if section:
                sections.append(section)
        return sections

    def _extract_tool_parts(self, messages: List[Any]) -> List[Any]:
        """Extract non-intermediate tool parts from messages."""
        from pydantic_ai.messages import ModelRequest, ToolReturnPart

        parts = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if (isinstance(part, ToolReturnPart)
                            and part.content is not None
                            and part.tool_name not in self.INTERMEDIATE_TOOLS):
                        parts.append(part)
        return parts

    def _normalize_content(self, content: Any) -> List[Any]:
        """Normalize content to a list (wraps single items)."""
        return content if isinstance(content, list) else [content]

    def _format_sourceable_item(self, item: Any, registry: SourceRegistry) -> Optional[str]:
        """Format a single Sourceable item with source tag.

        Returns:
            Formatted line or None if item is not Sourceable
        """
        if not isinstance(item, Sourceable):
            return None
        if not hasattr(item, 'source_url') or not item.source_url:
            return None

        title = getattr(item, 'title', '')
        summary = (
            getattr(item, 'body', '') or
            getattr(item, 'summary', '') or
            getattr(item, 'text', '')
            )[:200]
        source_name = (
            getattr(item, 'source', '') or
            getattr(item, 'source_name', '') or
            getattr(item, 'author_handle', '')
            )

        tag = registry.register(
            url=item.source_url,
            title=title if title else 'Source',
            source_name=source_name,
        )

        return f"- {tag} {title} [via {source_name}]: {summary}"

    @staticmethod
    def _is_relevant(item: Any, filter_text: str) -> bool:
        """Return True if the item's URL or source name appears in filter_text.

        Primary check: URL match (exact, since Quaesitor copies URLs verbatim).
        Fallback: source-name substring match for URL-less items (e.g. profiles).
        """
        url = (getattr(item, 'source_url', '') or '').strip()
        if url:
            return url in filter_text
        name = (getattr(item, 'source', '') or getattr(item, 'source_name', '') or '').lower()
        return bool(name) and name in filter_text.lower()

    def _format_section(
        self,
        tool_name: str,
        content: Any,
        registry: SourceRegistry,
        filter_text: Optional[str] = None,
    ) -> Optional[str]:
        """Format all results from a single tool call.

        Returns:
            Formatted section or None if no content
        """
        section_lines = [f"[{tool_name}]:"]
        items = self._normalize_content(content)

        for item in items:
            if filter_text is not None and isinstance(item, Sourceable) and not self._is_relevant(item, filter_text):
                continue
            formatted = self._format_sourceable_item(item, registry)
            if formatted:
                section_lines.append(formatted)
            elif item is not None:
                # Fallback for non-Sourceable items
                section_lines.append(f"- {item}")

        # Only return section if there's content beyond the header
        return "\n".join(section_lines) if len(section_lines) > 1 else None

    def _build_source_text(self, sections: List[str]) -> str:
        """Assemble final source data text from sections."""
        return "Source Data (use [SOURCE_N] tags to cite):\n" + "\n\n".join(sections)
