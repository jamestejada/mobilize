import re
import asyncio
import logging
import httpx
from dataclasses import dataclass, field
from typing import (
        Protocol,
        runtime_checkable,
        List,
        Tuple,
        Set,
        Any,
        Optional
    )


PRIMARY_DOMAINS = {
    '.gov', '.mil', '.edu',
    'courtlistener.com', 'congress.gov', 'fec.gov',
    'supremecourt.gov', 'uscourts.gov',
}


def _classify_primary(url: str) -> bool:
    """Heuristic: is this URL from an official/primary source?"""
    url_lower = url.lower()
    return any(domain in url_lower for domain in PRIMARY_DOMAINS)


def _cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


async def get_query_embedding(text: str) -> list:
    """Fetch an embedding vector from Ollama mxbai-embed-large."""
    from .settings import OllamaEndpoints
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            str(OllamaEndpoints.EMBEDDINGS),
            json={"model": "mxbai-embed-large", "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


@dataclass
class SourceItem:
    """A registered source with URL and title."""
    url: str
    title: str = "Source"
    source_name: str = ""
    description: str = ""
    corroboration_count: int = 1
    is_primary: bool = False
    embed_text: str = ""
    embedding: list = field(default=None, repr=False)  # list[float] | None

    @property
    def confidence_level(self) -> str:
        if self.corroboration_count >= 3 or (self.is_primary and self.corroboration_count >= 2):
            return "HIGH"
        elif self.corroboration_count >= 2 or self.is_primary:
            return "MEDIUM"
        return "LOW"


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

    def register(self, url: str, title: str = "", source_name: str = "", description: str = "") -> str:
        """Register a URL and return its placeholder tag.

        Args:
            url: The URL to register
            title: Optional title for markdown link text
            source_name: Optional outlet or author name
            description: Optional short snippet for LLM relevance judgement

        Returns:
            The placeholder tag (e.g., "[SOURCE_1]")
        """
        if not url or not url.strip():
            return ""

        url = url.strip()

        # Deduplicate: return existing tag if URL already registered
        if url in self._url_to_tag:
            existing_tag = self._url_to_tag[url]
            self._sources[existing_tag].corroboration_count += 1
            return existing_tag

        # Evict oldest entry if at capacity
        if len(self._sources) >= self.MAX_SIZE:
            oldest_tag, oldest_item = next(iter(self._sources.items()))
            del self._sources[oldest_tag]
            del self._url_to_tag[oldest_item.url]
            self.logger.debug(f"Registry full ({self.MAX_SIZE}): evicted {oldest_tag}")

        tag = f"[SOURCE_{self.counter}]"

        embed_text = description[:200] if description else (title or url)

        item = SourceItem(url=url, title=title if title else "Source",
                          source_name=source_name, description=description,
                          is_primary=_classify_primary(url),
                          embed_text=embed_text)
        self._sources[tag] = item
        self._url_to_tag[url] = tag

        return tag

    @staticmethod
    def _expand_compound_tags(text: str) -> str:
        """Expand [SOURCE_X, SOURCE_Y] into [SOURCE_X] [SOURCE_Y]."""
        def expand(match: re.Match) -> str:
            tags = re.findall(r'SOURCE_\w+', match.group(1), re.IGNORECASE)
            return ' '.join(f'[{t.upper()}]' for t in tags)
        return re.sub(r'\[(SOURCE_\w+(?:\s*,\s*SOURCE_\w+)+)\]', expand, text, flags=re.IGNORECASE)

    @staticmethod
    def _normalize_bare_tags(text: str) -> str:
        """Wrap bare SOURCE_N references in brackets.

        Catches patterns like SOURCE_3, *SOURCE_3*, (SOURCE_3, SOURCE_6) so
        they are handled by the primary substitution pass.
        Does not double-wrap already-bracketed tags.
        """
        # Match SOURCE_\w+ not already surrounded by [ ]
        def _upper_bracket(m: re.Match) -> str:
            return f'[{m.group(1).upper()}]'
        return re.sub(r'(?<!\[)(SOURCE_\w+)(?!\])', _upper_bracket, text, flags=re.IGNORECASE)

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

        # Fix unclosed brackets: [SOURCE_3 → [SOURCE_3]
        text = re.sub(r'\[source_(\w+)\b(?!\])', lambda m: f'[SOURCE_{m.group(1).upper()}]', text, flags=re.IGNORECASE)

        # Normalize bare references like SOURCE_3 or *SOURCE_3* → [SOURCE_3]
        text = self._normalize_bare_tags(text)

        # Normalize case: [Source_3], [source_3] → [SOURCE_3]
        text = re.sub(r'\[source_(\w+)\]', lambda m: f'[SOURCE_{m.group(1)}]', text, flags=re.IGNORECASE)

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
        text = re.sub(r'\[SOURCE_\w+\]', '', text, flags=re.IGNORECASE)

        # Strip parenthetical source references the model wrote instead of tags
        # e.g. (Source 51), (Sources 47, 49, 50), (Sources 57–64)
        text = re.sub(r'\s*\(Sources?\s+[\d,\s–\-]+\)', '', text)

        return text

    def format_for_user(self) -> str:
        """Format sources for Telegram user display."""
        if not self._sources:
            return "No sources collected yet."
        lines = [f"**{len(self._sources)} sources collected:**\n"]
        for tag, item in self._sources.items():
            lines.append(f"- [{item.title}]({item.url}) [{item.confidence_level}]")
        return "\n".join(lines)

    @property
    def source_map(self) -> dict[str, str]:
        """Returns a copy of the tag->URL mapping."""
        return {tag: item.url for tag, item in self._sources.items()}

    @property
    def count(self) -> int:
        """Returns the number of registered sources."""
        return self._counter

    FORMAT_LIMIT = 25
    FORMAT_DESC_LEN = 80

    def lookup_by_key(self, source_key: str) -> 'Optional[SourceItem]':
        """Look up a registered source by key in any common format.

        Accepts: "SOURCE_3", "[SOURCE_3]", "3", "source_3" (case-insensitive).
        """
        key = source_key.strip().strip("[]").upper()
        if not key.startswith("SOURCE_"):
            key = f"SOURCE_{key}"
        return self._sources.get(f"[{key}]")

    @staticmethod
    def register_sourceable(registry: 'SourceRegistry', item: Any) -> str:
        """Register a Sourceable item into registry and return its [SOURCE_N] tag.

        Extracts url, title, source_name, and description from common attribute
        names automatically. Returns empty string if item is not Sourceable or
        has no source_url.
        """
        if not isinstance(item, Sourceable) or not item.source_url:
            return ""
        title = getattr(item, 'title', '') or ''
        source_name = (
            getattr(item, 'source', '') or
            getattr(item, 'source_name', '') or
            getattr(item, 'author_handle', '') or ''
        )
        description = (
            getattr(item, 'body', '') or
            getattr(item, 'summary', '') or
            getattr(item, 'text', '') or ''
        )[:200]
        return registry.register(
            url=item.source_url,
            title=title,
            source_name=source_name,
            description=description,
        )

    @staticmethod
    def register_all(registry: 'Optional[SourceRegistry]', items: List[Any]) -> None:
        """Register a list of items and set each item's tag field. None-safe."""
        if registry is None:
            return
        for item in items:
            tag = SourceRegistry.register_sourceable(registry, item)
            if tag and hasattr(item, 'tag'):
                item.tag = tag

    @staticmethod
    def register_one(registry: 'Optional[SourceRegistry]', item: Any) -> None:
        """Register a single item and set its tag field. None-safe."""
        if registry is None or item is None:
            return
        tag = SourceRegistry.register_sourceable(registry, item)
        if tag and hasattr(item, 'tag'):
            item.tag = tag

    async def embed_sources(self) -> None:
        """Compute and store embeddings for all sources not yet embedded."""
        pending = [item for item in self._sources.values() if item.embedding is None]
        if not pending:
            return

        async def _embed_one(item: SourceItem) -> None:
            try:
                item.embedding = await get_query_embedding(item.embed_text)
            except Exception as e:
                self.logger.warning(f"Embedding failed for {item.url}: {e}")

        await asyncio.gather(*[_embed_one(item) for item in pending])

    def format_for_agent_semantic(self, query_embedding: list | None = None) -> str:
        """Return top FORMAT_LIMIT sources ranked by semantic similarity to query_embedding.

        If query_embedding is None or no embeddings have been computed, falls back
        to recency order. Sources with embeddings are always ranked before unembedded ones.
        """
        if not self._sources:
            return "No sources have been collected yet."

        items = list(self._sources.items())

        if query_embedding is not None:
            scored, unscored = [], []
            for tag, item in items:
                if item.embedding is not None:
                    score = _cosine_similarity(query_embedding, item.embedding)
                    scored.append((score, tag, item))
                else:
                    unscored.append((tag, item))
            scored.sort(key=lambda x: x[0], reverse=True)
            combined = [(tag, item) for _, tag, item in scored] + unscored
        else:
            combined = items

        shown = combined[:self.FORMAT_LIMIT]
        total = len(items)
        lines = ["Sources collected in this conversation:"]
        if total > self.FORMAT_LIMIT:
            lines.append(f"  ({total - self.FORMAT_LIMIT} lower-relevance sources not shown)")
        for tag, item in shown:
            desc = f" — {item.description[:self.FORMAT_DESC_LEN]}" if item.description else ""
            lines.append(f"- {tag} [{item.confidence_level}]: {item.title} ({item.url}){desc}")
        return "\n".join(lines)


class SourceDataBuilder:
    """Builds formatted source data from tool results with URL registry."""

    # Tools whose output is instructions to the researcher, not source data
    INTERMEDIATE_TOOLS = {
        "list_gov_rss_feeds",
        "list_world_news_rss_feeds",
        "search_wikipedia",   # context only — not a citable reference
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
        tag = SourceRegistry.register_sourceable(registry, item)
        if not tag:
            return None
        title = getattr(item, 'title', '') or ''
        summary = (
            getattr(item, 'body', '') or
            getattr(item, 'summary', '') or
            getattr(item, 'text', '') or ''
            )[:200]
        source_name = (
            getattr(item, 'source', '') or
            getattr(item, 'source_name', '') or
            getattr(item, 'author_handle', '') or ''
            )
        source_item = registry.lookup_by_key(tag)
        confidence = f" [{source_item.confidence_level}]" if source_item else ""
        return f"- {tag} {title} [via {source_name}]{confidence}: {summary}"

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
