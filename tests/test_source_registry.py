import pytest
from src.source_registry import (
    SourceRegistry,
    SourceItem,
    SourceDataBuilder,
    _classify_primary,
    _cosine_similarity,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def registry():
    """Create a registry with a few registered sources."""
    reg = SourceRegistry()
    reg.register("https://example.com/1", title="Article One", source_name="Reuters")
    reg.register("https://example.com/2", title="Article Two", source_name="BBC")
    reg.register("https://example.com/3", title="Article Three", source_name="AP News")
    return reg


class TestSubstituteStandardTags:
    def test_uppercase_tags(self, registry):
        text = "See [SOURCE_1] and [SOURCE_2]."
        result = registry.substitute(text)
        assert "[Article One](https://example.com/1)" in result
        assert "[Article Two](https://example.com/2)" in result
        assert "[SOURCE_" not in result

    def test_empty_text(self, registry):
        assert registry.substitute("") == ""

    def test_no_tags(self, registry):
        text = "Plain text with no sources."
        assert registry.substitute(text) == text


class TestMixedCaseTags:
    def test_mixed_case_bracketed(self, registry):
        """[Source_1] should resolve the same as [SOURCE_1]."""
        text = "See [Source_1] for details."
        result = registry.substitute(text)
        assert "[Article One](https://example.com/1)" in result
        assert "Source_1" not in result

    def test_lowercase_bracketed(self, registry):
        """[source_2] should resolve."""
        text = "Per [source_2]."
        result = registry.substitute(text)
        assert "[Article Two](https://example.com/2)" in result

    def test_multiple_mixed_case(self, registry):
        text = "[Source_1] [source_2] [SOURCE_3]"
        result = registry.substitute(text)
        assert "[Article One](https://example.com/1)" in result
        assert "[Article Two](https://example.com/2)" in result
        assert "[Article Three](https://example.com/3)" in result


class TestBareTags:
    def test_bare_uppercase(self, registry):
        """SOURCE_1 without brackets should be wrapped and resolved."""
        text = "According to SOURCE_1, the event occurred."
        result = registry.substitute(text)
        assert "[Article One](https://example.com/1)" in result

    def test_bare_mixed_case(self, registry):
        """Source_2 without brackets should be wrapped and resolved."""
        text = "Source_2 reports that..."
        result = registry.substitute(text)
        assert "[Article Two](https://example.com/2)" in result

    def test_bare_lowercase(self, registry):
        text = "source_3 confirms..."
        result = registry.substitute(text)
        assert "[Article Three](https://example.com/3)" in result


class TestUnclosedBrackets:
    def test_unclosed_uppercase(self, registry):
        """[SOURCE_1 (missing closing bracket) should be fixed."""
        text = "See [SOURCE_1 for details."
        result = registry.substitute(text)
        assert "[Article One](https://example.com/1)" in result
        assert "[SOURCE_" not in result

    def test_unclosed_mixed_case(self, registry):
        """[Source_2 should be fixed and resolved."""
        text = "Per [Source_2 the report says..."
        result = registry.substitute(text)
        assert "[Article Two](https://example.com/2)" in result

    def test_unclosed_lowercase(self, registry):
        text = "[source_3 confirms this."
        result = registry.substitute(text)
        assert "[Article Three](https://example.com/3)" in result

    def test_mix_of_closed_and_unclosed(self, registry):
        text = "[SOURCE_1] and [Source_2 and [source_3"
        result = registry.substitute(text)
        assert "[Article One](https://example.com/1)" in result
        assert "[Article Two](https://example.com/2)" in result
        assert "[Article Three](https://example.com/3)" in result


class TestCompoundTags:
    def test_compound_uppercase(self, registry):
        """[SOURCE_1, SOURCE_2] should split into two resolved links."""
        text = "See [SOURCE_1, SOURCE_2] for context."
        result = registry.substitute(text)
        assert "[Article One](https://example.com/1)" in result
        assert "[Article Two](https://example.com/2)" in result

    def test_compound_mixed_case(self, registry):
        """[Source_1, Source_3] should split and resolve."""
        text = "See [Source_1, Source_3]."
        result = registry.substitute(text)
        assert "[Article One](https://example.com/1)" in result
        assert "[Article Three](https://example.com/3)" in result


class TestSourceNameFallback:
    def test_parenthetical_source_name(self, registry):
        """(Reuters) should be replaced with the corresponding link."""
        text = "The explosion was confirmed (Reuters)."
        result = registry.substitute(text)
        assert "[Article One](https://example.com/1)" in result
        assert "(Reuters)" not in result


class TestHallucinatedTags:
    def test_hallucinated_tag_stripped(self, registry):
        """Tags not in the registry should be removed."""
        text = "See [SOURCE_99] for details."
        result = registry.substitute(text)
        assert "[SOURCE_99]" not in result
        assert "See  for details." == result

    def test_hallucinated_mixed_case_stripped(self, registry):
        """Hallucinated [Source_99] should also be stripped."""
        text = "Per [Source_99]."
        result = registry.substitute(text)
        assert "Source_99" not in result

    def test_hallucinated_lowercase_stripped(self, registry):
        text = "[source_999] says..."
        result = registry.substitute(text)
        assert "source_999" not in result


class TestParentheticalReferences:
    def test_source_number_parenthetical(self, registry):
        """(Source 51) should be stripped."""
        text = "The report (Source 51) confirms this."
        result = registry.substitute(text)
        assert "(Source 51)" not in result

    def test_sources_range_parenthetical(self, registry):
        """(Sources 47, 49, 50) should be stripped."""
        text = "Multiple reports (Sources 47, 49, 50) confirm."
        result = registry.substitute(text)
        assert "(Sources 47, 49, 50)" not in result

    def test_sources_dash_range(self, registry):
        """(Sources 57–64) should be stripped."""
        text = "Coverage (Sources 57–64) was extensive."
        result = registry.substitute(text)
        assert "(Sources 57–64)" not in result


class TestDeduplication:
    def test_same_url_returns_same_tag(self):
        reg = SourceRegistry()
        tag1 = reg.register("https://example.com/same", title="First")
        tag2 = reg.register("https://example.com/same", title="Second")
        assert tag1 == tag2

    def test_different_urls_get_different_tags(self):
        reg = SourceRegistry()
        tag1 = reg.register("https://example.com/a")
        tag2 = reg.register("https://example.com/b")
        assert tag1 != tag2


class TestRealWorldPatterns:
    """Test patterns observed in actual LLM output from training logs."""

    def test_taylor_swift_pattern(self):
        """Reproduce the exact pattern from the training log."""
        reg = SourceRegistry()
        # Register sources with high numbers like the real scenario
        for i in range(1, 67):
            reg.register(f"https://example.com/{i}", title=f"Article {i}")

        text = (
            "According to Source_62, the strikes were condemned. "
            "Source_65 reported that Swift made a statement. "
            "[Source_62] [Source_65] [Source_66] [Source_63] "
            "cited multiple perspectives."
        )
        result = reg.substitute(text)
        assert "Source_62" not in result
        assert "Source_65" not in result
        assert "Source_66" not in result
        assert "Source_63" not in result
        assert "[Article 62]" in result
        assert "[Article 65]" in result

    def test_mixed_mutations_in_one_text(self):
        """Multiple mutation types in a single response."""
        reg = SourceRegistry()
        reg.register("https://a.com", title="A")
        reg.register("https://b.com", title="B")
        reg.register("https://c.com", title="C")

        text = (
            "[SOURCE_1] confirmed. Source_2 also reported. "
            "[source_3 had details. (Source 99) was unverifiable."
        )
        result = reg.substitute(text)
        assert "[A](https://a.com)" in result
        assert "[B](https://b.com)" in result
        assert "[C](https://c.com)" in result
        assert "(Source 99)" not in result
        assert "SOURCE_" not in result.upper().replace("(HTTPS", "")


class TestClassifyPrimary:
    def test_gov_domain(self):
        assert _classify_primary("https://www.whitehouse.gov/path") is True

    def test_mil_domain(self):
        assert _classify_primary("https://army.mil/news") is True

    def test_edu_domain(self):
        assert _classify_primary("https://mit.edu/research") is True

    def test_plain_dotgov_tld(self):
        assert _classify_primary("https://epa.gov") is True

    def test_non_primary_domain(self):
        assert _classify_primary("https://example.com") is False

    def test_fakegov_domain(self):
        assert _classify_primary("https://fakegov.net") is False

    def test_empty_string(self):
        assert _classify_primary("") is False

    def test_courtlistener(self):
        assert _classify_primary("https://www.courtlistener.com/opinion/123/") is True

    def test_fec(self):
        assert _classify_primary("https://www.fec.gov/data/candidate/P00000001/") is True


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert _cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_no_error(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_both_zero_vectors(self):
        assert _cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_opposite_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        result = _cosine_similarity([1.0, 1.0], [1.0, 1.0])
        assert result == pytest.approx(1.0)


class TestSourceItemConfidenceLevel:
    def test_high_corroboration(self):
        item = SourceItem(url="https://a.com", corroboration_count=3)
        assert item.confidence_level == "HIGH"

    def test_more_than_3_corroborations(self):
        item = SourceItem(url="https://a.com", corroboration_count=5)
        assert item.confidence_level == "HIGH"

    def test_medium_corroboration(self):
        item = SourceItem(url="https://a.com", corroboration_count=2)
        assert item.confidence_level == "MEDIUM"

    def test_low_no_corroboration(self):
        item = SourceItem(url="https://a.com", corroboration_count=0, is_primary=False)
        assert item.confidence_level == "LOW"

    def test_single_corroboration_non_primary(self):
        item = SourceItem(url="https://a.com", corroboration_count=1, is_primary=False)
        assert item.confidence_level == "LOW"

    def test_primary_single_corroboration_is_medium(self):
        # is_primary=True with count=1 → "MEDIUM" (hits the `elif is_primary` branch)
        item = SourceItem(url="https://epa.gov", corroboration_count=1, is_primary=True)
        assert item.confidence_level == "MEDIUM"

    def test_primary_with_two_corroborations_is_high(self):
        # is_primary=True + count>=2 hits the first branch
        item = SourceItem(url="https://epa.gov", corroboration_count=2, is_primary=True)
        assert item.confidence_level == "HIGH"


class TestRegisterSequencing:
    def test_first_url_is_source_1(self):
        reg = SourceRegistry()
        tag = reg.register("https://first.com")
        assert tag == "[SOURCE_1]"

    def test_second_url_is_source_2(self):
        reg = SourceRegistry()
        reg.register("https://first.com")
        tag = reg.register("https://second.com")
        assert tag == "[SOURCE_2]"

    def test_eviction_at_max_size(self):
        reg = SourceRegistry()
        for i in range(SourceRegistry.MAX_SIZE):
            reg.register(f"https://example.com/{i}")
        # Registry is full; adding one more evicts the oldest
        first_tag = "[SOURCE_1]"
        assert first_tag in reg._sources  # still present before eviction
        reg.register("https://example.com/overflow")
        assert first_tag not in reg._sources  # evicted

    def test_blank_url_returns_empty(self):
        reg = SourceRegistry()
        assert reg.register("") == ""
        assert reg.register("   ") == ""


class TestLookupByKey:
    def test_lookup_source_n_format(self):
        reg = SourceRegistry()
        reg.register("https://a.com", title="A")
        item = reg.lookup_by_key("SOURCE_1")
        assert item is not None
        assert item.url == "https://a.com"

    def test_lookup_bracketed_format(self):
        reg = SourceRegistry()
        reg.register("https://a.com", title="A")
        item = reg.lookup_by_key("[SOURCE_1]")
        assert item is not None

    def test_lookup_numeric_format(self):
        reg = SourceRegistry()
        reg.register("https://a.com")
        item = reg.lookup_by_key("1")
        assert item is not None

    def test_lookup_lowercase(self):
        reg = SourceRegistry()
        reg.register("https://a.com")
        item = reg.lookup_by_key("source_1")
        assert item is not None

    def test_lookup_unknown_key_returns_none(self):
        reg = SourceRegistry()
        reg.register("https://a.com")
        item = reg.lookup_by_key("SOURCE_99")
        assert item is None


class TestExpandCompoundTags:
    def test_two_item_compound(self):
        result = SourceRegistry._expand_compound_tags("[SOURCE_1, SOURCE_2]")
        assert result == "[SOURCE_1] [SOURCE_2]"

    def test_three_item_compound(self):
        result = SourceRegistry._expand_compound_tags("[SOURCE_1, SOURCE_2, SOURCE_3]")
        assert result == "[SOURCE_1] [SOURCE_2] [SOURCE_3]"

    def test_single_tag_unchanged(self):
        text = "[SOURCE_1]"
        assert SourceRegistry._expand_compound_tags(text) == text

    def test_plain_text_unchanged(self):
        text = "no tags here"
        assert SourceRegistry._expand_compound_tags(text) == text


class TestNormalizeBareTags:
    def test_bare_uppercase(self):
        result = SourceRegistry._normalize_bare_tags("SOURCE_1")
        assert result == "[SOURCE_1]"

    def test_already_bracketed_unchanged(self):
        text = "[SOURCE_1]"
        assert SourceRegistry._normalize_bare_tags(text) == text

    def test_bare_lowercase(self):
        result = SourceRegistry._normalize_bare_tags("source_1")
        assert "[SOURCE_1]" in result


class TestFormatForUser:
    def test_empty_registry_fallback(self):
        reg = SourceRegistry()
        result = reg.format_for_user()
        assert "No sources" in result

    def test_populated_registry_contains_source(self):
        reg = SourceRegistry()
        reg.register("https://a.gov", title="Gov Report")
        result = reg.format_for_user()
        assert "Gov Report" in result
        assert "https://a.gov" in result

    def test_confidence_levels_present(self):
        reg = SourceRegistry()
        reg.register("https://a.gov", title="Primary")
        reg.register("https://b.com", title="Secondary")
        result = reg.format_for_user()
        assert "HIGH" in result or "MEDIUM" in result or "LOW" in result


class TestSourceDataBuilderIsRelevant:
    def test_url_in_filter_text(self):
        class FakeItem:
            source_url = "https://example.com/article"
        assert SourceDataBuilder._is_relevant(FakeItem(), "https://example.com/article") is True

    def test_url_not_in_filter_text(self):
        class FakeItem:
            source_url = "https://other.com/article"
        assert SourceDataBuilder._is_relevant(FakeItem(), "https://example.com") is False

    def test_source_name_in_filter_text(self):
        class FakeItem:
            source_url = ""
            source = "reuters"
        assert SourceDataBuilder._is_relevant(FakeItem(), "reuters reported today") is True

    def test_source_name_not_in_filter_text(self):
        class FakeItem:
            source_url = ""
            source = "bbc"
        assert SourceDataBuilder._is_relevant(FakeItem(), "reuters reported today") is False

    def test_no_url_no_name_returns_false(self):
        class FakeItem:
            source_url = ""
            source = ""
        assert SourceDataBuilder._is_relevant(FakeItem(), "anything") is False
