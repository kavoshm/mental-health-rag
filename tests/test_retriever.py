"""
Tests for the retriever module: RetrievalResult data class and
TherapySessionRetriever helper methods.
"""

import pytest

# The retriever module imports chromadb at the top level.
# Skip the entire module if chromadb is not installed.
chromadb = pytest.importorskip("chromadb", reason="chromadb not installed")
langchain = pytest.importorskip("langchain", reason="langchain not installed")

from src.retriever import RetrievalResult, TherapySessionRetriever


# =========================================================================
# RetrievalResult
# =========================================================================

class TestRetrievalResult:
    """Tests for the RetrievalResult data class."""

    def test_similarity_score_from_distance(self):
        r = RetrievalResult(
            chunk_id="chunk_001",
            text="some text",
            metadata={"client_id": "CLT-0001"},
            distance=0.2,
        )
        assert r.similarity_score == pytest.approx(0.8, abs=0.001)

    def test_similarity_score_clamps_to_zero(self):
        """Distance > 1 should not produce negative similarity."""
        r = RetrievalResult(
            chunk_id="chunk_001",
            text="text",
            metadata={},
            distance=1.5,
        )
        assert r.similarity_score == 0.0

    def test_perfect_match_score(self):
        r = RetrievalResult(
            chunk_id="chunk_001",
            text="text",
            metadata={},
            distance=0.0,
        )
        assert r.similarity_score == 1.0

    def test_client_id_property(self):
        r = RetrievalResult(
            chunk_id="c1",
            text="t",
            metadata={"client_id": "CLT-4401"},
            distance=0.5,
        )
        assert r.client_id == "CLT-4401"

    def test_client_id_default(self):
        r = RetrievalResult(chunk_id="c1", text="t", metadata={}, distance=0.5)
        assert r.client_id == "unknown"

    def test_session_date_property(self):
        r = RetrievalResult(
            chunk_id="c1",
            text="t",
            metadata={"session_date": "2025-01-06"},
            distance=0.5,
        )
        assert r.session_date == "2025-01-06"

    def test_source_file_property(self):
        r = RetrievalResult(
            chunk_id="c1",
            text="t",
            metadata={"source_file": "session_001.txt"},
            distance=0.5,
        )
        assert r.source_file == "session_001.txt"

    def test_to_dict(self):
        r = RetrievalResult(
            chunk_id="session_001_chunk_000",
            text="Therapist: Hello",
            metadata={"client_id": "CLT-4401"},
            distance=0.15,
        )
        d = r.to_dict()
        assert d["chunk_id"] == "session_001_chunk_000"
        assert d["text"] == "Therapist: Hello"
        assert d["similarity_score"] == pytest.approx(0.85, abs=0.01)
        assert d["metadata"]["client_id"] == "CLT-4401"


# =========================================================================
# TherapySessionRetriever._build_where_filter
# =========================================================================

class TestBuildWhereFilter:
    """Tests for the where-filter builder (no ChromaDB connection needed)."""

    @pytest.fixture(autouse=True)
    def _setup_retriever(self, monkeypatch):
        """Patch out ChromaDB initialization."""
        monkeypatch.setattr(
            "src.retriever.TherapySessionRetriever.__init__",
            lambda self: None,
        )
        self.retriever = TherapySessionRetriever()

    def test_no_filters_returns_none(self):
        result = self.retriever._build_where_filter()
        assert result is None

    def test_single_client_filter(self):
        result = self.retriever._build_where_filter(client_id="CLT-4401")
        assert result == {"client_id": {"$eq": "CLT-4401"}}

    def test_exclude_client_filter(self):
        result = self.retriever._build_where_filter(exclude_client="CLT-4401")
        assert result == {"client_id": {"$ne": "CLT-4401"}}

    def test_date_range_filter(self):
        result = self.retriever._build_where_filter(
            date_from="2025-01-01",
            date_to="2025-03-31",
        )
        assert "$and" in result
        conditions = result["$and"]
        assert len(conditions) == 2

    def test_combined_filters(self):
        result = self.retriever._build_where_filter(
            client_id="CLT-4401",
            date_from="2025-01-01",
        )
        assert "$and" in result
        conditions = result["$and"]
        assert len(conditions) == 2


# =========================================================================
# TherapySessionRetriever._parse_results
# =========================================================================

class TestParseResults:
    """Tests for raw ChromaDB result parsing."""

    @pytest.fixture(autouse=True)
    def _setup_retriever(self, monkeypatch):
        monkeypatch.setattr(
            "src.retriever.TherapySessionRetriever.__init__",
            lambda self: None,
        )
        self.retriever = TherapySessionRetriever()

    def test_empty_results(self):
        parsed = self.retriever._parse_results({})
        assert parsed == []

    def test_none_results(self):
        parsed = self.retriever._parse_results(None)
        assert parsed == []

    def test_valid_results_parsed(self):
        raw = {
            "ids": [["chunk_a", "chunk_b"]],
            "documents": [["text a", "text b"]],
            "metadatas": [[{"client_id": "C1"}, {"client_id": "C2"}]],
            "distances": [[0.1, 0.3]],
        }
        parsed = self.retriever._parse_results(raw)
        assert len(parsed) == 2
        assert parsed[0].chunk_id == "chunk_a"
        assert parsed[0].similarity_score > parsed[1].similarity_score

    def test_results_sorted_by_distance(self):
        raw = {
            "ids": [["far", "close"]],
            "documents": [["text far", "text close"]],
            "metadatas": [[{}, {}]],
            "distances": [[0.9, 0.1]],
        }
        parsed = self.retriever._parse_results(raw)
        assert parsed[0].chunk_id == "close"
        assert parsed[1].chunk_id == "far"
