"""
Tests for the FastAPI endpoints.

Uses FastAPI's TestClient (backed by httpx) to validate request/response
contracts without starting a real server or connecting to ChromaDB.
"""

import pytest
from unittest.mock import MagicMock

# The API module transitively imports chromadb and langchain.
# Skip if those dependencies are unavailable.
chromadb = pytest.importorskip("chromadb", reason="chromadb not installed")
langchain = pytest.importorskip("langchain", reason="langchain not installed")

from fastapi.testclient import TestClient

from src.models import (
    HealthResponse,
    RiskAssessment,
    RiskLevel,
    SessionSummary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_summarizer():
    """A mock SessionSummarizer that returns a predictable summary."""
    summarizer = MagicMock()
    summarizer.summarize_session.return_value = SessionSummary(
        session_id="test_session",
        client_id="CLT-TEST",
        presenting_problem="Test presenting problem",
        risk_assessment=RiskAssessment(level=RiskLevel.NONE),
        mood_indicators=["stable mood"],
        key_themes=["test theme"],
        therapeutic_interventions=["supportive therapy"],
    )
    return summarizer


@pytest.fixture
def mock_retriever():
    """A mock TherapySessionRetriever."""
    retriever = MagicMock()
    retriever.collection_count = 10
    retriever.list_sessions.return_value = [
        {
            "session_id": "session_001",
            "client_id": "CLT-4401",
            "session_date": "2025-01-06",
            "session_number": 1,
            "chunk_count": 4,
        }
    ]
    return retriever


@pytest.fixture
def client(mock_summarizer, mock_retriever):
    """TestClient with mocked dependencies."""
    import src.api as api_module

    # Reset singletons
    api_module._summarizer = mock_summarizer
    api_module._retriever = mock_retriever

    with TestClient(api_module.app) as c:
        yield c

    # Clean up singletons
    api_module._summarizer = None
    api_module._retriever = None


# =========================================================================
# Health Endpoint
# =========================================================================

class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_contains_required_fields(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "collection_name" in data
        assert "document_count" in data
        assert "embedding_model" in data
        assert "llm_model" in data

    def test_health_reports_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"


# =========================================================================
# Sessions Endpoint
# =========================================================================

class TestSessionsEndpoint:
    """Tests for GET /sessions."""

    def test_sessions_returns_200(self, client):
        response = client.get("/sessions")
        assert response.status_code == 200

    def test_sessions_returns_list(self, client):
        data = client.get("/sessions").json()
        assert isinstance(data, list)
        assert len(data) == 1

    def test_session_item_fields(self, client):
        data = client.get("/sessions").json()
        item = data[0]
        assert item["session_id"] == "session_001"
        assert item["client_id"] == "CLT-4401"
        assert item["chunk_count"] == 4


# =========================================================================
# Summarize Endpoint
# =========================================================================

class TestSummarizeEndpoint:
    """Tests for POST /summarize."""

    def test_summarize_returns_200(self, client):
        response = client.post("/summarize", json={
            "transcript": "A" * 150,
            "include_similar": False,
        })
        assert response.status_code == 200

    def test_summarize_returns_structured_summary(self, client):
        response = client.post("/summarize", json={
            "transcript": "A" * 150,
        })
        data = response.json()
        assert "session_id" in data
        assert "risk_assessment" in data
        assert "presenting_problem" in data

    def test_summarize_rejects_short_transcript(self, client):
        response = client.post("/summarize", json={
            "transcript": "Too short",
        })
        assert response.status_code == 422  # Validation error

    def test_summarize_with_client_id(self, client, mock_summarizer):
        client.post("/summarize", json={
            "transcript": "A" * 150,
            "client_id": "CLT-9999",
        })
        call_kwargs = mock_summarizer.summarize_session.call_args
        assert call_kwargs.kwargs.get("client_id") == "CLT-9999" or \
               (call_kwargs.args and len(call_kwargs.args) > 1)

    def test_summarize_without_similar(self, client, mock_summarizer):
        client.post("/summarize", json={
            "transcript": "A" * 150,
            "include_similar": False,
        })
        call_kwargs = mock_summarizer.summarize_session.call_args
        assert call_kwargs.kwargs.get("include_similar") is False or \
               (len(call_kwargs.args) > 2 and call_kwargs.args[2] is False)


# =========================================================================
# Error Handling
# =========================================================================

class TestErrorHandling:
    """Tests for API error responses."""

    def test_missing_transcript_field(self, client):
        response = client.post("/summarize", json={})
        assert response.status_code == 422

    def test_invalid_json_body(self, client):
        response = client.post(
            "/summarize",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_unknown_endpoint_returns_404(self, client):
        response = client.get("/unknown")
        assert response.status_code == 404
