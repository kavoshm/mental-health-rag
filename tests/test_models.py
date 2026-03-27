"""
Tests for Pydantic model validation.

Ensures that the data models used across the pipeline enforce correct
types, constraints, and defaults.
"""

import pytest
from datetime import date

from pydantic import ValidationError

from src.models import (
    HealthResponse,
    RiskAssessment,
    RiskLevel,
    SessionListItem,
    SessionSummary,
    SimilarSession,
    TranscriptInput,
)


# =========================================================================
# RiskLevel Enum
# =========================================================================

class TestRiskLevel:
    """Tests for the RiskLevel enum."""

    def test_all_levels_exist(self):
        assert RiskLevel.NONE == "none"
        assert RiskLevel.LOW == "low"
        assert RiskLevel.MODERATE == "moderate"
        assert RiskLevel.HIGH == "high"
        assert RiskLevel.IMMINENT == "imminent"

    def test_level_count(self):
        assert len(RiskLevel) == 5


# =========================================================================
# RiskAssessment Model
# =========================================================================

class TestRiskAssessment:
    """Tests for the RiskAssessment Pydantic model."""

    def test_valid_risk_assessment(self):
        ra = RiskAssessment(
            level=RiskLevel.LOW,
            factors=["passive ideation"],
            protective_factors=["family support"],
            recommended_actions=["monitor"],
        )
        assert ra.level == RiskLevel.LOW
        assert len(ra.factors) == 1

    def test_defaults_to_empty_lists(self):
        ra = RiskAssessment(level=RiskLevel.NONE)
        assert ra.factors == []
        assert ra.protective_factors == []
        assert ra.recommended_actions == []

    def test_invalid_level_raises(self):
        with pytest.raises(ValidationError):
            RiskAssessment(level="critical")

    def test_serialization_roundtrip(self):
        ra = RiskAssessment(
            level=RiskLevel.MODERATE,
            factors=["active ideation"],
            protective_factors=["engaged in treatment"],
            recommended_actions=["increase frequency"],
        )
        data = ra.model_dump()
        restored = RiskAssessment(**data)
        assert restored.level == ra.level
        assert restored.factors == ra.factors


# =========================================================================
# SimilarSession Model
# =========================================================================

class TestSimilarSession:
    """Tests for the SimilarSession model."""

    def test_valid_similar_session(self):
        ss = SimilarSession(
            session_id="session_005",
            client_id="CLT-4405",
            similarity_score=0.78,
            relevant_excerpt="Client reports relapse...",
            relevance_reason="Similar coping patterns",
        )
        assert ss.similarity_score == 0.78

    def test_similarity_score_range_lower(self):
        with pytest.raises(ValidationError):
            SimilarSession(
                session_id="s1",
                client_id="c1",
                similarity_score=-0.1,
                relevant_excerpt="text",
                relevance_reason="reason",
            )

    def test_similarity_score_range_upper(self):
        with pytest.raises(ValidationError):
            SimilarSession(
                session_id="s1",
                client_id="c1",
                similarity_score=1.5,
                relevant_excerpt="text",
                relevance_reason="reason",
            )

    def test_boundary_scores_valid(self):
        for score in [0.0, 0.5, 1.0]:
            ss = SimilarSession(
                session_id="s1",
                client_id="c1",
                similarity_score=score,
                relevant_excerpt="text",
                relevance_reason="reason",
            )
            assert ss.similarity_score == score


# =========================================================================
# SessionSummary Model
# =========================================================================

class TestSessionSummary:
    """Tests for the SessionSummary model."""

    def _make_summary(self, **overrides):
        defaults = {
            "session_id": "session_001",
            "client_id": "CLT-4401",
            "presenting_problem": "Depression following job loss",
            "risk_assessment": RiskAssessment(level=RiskLevel.NONE),
        }
        defaults.update(overrides)
        return SessionSummary(**defaults)

    def test_minimal_valid_summary(self):
        s = self._make_summary()
        assert s.session_id == "session_001"
        assert s.session_date is None
        assert s.mood_indicators == []
        assert s.similar_sessions == []

    def test_full_summary(self):
        s = self._make_summary(
            session_date=date(2025, 1, 6),
            session_number=1,
            clinician="Dr. Navarro, PsyD",
            mood_indicators=["PHQ-9: 19"],
            key_themes=["career disappointment"],
            therapeutic_interventions=["CBT"],
            client_progress="Initial session",
            homework_assigned=["mood log"],
            recommended_followup=["weekly sessions"],
        )
        assert s.session_date == date(2025, 1, 6)
        assert s.session_number == 1
        assert len(s.mood_indicators) == 1

    def test_json_serialization(self):
        s = self._make_summary(session_date=date(2025, 1, 6))
        json_str = s.model_dump_json()
        assert "session_001" in json_str
        assert "2025-01-06" in json_str

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            SessionSummary(session_id="s1")  # missing client_id, presenting_problem, risk


# =========================================================================
# TranscriptInput Model
# =========================================================================

class TestTranscriptInput:
    """Tests for the API input model."""

    def test_valid_input(self):
        ti = TranscriptInput(transcript="A" * 100)
        assert len(ti.transcript) == 100
        assert ti.include_similar is True
        assert ti.client_id is None

    def test_transcript_min_length(self):
        with pytest.raises(ValidationError):
            TranscriptInput(transcript="Too short")

    def test_optional_client_id(self):
        ti = TranscriptInput(transcript="A" * 150, client_id="CLT-0001")
        assert ti.client_id == "CLT-0001"

    def test_include_similar_defaults_true(self):
        ti = TranscriptInput(transcript="A" * 150)
        assert ti.include_similar is True

    def test_include_similar_override(self):
        ti = TranscriptInput(transcript="A" * 150, include_similar=False)
        assert ti.include_similar is False


# =========================================================================
# SessionListItem Model
# =========================================================================

class TestSessionListItem:
    """Tests for the session listing model."""

    def test_valid_item(self):
        item = SessionListItem(
            session_id="session_001",
            client_id="CLT-4401",
            chunk_count=4,
        )
        assert item.session_date is None
        assert item.chunk_count == 4

    def test_with_optional_fields(self):
        item = SessionListItem(
            session_id="session_001",
            client_id="CLT-4401",
            session_date="2025-01-06",
            session_number=1,
            chunk_count=4,
        )
        assert item.session_date == "2025-01-06"


# =========================================================================
# HealthResponse Model
# =========================================================================

class TestHealthResponse:
    """Tests for the health check response model."""

    def test_defaults(self):
        hr = HealthResponse()
        assert hr.status == "healthy"
        assert hr.document_count == 0
        assert hr.collection_name == ""

    def test_populated(self):
        hr = HealthResponse(
            status="healthy",
            collection_name="therapy_sessions",
            document_count=52,
            embedding_model="text-embedding-3-small",
            llm_model="gpt-4o-mini",
        )
        assert hr.document_count == 52
        assert hr.embedding_model == "text-embedding-3-small"
