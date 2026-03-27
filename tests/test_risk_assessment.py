"""
Tests for risk keyword detection and risk level classification.

Validates the keyword-based risk assessment in the summarizer against
known transcript patterns covering each risk tier.
"""

import pytest

# The summarizer module transitively imports langchain and chromadb.
# Skip the entire module if those dependencies are unavailable.
langchain = pytest.importorskip("langchain", reason="langchain not installed")
chromadb = pytest.importorskip("chromadb", reason="chromadb not installed")

from src.config import RISK_KEYWORDS, RISK_LEVELS
from src.models import RiskLevel
from src.summarizer import SessionSummarizer


# =========================================================================
# Risk Keywords Configuration
# =========================================================================

class TestRiskKeywords:
    """Tests for the risk keyword list and level definitions."""

    def test_risk_keywords_is_nonempty(self):
        assert len(RISK_KEYWORDS) > 0

    def test_risk_keywords_are_lowercase(self):
        for kw in RISK_KEYWORDS:
            assert kw == kw.lower(), f"Keyword '{kw}' should be lowercase"

    def test_risk_levels_cover_all_enum_values(self):
        for level in RiskLevel:
            assert level.value in RISK_LEVELS

    def test_critical_keywords_present(self):
        """Ensure the most safety-critical keywords are in the list."""
        critical = ["suicidal", "suicide", "self-harm", "kill myself", "overdose"]
        for kw in critical:
            assert kw in RISK_KEYWORDS, f"Critical keyword '{kw}' missing"


# =========================================================================
# Risk Assessment Logic
# =========================================================================

class TestRiskAssessment:
    """Tests for _assess_risk method on the SessionSummarizer.

    Uses the summarizer's keyword-based risk assessment directly,
    without requiring an LLM or ChromaDB connection.
    """

    @pytest.fixture(autouse=True)
    def _setup_summarizer(self, monkeypatch):
        """Patch out ChromaDB and LLM initialization for unit testing."""
        monkeypatch.setattr(
            "src.summarizer.TherapySessionRetriever.__init__",
            lambda self: None,
        )
        monkeypatch.setattr(
            "src.summarizer.SessionSummarizer._init_llm",
            lambda self: None,
        )
        self.summarizer = SessionSummarizer()

    def test_no_risk_when_clean(self, risk_transcript_none):
        result = self.summarizer._assess_risk(risk_transcript_none)
        assert result.level == RiskLevel.NONE

    def test_low_risk_passive_ideation(self, risk_transcript_passive):
        result = self.summarizer._assess_risk(risk_transcript_passive)
        assert result.level == RiskLevel.LOW

    def test_moderate_risk_active_ideation(self, risk_transcript_active):
        result = self.summarizer._assess_risk(risk_transcript_active)
        assert result.level == RiskLevel.MODERATE

    def test_denied_risk_stays_low(self, risk_transcript_denied):
        """When risk terms appear alongside explicit denials, level should be low."""
        result = self.summarizer._assess_risk(risk_transcript_denied)
        assert result.level in (RiskLevel.NONE, RiskLevel.LOW)

    def test_protective_factors_detected(self, risk_transcript_passive):
        result = self.summarizer._assess_risk(risk_transcript_passive)
        assert len(result.protective_factors) > 0

    def test_risk_factors_detected(self, risk_transcript_passive):
        result = self.summarizer._assess_risk(risk_transcript_passive)
        assert len(result.factors) > 0

    def test_recommended_actions_nonempty(self, risk_transcript_passive):
        result = self.summarizer._assess_risk(risk_transcript_passive)
        assert len(result.recommended_actions) > 0

    def test_none_risk_has_routine_monitoring(self, risk_transcript_none):
        result = self.summarizer._assess_risk(risk_transcript_none)
        assert any("routine" in a.lower() or "monitoring" in a.lower()
                    for a in result.recommended_actions)

    def test_moderate_risk_recommends_increased_frequency(self, risk_transcript_active):
        result = self.summarizer._assess_risk(risk_transcript_active)
        assert any("frequency" in a.lower() or "safety" in a.lower()
                    for a in result.recommended_actions)

    def test_full_transcript_risk_assessment(self, sample_full_transcript):
        """Session 001-style transcript should be flagged as low risk."""
        result = self.summarizer._assess_risk(sample_full_transcript)
        assert result.level in (RiskLevel.LOW, RiskLevel.MODERATE)
        assert len(result.factors) > 0
        assert len(result.protective_factors) > 0

    def test_keyword_matching_is_case_insensitive(self):
        text = "The client expressed SUICIDAL thoughts but DENIES SUICIDAL ideation."
        result = self.summarizer._assess_risk(text)
        # Should detect both the keyword and the denial
        assert len(result.factors) > 0
