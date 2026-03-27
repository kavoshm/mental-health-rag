"""
Pydantic Models for Session Summaries and Risk Assessment.

Defines the structured output schemas for the RAG pipeline. These models
enforce type safety, validation, and serialization for both LLM outputs
and API responses.
"""

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Risk level classification for clinical safety assessment."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    IMMINENT = "imminent"


class RiskAssessment(BaseModel):
    """Structured risk assessment extracted from a therapy session.

    This model captures risk indicators, protective factors, and
    recommended clinical actions based on session content.
    """

    level: RiskLevel = Field(
        description="Overall risk level based on session content"
    )
    factors: list[str] = Field(
        default_factory=list,
        description="Identified risk factors (e.g., 'passive death wishes', "
                    "'access to firearms', 'substance use increase')",
    )
    protective_factors: list[str] = Field(
        default_factory=list,
        description="Identified protective factors (e.g., 'supportive family', "
                    "'engaged in treatment', 'denies intent')",
    )
    recommended_actions: list[str] = Field(
        default_factory=list,
        description="Recommended clinical actions based on risk level",
    )


class SimilarSession(BaseModel):
    """A similar past session found via retrieval."""

    session_id: str = Field(description="Source session identifier")
    client_id: str = Field(description="Client identifier from the session")
    similarity_score: float = Field(
        description="Cosine similarity score (0-1)",
        ge=0.0,
        le=1.0,
    )
    relevant_excerpt: str = Field(
        description="The most relevant excerpt from the similar session"
    )
    relevance_reason: str = Field(
        description="Why this session is relevant to the current one"
    )


class SessionSummary(BaseModel):
    """Comprehensive structured summary of a therapy session.

    This is the primary output of the RAG pipeline. It combines LLM
    summarization of the current session with context from similar
    past sessions to produce a clinically useful summary.
    """

    # Session identification
    session_id: str = Field(description="Unique session identifier")
    client_id: str = Field(description="Anonymized client identifier")
    session_date: Optional[date] = Field(
        default=None,
        description="Date of the session",
    )
    session_number: Optional[int] = Field(
        default=None,
        description="Sequential session number for this client",
    )
    clinician: Optional[str] = Field(
        default=None,
        description="Clinician name and credentials",
    )

    # Clinical content
    presenting_problem: str = Field(
        description="Primary issue or concern addressed in this session"
    )
    mood_indicators: list[str] = Field(
        default_factory=list,
        description="Observed and reported mood indicators "
                    "(e.g., 'flat affect', 'PHQ-9: 18', 'tearful')",
    )
    key_themes: list[str] = Field(
        default_factory=list,
        description="Major therapeutic themes discussed in the session",
    )
    therapeutic_interventions: list[str] = Field(
        default_factory=list,
        description="Therapeutic techniques and interventions used "
                    "(e.g., 'CBT thought challenging', 'imaginal exposure', "
                    "'behavioral activation')",
    )
    client_progress: str = Field(
        default="",
        description="Summary of client's progress or regression since last session",
    )
    homework_assigned: list[str] = Field(
        default_factory=list,
        description="Between-session assignments given to the client",
    )

    # Risk and safety
    risk_assessment: RiskAssessment = Field(
        description="Structured risk assessment from the session"
    )

    # Follow-up
    recommended_followup: list[str] = Field(
        default_factory=list,
        description="Recommended follow-up actions, referrals, or focus areas",
    )

    # RAG-specific: similar sessions from the vector store
    similar_sessions: list[SimilarSession] = Field(
        default_factory=list,
        description="Similar past sessions retrieved from the knowledge base",
    )


class TranscriptInput(BaseModel):
    """Input model for the /summarize API endpoint."""

    transcript: str = Field(
        description="Full therapy session transcript text",
        min_length=100,
    )
    client_id: Optional[str] = Field(
        default=None,
        description="Optional client ID to filter similar session search",
    )
    include_similar: bool = Field(
        default=True,
        description="Whether to retrieve and include similar past sessions",
    )


class SessionListItem(BaseModel):
    """Brief session info for the /sessions listing endpoint."""

    session_id: str
    client_id: str
    session_date: Optional[str] = None
    session_number: Optional[int] = None
    chunk_count: int = Field(
        description="Number of chunks stored for this session"
    )


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = "healthy"
    collection_name: str = ""
    document_count: int = 0
    embedding_model: str = ""
    llm_model: str = ""
