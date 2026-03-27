"""
FastAPI Application for the Mental Health Session Summary RAG Pipeline.

Endpoints:
- POST /summarize  — Accept a session transcript, return a structured summary
- GET  /sessions   — List all ingested sessions
- GET  /health     — Health check with system status

Includes proper error handling, input validation via Pydantic models,
and CORS configuration for development.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
)
from src.models import (
    HealthResponse,
    SessionListItem,
    SessionSummary,
    TranscriptInput,
)
from src.logging_config import get_logger
from src.retriever import TherapySessionRetriever
from src.summarizer import SessionSummarizer

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Application Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Mental Health Session Summary RAG",
    description=(
        "A RAG pipeline that generates structured summaries of therapy "
        "session transcripts, including risk assessment and retrieval of "
        "similar past sessions."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy-initialized singletons
# ---------------------------------------------------------------------------

_summarizer: SessionSummarizer | None = None
_retriever: TherapySessionRetriever | None = None


def get_summarizer() -> SessionSummarizer:
    """Get or create the summarizer singleton."""
    global _summarizer
    if _summarizer is None:
        _summarizer = SessionSummarizer()
    return _summarizer


def get_retriever() -> TherapySessionRetriever:
    """Get or create the retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = TherapySessionRetriever()
    return _retriever


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/summarize",
    response_model=SessionSummary,
    summary="Summarize a therapy session",
    description=(
        "Accepts a therapy session transcript and returns a structured "
        "summary including presenting problem, interventions, risk "
        "assessment, and similar past sessions."
    ),
)
async def summarize_session(input_data: TranscriptInput) -> SessionSummary:
    """Generate a structured summary from a therapy session transcript.

    Args:
        input_data: TranscriptInput with the transcript text and options.

    Returns:
        A complete SessionSummary.

    Raises:
        HTTPException: If summarization fails.
    """
    logger.info(
        "API /summarize request",
        extra={
            "client_id": input_data.client_id,
            "include_similar": input_data.include_similar,
            "transcript_length": len(input_data.transcript),
        },
    )
    try:
        summarizer = get_summarizer()
        summary = summarizer.summarize_session(
            transcript=input_data.transcript,
            client_id=input_data.client_id,
            include_similar=input_data.include_similar,
        )
        logger.info(
            "API /summarize response",
            extra={
                "session_id": summary.session_id,
                "risk_level": summary.risk_assessment.level.value,
            },
        )
        return summary
    except Exception as e:
        logger.error("API /summarize failed", extra={"error": str(e)}, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {str(e)}",
        )


@app.get(
    "/sessions",
    response_model=list[SessionListItem],
    summary="List ingested sessions",
    description="Returns a list of all therapy sessions currently in the vector store.",
)
async def list_sessions() -> list[SessionListItem]:
    """List all sessions ingested into the vector store.

    Returns:
        List of SessionListItem objects with brief session info.

    Raises:
        HTTPException: If retrieval fails.
    """
    logger.info("API /sessions request")
    try:
        retriever = get_retriever()
        sessions = retriever.list_sessions()

        logger.info("API /sessions response", extra={"session_count": len(sessions)})
        return [
            SessionListItem(
                session_id=s["session_id"],
                client_id=s["client_id"],
                session_date=s.get("session_date"),
                session_number=s.get("session_number"),
                chunk_count=s["chunk_count"],
            )
            for s in sessions
        ]
    except Exception as e:
        logger.error("API /sessions failed", extra={"error": str(e)}, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list sessions: {str(e)}",
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns system health status including collection info and model config.",
)
async def health_check() -> HealthResponse:
    """Return health status and system configuration.

    Returns:
        HealthResponse with status and configuration details.
    """
    try:
        retriever = get_retriever()
        doc_count = retriever.collection_count
        status = "healthy"
    except Exception as e:
        logger.warning("Health check degraded", extra={"error": str(e)})
        doc_count = 0
        status = "degraded"

    logger.info("API /health response", extra={"status": status, "document_count": doc_count})
    return HealthResponse(
        status=status,
        collection_name=COLLECTION_NAME,
        document_count=doc_count,
        embedding_model=EMBEDDING_MODEL,
        llm_model=LLM_MODEL,
    )


# ---------------------------------------------------------------------------
# Development server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
