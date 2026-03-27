"""
LLM Summarization Module for Therapy Session Transcripts.

Takes a new session transcript, retrieves similar past sessions for context,
and generates a structured SessionSummary using an LLM. Includes risk flag
detection and well-crafted prompts for clinical accuracy.

The summarizer operates in two modes:
1. With OpenAI API key: Uses GPT-4o-mini for generation
2. Without API key: Returns a template-based mock summary for development
"""

import json
import re
from datetime import date
from typing import Any, Optional

from rich.console import Console

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None  # type: ignore[assignment, misc]

from src.config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
    RISK_KEYWORDS,
)
from src.ingest import extract_metadata
from src.models import (
    RiskAssessment,
    RiskLevel,
    SessionSummary,
    SimilarSession,
)
from src.prompts import (
    SYSTEM_PROMPT_RISK_ASSESSMENT,
    SYSTEM_PROMPT_SUMMARIZER,
    USER_PROMPT_RISK,
    USER_PROMPT_SIMILAR_ANALYSIS,
    USER_PROMPT_SUMMARIZE,
)
from src.retriever import RetrievalResult, TherapySessionRetriever
from src.logging_config import get_logger

console = Console()
logger = get_logger(__name__)


class SessionSummarizer:
    """Generates structured summaries of therapy sessions using RAG.

    Combines retrieval of similar past sessions with LLM summarization
    to produce clinically grounded, structured session summaries.
    """

    def __init__(self) -> None:
        """Initialize the summarizer with retriever and LLM."""
        self._retriever = TherapySessionRetriever()
        self._llm = self._init_llm()

    def _init_llm(self) -> Any:
        """Initialize the LLM if API key is available."""
        if OPENAI_API_KEY and ChatOpenAI is not None:
            return ChatOpenAI(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                openai_api_key=OPENAI_API_KEY,
            )
        return None

    def summarize_session(
        self,
        transcript: str,
        client_id: Optional[str] = None,
        include_similar: bool = True,
    ) -> SessionSummary:
        """Generate a structured summary of a therapy session.

        Pipeline:
        1. Extract metadata from the transcript
        2. Retrieve similar past sessions (if enabled)
        3. Detect risk flags using keyword analysis
        4. Generate structured summary via LLM (or mock)

        Args:
            transcript: Full therapy session transcript text.
            client_id: Optional client ID for filtered retrieval.
            include_similar: Whether to retrieve similar past sessions.

        Returns:
            A complete SessionSummary object.
        """
        logger.info(
            "Summarization request received",
            extra={
                "client_id": client_id,
                "include_similar": include_similar,
                "transcript_length": len(transcript),
            },
        )
        console.print("\n[bold blue]Generating Session Summary[/bold blue]")

        # Step 1: Extract metadata
        metadata = extract_metadata(transcript, "input_transcript")
        session_client_id = client_id or metadata.get("client_id", "unknown")
        console.print(f"  Client: {session_client_id}")

        # Step 2: Retrieve similar sessions
        similar_sessions: list[SimilarSession] = []
        similar_context = "No similar past sessions available."

        if include_similar and self._retriever.collection_count > 0:
            console.print("  Retrieving similar sessions...")
            retrieval_results = self._retriever.find_similar_sessions(
                query_text=transcript[:2000],  # Use first 2000 chars as query
                top_k=3,
                exclude_client=session_client_id,
            )
            similar_sessions = self._format_similar_sessions(retrieval_results)
            similar_context = self._format_context_for_prompt(retrieval_results)
            console.print(f"  Found {len(similar_sessions)} similar sessions")

        # Step 3: Risk assessment
        console.print("  Running risk assessment...")
        risk_assessment = self._assess_risk(transcript)
        logger.info(
            "Risk assessment completed",
            extra={
                "client_id": session_client_id,
                "risk_level": risk_assessment.level.value,
                "risk_factor_count": len(risk_assessment.factors),
                "protective_factor_count": len(risk_assessment.protective_factors),
            },
        )
        console.print(f"  Risk level: {risk_assessment.level.value}")

        # Step 4: Generate summary
        console.print("  Generating summary...")
        if self._llm is not None:
            summary = self._generate_with_llm(
                transcript=transcript,
                similar_context=similar_context,
                metadata=metadata,
                risk_assessment=risk_assessment,
                similar_sessions=similar_sessions,
            )
        else:
            console.print("  [yellow]No API key — using template-based summary[/yellow]")
            summary = self._generate_mock_summary(
                transcript=transcript,
                metadata=metadata,
                risk_assessment=risk_assessment,
                similar_sessions=similar_sessions,
            )

        logger.info(
            "Summary generation completed",
            extra={
                "client_id": session_client_id,
                "session_id": summary.session_id,
                "risk_level": summary.risk_assessment.level.value,
                "similar_sessions_count": len(summary.similar_sessions),
                "used_llm": self._llm is not None,
            },
        )
        console.print("  [green]Summary generated successfully[/green]")
        return summary

    def _assess_risk(self, transcript: str) -> RiskAssessment:
        """Perform keyword-based risk assessment on a transcript.

        Uses a combination of keyword detection and contextual heuristics
        to classify risk level. In production, this would be augmented
        with an LLM-based assessment.

        Args:
            transcript: The session transcript text.

        Returns:
            RiskAssessment object.
        """
        text_lower = transcript.lower()

        # Detect risk factors
        found_keywords: list[str] = [
            kw for kw in RISK_KEYWORDS if kw.lower() in text_lower
        ]

        # Detect protective factors (keyword-based heuristic)
        protective_keywords = [
            "denies suicidal", "denied suicidal", "no suicidal ideation",
            "denies self-harm", "no self-harm", "safety plan",
            "supportive", "engaged in treatment", "protective factors",
            "children", "family", "strong support",
        ]
        found_protective: list[str] = [
            kw for kw in protective_keywords if kw.lower() in text_lower
        ]

        # Classify risk level
        denial_present = any(
            phrase in text_lower for phrase in [
                "denies suicidal", "denied suicidal", "no suicidal ideation",
                "denies self-harm", "denied self-harm",
            ]
        )

        active_indicators = [
            "suicidal ideation", "kill myself", "end it",
            "want to die", "plan to harm",
        ]
        active_present = any(ind in text_lower for ind in active_indicators)

        passive_indicators = [
            "passive death wishes", "not wanting to be alive",
            "better off without me", "want to disappear",
            "no reason to live", "wish i was dead",
        ]
        passive_present = any(ind in text_lower for ind in passive_indicators)

        # Risk level determination
        if active_present and not denial_present:
            level = RiskLevel.MODERATE
        elif passive_present and not denial_present:
            level = RiskLevel.LOW
        elif found_keywords and denial_present:
            level = RiskLevel.LOW
        elif found_keywords:
            level = RiskLevel.LOW
        else:
            level = RiskLevel.NONE

        # Recommended actions based on level
        actions: list[str] = []
        if level == RiskLevel.NONE:
            actions = ["Continue routine monitoring"]
        elif level == RiskLevel.LOW:
            actions = [
                "Monitor at each session",
                "Maintain updated safety plan",
                "Document risk and protective factors",
            ]
        elif level == RiskLevel.MODERATE:
            actions = [
                "Increase session frequency",
                "Update safety plan with specific lethal means restriction",
                "Consider psychiatric consultation for medication",
                "Document detailed risk assessment",
                "Coordinate with emergency contacts",
            ]
        elif level in (RiskLevel.HIGH, RiskLevel.IMMINENT):
            actions = [
                "Immediate safety evaluation",
                "Consider hospitalization",
                "Restrict access to lethal means",
                "Contact emergency services if needed",
                "Do not leave client alone",
            ]

        return RiskAssessment(
            level=level,
            factors=found_keywords,
            protective_factors=found_protective,
            recommended_actions=actions,
        )

    def _format_similar_sessions(
        self,
        results: list[RetrievalResult],
    ) -> list[SimilarSession]:
        """Convert retrieval results to SimilarSession model objects.

        Args:
            results: Raw retrieval results.

        Returns:
            List of SimilarSession objects.
        """
        similar: list[SimilarSession] = []
        for r in results:
            similar.append(
                SimilarSession(
                    session_id=r.source_file.replace(".txt", ""),
                    client_id=r.client_id,
                    similarity_score=r.similarity_score,
                    relevant_excerpt=r.text[:500],
                    relevance_reason=(
                        f"Similar clinical content from {r.client_id} "
                        f"(session date: {r.session_date})"
                    ),
                )
            )
        return similar

    def _format_context_for_prompt(
        self,
        results: list[RetrievalResult],
    ) -> str:
        """Format retrieval results into a prompt context string.

        Args:
            results: Retrieval results to format.

        Returns:
            Formatted string for prompt insertion.
        """
        if not results:
            return "No similar past sessions available."

        context_parts: list[str] = []
        for i, r in enumerate(results):
            context_parts.append(
                f"### Similar Session {i + 1}\n"
                f"Source: {r.source_file} | Client: {r.client_id} | "
                f"Date: {r.session_date} | Similarity: {r.similarity_score:.3f}\n"
                f"Excerpt: {r.text[:500]}"
            )
        return "\n\n".join(context_parts)

    def _generate_with_llm(
        self,
        transcript: str,
        similar_context: str,
        metadata: dict[str, Any],
        risk_assessment: RiskAssessment,
        similar_sessions: list[SimilarSession],
    ) -> SessionSummary:
        """Generate a summary using the LLM.

        Args:
            transcript: Full transcript text.
            similar_context: Formatted similar session context.
            metadata: Extracted session metadata.
            risk_assessment: Pre-computed risk assessment.
            similar_sessions: Similar session objects.

        Returns:
            SessionSummary generated by the LLM.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=SYSTEM_PROMPT_SUMMARIZER),
            HumanMessage(
                content=USER_PROMPT_SUMMARIZE.format(
                    transcript=transcript[:4000],
                    similar_sessions_context=similar_context,
                )
            ),
        ]

        response = self._llm.invoke(messages)
        response_text = response.content

        # Parse JSON response
        try:
            # Clean potential markdown code blocks
            cleaned = re.sub(r"```json\s*", "", response_text)
            cleaned = re.sub(r"```\s*$", "", cleaned)
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            console.print("[yellow]LLM response not valid JSON, "
                          "using template summary[/yellow]")
            return self._generate_mock_summary(
                transcript, metadata, risk_assessment, similar_sessions
            )

        # Build SessionSummary from LLM output + pre-computed fields
        session_date = None
        if metadata.get("session_date"):
            try:
                session_date = date.fromisoformat(metadata["session_date"])
            except (ValueError, TypeError):
                pass

        return SessionSummary(
            session_id=metadata.get("source_file", "input").replace(".txt", ""),
            client_id=metadata.get("client_id", "unknown"),
            session_date=session_date,
            session_number=metadata.get("session_number"),
            clinician=metadata.get("clinician"),
            presenting_problem=data.get("presenting_problem", ""),
            mood_indicators=data.get("mood_indicators", []),
            key_themes=data.get("key_themes", []),
            therapeutic_interventions=data.get("therapeutic_interventions", []),
            client_progress=data.get("client_progress", ""),
            homework_assigned=data.get("homework_assigned", []),
            risk_assessment=risk_assessment,
            recommended_followup=data.get("recommended_followup", []),
            similar_sessions=similar_sessions,
        )

    def _generate_mock_summary(
        self,
        transcript: str,
        metadata: dict[str, Any],
        risk_assessment: RiskAssessment,
        similar_sessions: list[SimilarSession],
    ) -> SessionSummary:
        """Generate a template-based summary without LLM API.

        Provides a reasonable summary using text analysis heuristics.
        Useful for development and testing without API costs.

        Args:
            transcript: Full transcript text.
            metadata: Extracted session metadata.
            risk_assessment: Pre-computed risk assessment.
            similar_sessions: Similar session objects.

        Returns:
            SessionSummary with heuristically extracted content.
        """
        text_lower = transcript.lower()

        # Extract presenting problem from first substantive exchange
        presenting = "See transcript for session details."
        lines = transcript.split("\n")
        for line in lines:
            if line.strip().startswith("Client:") and len(line) > 50:
                presenting = line.replace("Client:", "").strip()[:200]
                break

        # Detect mood indicators
        mood_indicators: list[str] = []
        mood_terms = {
            "flat affect": "flat affect",
            "tearful": "tearful",
            "anxious": "anxious presentation",
            "irritable": "irritable",
            "low mood": "low mood reported",
            "depressed": "depressed mood",
            "fatigued": "fatigued appearance",
        }
        for term, indicator in mood_terms.items():
            if term in text_lower:
                mood_indicators.append(indicator)

        # Detect PHQ-9, GAD-7, or other scores
        score_patterns = [
            r"PHQ-9[:\s]+(\d+)",
            r"GAD-7[:\s]+(\d+)",
            r"SUDS[:\s]+(\d+)",
            r"PCL-5[:\s]+(\d+)",
            r"AUDIT[:\s]+",
        ]
        for pattern in score_patterns:
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                mood_indicators.append(match.group(0).strip())

        # Detect interventions
        interventions: list[str] = []
        intervention_keywords = {
            "cbt": "Cognitive Behavioral Therapy (CBT)",
            "cognitive restructuring": "Cognitive restructuring",
            "behavioral activation": "Behavioral activation",
            "exposure": "Exposure therapy",
            "empty chair": "Empty chair technique (Gestalt)",
            "mindfulness": "Mindfulness exercise",
            "breathing": "Breathing technique",
            "safety plan": "Safety planning",
            "gottman": "Gottman method",
            "erp": "Exposure and Response Prevention (ERP)",
            "imaginal exposure": "Imaginal exposure",
            "dbt": "Dialectical Behavior Therapy (DBT)",
            "prolonged exposure": "Prolonged Exposure (PE)",
            "cpt": "Cognitive Processing Therapy (CPT)",
            "self-compassion": "Self-compassion exercise",
        }
        for keyword, intervention in intervention_keywords.items():
            if keyword in text_lower:
                interventions.append(intervention)

        # Detect homework
        homework: list[str] = []
        homework_section = False
        for line in lines:
            if "homework" in line.lower() or "assignment" in line.lower():
                homework_section = True
            if "this week" in line.lower() and ("like you to" in line.lower()
                                                 or "want you to" in line.lower()):
                homework.append(line.strip()[:200])

        # Build session date
        session_date = None
        if metadata.get("session_date"):
            try:
                session_date = date.fromisoformat(metadata["session_date"])
            except (ValueError, TypeError):
                pass

        return SessionSummary(
            session_id=metadata.get("source_file", "input").replace(".txt", ""),
            client_id=metadata.get("client_id", "unknown"),
            session_date=session_date,
            session_number=metadata.get("session_number"),
            clinician=metadata.get("clinician"),
            presenting_problem=presenting,
            mood_indicators=mood_indicators if mood_indicators else [
                "See transcript for detailed mood assessment"
            ],
            key_themes=["See transcript for therapeutic themes discussed"],
            therapeutic_interventions=interventions if interventions else [
                "See transcript for interventions used"
            ],
            client_progress="See transcript for progress update.",
            homework_assigned=homework if homework else [
                "See transcript for between-session assignments"
            ],
            risk_assessment=risk_assessment,
            recommended_followup=["Continue current treatment plan"],
            similar_sessions=similar_sessions,
        )


def demo_summarizer() -> None:
    """Demonstrate the summarizer with a sample transcript."""
    console.print("\n[bold blue]Session Summarizer — Demo[/bold blue]\n")

    # Load a sample transcript
    from src.config import TRANSCRIPTS_DIR

    sample_file = TRANSCRIPTS_DIR / "session_001.txt"
    if not sample_file.exists():
        console.print("[red]Sample transcript not found. Ensure data is present.[/red]")
        return

    transcript = sample_file.read_text(encoding="utf-8")

    summarizer = SessionSummarizer()
    summary = summarizer.summarize_session(transcript)

    # Print summary
    console.print("\n[bold green]Generated Summary:[/bold green]")
    console.print_json(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    demo_summarizer()
