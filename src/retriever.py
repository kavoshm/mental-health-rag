"""
Custom Retrieval Logic for Therapy Session RAG.

Provides retrieval functions that go beyond basic similarity search:
- Find similar past sessions given a new transcript
- Filter by client ID, date range, or risk indicators
- Return results with rich metadata for clinical context

Supports both OpenAI embeddings (when API key is available) and
ChromaDB's default embeddings for local development.
"""

from datetime import datetime
from typing import Any, Optional

import chromadb
from rich.console import Console

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None  # type: ignore[assignment, misc]

from src.config import (
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    RISK_KEYWORDS,
)
from src.ingest import get_chroma_client, get_or_create_collection
from src.logging_config import get_logger

console = Console()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Classes for Retrieval Results
# ---------------------------------------------------------------------------

class RetrievalResult:
    """A single retrieval result with metadata and scoring."""

    def __init__(
        self,
        chunk_id: str,
        text: str,
        metadata: dict[str, Any],
        distance: float,
    ) -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata
        self.distance = distance
        self.similarity_score = max(0.0, 1.0 - distance)

    @property
    def client_id(self) -> str:
        return self.metadata.get("client_id", "unknown")

    @property
    def session_date(self) -> str:
        return self.metadata.get("session_date", "unknown")

    @property
    def source_file(self) -> str:
        return self.metadata.get("source_file", "unknown")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for API responses."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "similarity_score": round(self.similarity_score, 4),
        }


# ---------------------------------------------------------------------------
# Retriever Class
# ---------------------------------------------------------------------------

class TherapySessionRetriever:
    """Retrieves similar therapy sessions and relevant context from ChromaDB.

    Supports:
    - Basic similarity search
    - Metadata-filtered search (by client, date, session type)
    - Risk-focused retrieval
    - Multi-session context aggregation
    """

    def __init__(self) -> None:
        """Initialize the retriever with ChromaDB connection."""
        self._client = get_chroma_client()
        self._collection = get_or_create_collection(self._client)
        self._embedder = self._init_embedder()

    def _init_embedder(self) -> Any:
        """Initialize the embedding function if API key is available."""
        if OPENAI_API_KEY and OpenAIEmbeddings is not None:
            return OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY,
            )
        return None

    @property
    def collection_count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    def find_similar_sessions(
        self,
        query_text: str,
        top_k: int = DEFAULT_TOP_K,
        client_id: Optional[str] = None,
        exclude_client: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """Find similar therapy session chunks based on a query.

        Args:
            query_text: The text to search for (new transcript or query).
            top_k: Maximum number of results to return.
            client_id: If provided, only return results from this client.
            exclude_client: If provided, exclude results from this client.
            date_from: ISO date string for earliest session date.
            date_to: ISO date string for latest session date.

        Returns:
            List of RetrievalResult objects sorted by similarity.
        """
        # Build metadata filter
        where_filter = self._build_where_filter(
            client_id=client_id,
            exclude_client=exclude_client,
            date_from=date_from,
            date_to=date_to,
        )

        # Query ChromaDB
        query_kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(top_k, self._collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        try:
            results = self._collection.query(**query_kwargs)
        except Exception as e:
            logger.error(
                "Retrieval query failed",
                extra={"error": str(e), "top_k": top_k},
                exc_info=True,
            )
            console.print(f"[red]Retrieval error: {e}[/red]")
            return []

        parsed = self._parse_results(results)
        logger.info(
            "Similarity search completed",
            extra={
                "top_k": top_k,
                "results_returned": len(parsed),
                "client_filter": client_id,
                "exclude_client": exclude_client,
            },
        )
        return parsed

    def find_risk_relevant_sessions(
        self,
        query_text: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[RetrievalResult]:
        """Find sessions relevant to risk assessment.

        Searches for the query AND adds risk-specific keywords to bias
        retrieval toward safety-relevant content.

        Args:
            query_text: The text to search for.
            top_k: Maximum number of results.

        Returns:
            List of RetrievalResult objects with risk-relevant content.
        """
        # Augment query with risk keywords for better retrieval
        risk_query = f"{query_text} {' '.join(RISK_KEYWORDS[:5])}"

        results = self.find_similar_sessions(
            query_text=risk_query,
            top_k=top_k,
        )

        # Re-score based on presence of risk keywords in retrieved text
        for result in results:
            risk_count = sum(
                1 for keyword in RISK_KEYWORDS
                if keyword.lower() in result.text.lower()
            )
            if risk_count > 0:
                # Boost relevance score for risk-containing chunks
                result.metadata["risk_keyword_count"] = risk_count
                result.metadata["contains_risk_content"] = True

        return results

    def get_client_history(
        self,
        client_id: str,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        """Retrieve all stored chunks for a specific client.

        Args:
            client_id: The client identifier to search for.
            top_k: Maximum number of chunks to return.

        Returns:
            List of RetrievalResult objects for the client.
        """
        logger.info("Retrieving client history", extra={"client_id": client_id, "top_k": top_k})
        try:
            results = self._collection.get(
                where={"client_id": client_id},
                include=["documents", "metadatas"],
                limit=top_k,
            )
        except Exception as e:
            logger.error("Client history retrieval failed", extra={"client_id": client_id, "error": str(e)}, exc_info=True)
            console.print(f"[red]Client history error: {e}[/red]")
            return []

        retrieval_results: list[RetrievalResult] = []
        if results and results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                retrieval_results.append(
                    RetrievalResult(
                        chunk_id=chunk_id,
                        text=results["documents"][i] if results["documents"] else "",
                        metadata=results["metadatas"][i] if results["metadatas"] else {},
                        distance=0.0,  # Not a similarity search
                    )
                )
        return retrieval_results

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all ingested sessions with summary info.

        Returns:
            List of session summary dictionaries.
        """
        try:
            all_docs = self._collection.get(
                include=["metadatas"],
            )
        except Exception:
            return []

        # Group by source file
        sessions: dict[str, dict[str, Any]] = {}
        if all_docs and all_docs["metadatas"]:
            for meta in all_docs["metadatas"]:
                source = meta.get("source_file", "unknown")
                if source not in sessions:
                    sessions[source] = {
                        "session_id": source.replace(".txt", ""),
                        "client_id": meta.get("client_id", "unknown"),
                        "session_date": meta.get("session_date"),
                        "session_number": meta.get("session_number"),
                        "chunk_count": 0,
                    }
                sessions[source]["chunk_count"] += 1

        return list(sessions.values())

    def _build_where_filter(
        self,
        client_id: Optional[str] = None,
        exclude_client: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Build a ChromaDB where filter from parameters.

        Args:
            client_id: Filter to this client only.
            exclude_client: Exclude this client.
            date_from: Earliest date (inclusive).
            date_to: Latest date (inclusive).

        Returns:
            ChromaDB-compatible where filter, or None.
        """
        conditions: list[dict[str, Any]] = []

        if client_id:
            conditions.append({"client_id": {"$eq": client_id}})
        if exclude_client:
            conditions.append({"client_id": {"$ne": exclude_client}})
        if date_from:
            conditions.append({"session_date": {"$gte": date_from}})
        if date_to:
            conditions.append({"session_date": {"$lte": date_to}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _parse_results(
        self,
        raw_results: dict[str, Any],
    ) -> list[RetrievalResult]:
        """Parse raw ChromaDB query results into RetrievalResult objects.

        Args:
            raw_results: Raw results from ChromaDB query.

        Returns:
            List of RetrievalResult objects.
        """
        results: list[RetrievalResult] = []

        if not raw_results or not raw_results.get("ids"):
            return results

        ids = raw_results["ids"][0]
        documents = raw_results["documents"][0] if raw_results.get("documents") else []
        metadatas = raw_results["metadatas"][0] if raw_results.get("metadatas") else []
        distances = raw_results["distances"][0] if raw_results.get("distances") else []

        for i, chunk_id in enumerate(ids):
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=documents[i] if i < len(documents) else "",
                    metadata=metadatas[i] if i < len(metadatas) else {},
                    distance=distances[i] if i < len(distances) else 1.0,
                )
            )

        return sorted(results, key=lambda r: r.distance)


def demo_retrieval() -> None:
    """Demonstrate retrieval functionality."""
    console.print("\n[bold blue]Therapy Session Retriever — Demo[/bold blue]\n")

    retriever = TherapySessionRetriever()
    count = retriever.collection_count

    if count == 0:
        console.print("[yellow]No documents in collection. Run ingestion first:[/yellow]")
        console.print("  python -m src.ingest")
        return

    console.print(f"Collection contains {count} chunks\n")

    # Demo 1: Basic similarity search
    console.print("[bold]Query: 'panic attacks and anxiety treatment'[/bold]")
    results = retriever.find_similar_sessions(
        query_text="panic attacks and anxiety treatment",
        top_k=3,
    )
    for r in results:
        console.print(f"  [{r.similarity_score:.3f}] {r.client_id} — "
                       f"{r.text[:120]}...")

    # Demo 2: Risk-focused search
    console.print("\n[bold]Risk-focused: 'suicidal ideation self-harm'[/bold]")
    risk_results = retriever.find_risk_relevant_sessions(
        query_text="suicidal ideation self-harm",
        top_k=3,
    )
    for r in risk_results:
        risk_flag = r.metadata.get("contains_risk_content", False)
        console.print(f"  [{r.similarity_score:.3f}] risk={risk_flag} — "
                       f"{r.text[:120]}...")

    # Demo 3: Session listing
    console.print("\n[bold]All ingested sessions:[/bold]")
    sessions = retriever.list_sessions()
    for s in sessions:
        console.print(f"  {s['session_id']} | {s['client_id']} | "
                       f"chunks={s['chunk_count']}")


if __name__ == "__main__":
    demo_retrieval()
