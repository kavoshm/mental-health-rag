"""
Ingestion Pipeline for Therapy Session Transcripts.

Loads transcripts from the data/transcripts/ directory, splits them using
a dialogue-aware strategy that preserves therapist-client exchange context,
generates embeddings, and stores everything in a persistent ChromaDB
collection.

Usage:
    python -m src.ingest                    # Ingest all transcripts
    python -m src.ingest --reset            # Clear collection and re-ingest
    python -m src.ingest --file session_001.txt  # Ingest a single file
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None  # type: ignore[assignment, misc]

import chromadb
from rich.console import Console
from rich.table import Table

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_PERSIST_DIR,
    CLINICAL_SEPARATORS,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    SIMILARITY_METRIC,
    TRANSCRIPTS_DIR,
)
from src.logging_config import get_logger

console = Console()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metadata Extraction
# ---------------------------------------------------------------------------

def extract_metadata(content: str, filename: str) -> dict[str, Any]:
    """Extract structured metadata from transcript header.

    Parses the SESSION METADATA block at the top of each transcript file
    to extract date, session number, client ID, and clinician information.

    Args:
        content: Full transcript text.
        filename: Source filename for reference.

    Returns:
        Dictionary of extracted metadata fields.
    """
    metadata: dict[str, Any] = {"source_file": filename}
    logger.debug("Extracting metadata", extra={"filename": filename})

    patterns: dict[str, str] = {
        "session_date": r"Date:\s*(\d{4}-\d{2}-\d{2})",
        "session_number": r"Session Number:\s*(\d+)",
        "client_id": r"Client ID:\s*([\w-]+)",
        "clinician": r"Clinician:\s*(.+?)(?:\n|$)",
        "session_type": r"Session Type:\s*(.+?)(?:\n|$)",
        "duration": r"Duration:\s*(.+?)(?:\n|$)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            value = match.group(1).strip()
            # Convert session_number to int for filtering
            if key == "session_number":
                try:
                    metadata[key] = int(value)
                except ValueError:
                    metadata[key] = value
            else:
                metadata[key] = value

    return metadata


def extract_transcript_body(content: str) -> str:
    """Extract the dialogue portion of a transcript, excluding metadata header.

    Args:
        content: Full transcript text including metadata header.

    Returns:
        The transcript body (dialogue section only).
    """
    # Split on the "---" separator between metadata and transcript
    parts = content.split("---", maxsplit=1)
    if len(parts) > 1:
        body = parts[1].strip()
        # Remove the "TRANSCRIPT" header if present
        body = re.sub(r"^TRANSCRIPT\s*\n", "", body, flags=re.IGNORECASE)
        return body.strip()
    return content


# ---------------------------------------------------------------------------
# Text Splitting
# ---------------------------------------------------------------------------

def create_therapy_splitter() -> RecursiveCharacterTextSplitter:
    """Create a text splitter tuned for therapy session transcripts.

    Uses custom separators that respect dialogue turn boundaries, keeping
    therapist-client exchanges together as much as possible.

    Returns:
        Configured RecursiveCharacterTextSplitter instance.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CLINICAL_SEPARATORS,
        keep_separator=True,
        strip_whitespace=True,
    )


def split_transcript(
    content: str,
    metadata: dict[str, Any],
) -> list[Document]:
    """Split a therapy transcript into chunks with metadata.

    Args:
        content: The transcript body text.
        metadata: Metadata dictionary to attach to each chunk.

    Returns:
        List of Document objects, each representing a chunk.
    """
    splitter = create_therapy_splitter()
    chunks = splitter.split_text(content)

    documents: list[Document] = []
    for i, chunk in enumerate(chunks):
        chunk_metadata = {
            **metadata,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunk_length": len(chunk),
        }
        documents.append(Document(page_content=chunk, metadata=chunk_metadata))

    return documents


# ---------------------------------------------------------------------------
# ChromaDB Operations
# ---------------------------------------------------------------------------

def get_chroma_client() -> chromadb.Client:
    """Get or create a persistent ChromaDB client.

    Returns:
        ChromaDB client instance.
    """
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    return chromadb.Client(
        chromadb.Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(CHROMA_PERSIST_DIR),
            anonymized_telemetry=False,
        )
    )


def get_or_create_collection(
    client: chromadb.Client,
) -> chromadb.Collection:
    """Get or create the therapy sessions collection.

    Args:
        client: ChromaDB client instance.

    Returns:
        ChromaDB collection.
    """
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": SIMILARITY_METRIC},
    )


def add_documents_to_collection(
    collection: chromadb.Collection,
    documents: list[Document],
    embedding_fn: Any = None,
) -> int:
    """Add document chunks to the ChromaDB collection.

    If an OpenAI API key is available, uses OpenAI embeddings. Otherwise,
    falls back to ChromaDB's default embedding function.

    Args:
        collection: The target ChromaDB collection.
        documents: List of Document objects to add.
        embedding_fn: Optional embedding function override.

    Returns:
        Number of documents added.
    """
    if not documents:
        logger.info("No documents to add to collection")
        return 0

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for doc in documents:
        # Create a unique ID from source file and chunk index
        source = doc.metadata.get("source_file", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        doc_id = f"{Path(source).stem}_chunk_{chunk_idx:03d}"

        ids.append(doc_id)
        texts.append(doc.page_content)

        # ChromaDB requires metadata values to be str, int, float, or bool
        clean_meta = {
            k: v for k, v in doc.metadata.items()
            if isinstance(v, (str, int, float, bool))
        }
        metadatas.append(clean_meta)

    logger.info(
        "Adding documents to collection",
        extra={"document_count": len(ids), "has_api_key": bool(OPENAI_API_KEY)},
    )

    # Generate embeddings if API key is available
    if OPENAI_API_KEY and OpenAIEmbeddings is not None:
        console.print(f"  Using OpenAI embeddings ({EMBEDDING_MODEL})")
        embedder = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )
        embeddings = embedder.embed_documents(texts)
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
    else:
        console.print("  Using ChromaDB default embeddings (no API key)")
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

    return len(ids)


# ---------------------------------------------------------------------------
# Ingestion Pipeline
# ---------------------------------------------------------------------------

def load_transcripts(
    directory: Optional[Path] = None,
    single_file: Optional[str] = None,
) -> list[tuple[str, str]]:
    """Load transcript files from the data directory.

    Args:
        directory: Directory containing transcript files.
        single_file: If provided, load only this file.

    Returns:
        List of (filename, content) tuples.
    """
    if directory is None:
        directory = TRANSCRIPTS_DIR

    transcripts: list[tuple[str, str]] = []

    if single_file:
        filepath = directory / single_file
        if filepath.exists():
            transcripts.append((single_file, filepath.read_text(encoding="utf-8")))
        else:
            console.print(f"[red]File not found: {filepath}[/red]")
    else:
        for filepath in sorted(directory.glob("*.txt")):
            content = filepath.read_text(encoding="utf-8")
            transcripts.append((filepath.name, content))

    return transcripts


def ingest_transcripts(
    reset: bool = False,
    single_file: Optional[str] = None,
) -> dict[str, Any]:
    """Run the full ingestion pipeline.

    1. Load transcripts from the data directory
    2. Extract metadata from each transcript
    3. Split transcripts into chunks
    4. Store chunks with embeddings in ChromaDB

    Args:
        reset: If True, delete and recreate the collection.
        single_file: If provided, ingest only this file.

    Returns:
        Summary dictionary with ingestion statistics.
    """
    logger.info(
        "Ingestion pipeline started",
        extra={"reset": reset, "single_file": single_file},
    )
    console.print("\n[bold blue]Mental Health RAG — Ingestion Pipeline[/bold blue]\n")

    # Load transcripts
    console.print("[bold]Step 1: Loading transcripts[/bold]")
    transcripts = load_transcripts(single_file=single_file)
    console.print(f"  Found {len(transcripts)} transcript(s)")

    if not transcripts:
        logger.warning("No transcripts found in data directory")
        console.print("[red]No transcripts found. Exiting.[/red]")
        return {"status": "error", "message": "No transcripts found"}

    # Initialize ChromaDB
    console.print("\n[bold]Step 2: Initializing ChromaDB[/bold]")
    client = get_chroma_client()

    if reset:
        console.print("  [yellow]Resetting collection...[/yellow]")
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # Collection may not exist yet

    collection = get_or_create_collection(client)
    initial_count = collection.count()
    console.print(f"  Collection '{COLLECTION_NAME}' — {initial_count} existing docs")

    # Process each transcript
    console.print("\n[bold]Step 3: Processing transcripts[/bold]")
    total_chunks = 0
    results: list[dict[str, Any]] = []

    for filename, content in transcripts:
        logger.info("Processing transcript", extra={"filename": filename})
        console.print(f"\n  Processing: {filename}")

        # Extract metadata
        metadata = extract_metadata(content, filename)
        console.print(f"    Client: {metadata.get('client_id', 'unknown')}, "
                       f"Session: {metadata.get('session_number', '?')}, "
                       f"Date: {metadata.get('session_date', 'unknown')}")

        # Extract transcript body
        body = extract_transcript_body(content)
        console.print(f"    Body length: {len(body)} chars")

        # Split into chunks
        documents = split_transcript(body, metadata)
        console.print(f"    Chunks: {len(documents)}")

        # Add to collection
        added = add_documents_to_collection(collection, documents)
        total_chunks += added

        results.append({
            "filename": filename,
            "client_id": metadata.get("client_id", "unknown"),
            "chunks": len(documents),
        })

    # Summary
    final_count = collection.count()

    console.print("\n[bold]Step 4: Ingestion Summary[/bold]")
    table = Table(title="Ingested Transcripts")
    table.add_column("File", style="cyan")
    table.add_column("Client ID", style="green")
    table.add_column("Chunks", justify="right", style="magenta")

    for result in results:
        table.add_row(
            result["filename"],
            result["client_id"],
            str(result["chunks"]),
        )

    console.print(table)
    logger.info(
        "Ingestion pipeline completed",
        extra={
            "transcripts_processed": len(transcripts),
            "total_chunks": total_chunks,
            "collection_size": final_count,
        },
    )
    console.print(f"\n  Total chunks added: {total_chunks}")
    console.print(f"  Collection size: {final_count}")
    console.print(f"  Persist directory: {CHROMA_PERSIST_DIR}")

    # Persist
    try:
        client.persist()
        console.print("  [green]Collection persisted to disk.[/green]")
    except AttributeError:
        pass  # Some ChromaDB versions auto-persist

    return {
        "status": "success",
        "transcripts_processed": len(transcripts),
        "total_chunks": total_chunks,
        "collection_size": final_count,
        "details": results,
    }


def main() -> None:
    """CLI entry point for the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest therapy session transcripts into ChromaDB"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the collection and re-ingest all transcripts",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Ingest a single transcript file (filename only, not path)",
    )
    args = parser.parse_args()

    result = ingest_transcripts(reset=args.reset, single_file=args.file)

    if result["status"] != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
