"""
CLI Entry Point for the Mental Health Session Summary RAG Pipeline.

Provides a command-line interface for:
- Ingesting therapy transcripts into ChromaDB
- Querying the vector store for similar sessions
- Generating session summaries
- Running the FastAPI server

Usage:
    python -m src.main ingest               # Ingest all transcripts
    python -m src.main ingest --reset       # Re-ingest from scratch
    python -m src.main query "anxiety"      # Search for similar sessions
    python -m src.main summarize session_001.txt  # Summarize a transcript
    python -m src.main serve                # Start the API server
    python -m src.main status               # Show collection status
"""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import CHROMA_PERSIST_DIR, OUTPUTS_DIR, TRANSCRIPTS_DIR

console = Console()


def cmd_ingest(args: argparse.Namespace) -> None:
    """Run the ingestion pipeline."""
    from src.ingest import ingest_transcripts

    result = ingest_transcripts(
        reset=args.reset,
        single_file=args.file,
    )

    if result["status"] != "success":
        console.print("[red]Ingestion failed.[/red]")
        sys.exit(1)


def cmd_query(args: argparse.Namespace) -> None:
    """Query the vector store for similar sessions."""
    from src.retriever import TherapySessionRetriever

    retriever = TherapySessionRetriever()

    if retriever.collection_count == 0:
        console.print("[yellow]Collection is empty. Run 'ingest' first.[/yellow]")
        return

    console.print(f"\n[bold]Searching for: '{args.query}'[/bold]")
    console.print(f"Top-K: {args.top_k}")

    if args.client:
        console.print(f"Filtering by client: {args.client}")

    results = retriever.find_similar_sessions(
        query_text=args.query,
        top_k=args.top_k,
        client_id=args.client,
    )

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title="Retrieval Results")
    table.add_column("Score", justify="right", style="cyan")
    table.add_column("Source", style="green")
    table.add_column("Client", style="magenta")
    table.add_column("Date")
    table.add_column("Excerpt", max_width=60)

    for r in results:
        table.add_row(
            f"{r.similarity_score:.3f}",
            r.source_file,
            r.client_id,
            r.session_date,
            r.text[:100] + "...",
        )

    console.print(table)

    # Optionally save results
    if args.output:
        output_data = [r.to_dict() for r in results]
        output_path = Path(args.output)
        output_path.write_text(json.dumps(output_data, indent=2))
        console.print(f"\nResults saved to {output_path}")


def cmd_summarize(args: argparse.Namespace) -> None:
    """Summarize a therapy session transcript."""
    from src.summarizer import SessionSummarizer

    # Load transcript
    transcript_path = TRANSCRIPTS_DIR / args.file
    if not transcript_path.exists():
        console.print(f"[red]File not found: {transcript_path}[/red]")
        sys.exit(1)

    transcript = transcript_path.read_text(encoding="utf-8")
    console.print(f"\n[bold]Summarizing: {args.file}[/bold]")
    console.print(f"Transcript length: {len(transcript)} characters")

    summarizer = SessionSummarizer()
    summary = summarizer.summarize_session(
        transcript=transcript,
        include_similar=not args.no_similar,
    )

    # Display summary
    console.print(Panel(
        summary.model_dump_json(indent=2),
        title="Session Summary",
        border_style="green",
    ))

    # Save to outputs
    if args.save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUTS_DIR / f"summary_{args.file.replace('.txt', '.json')}"
        output_file.write_text(summary.model_dump_json(indent=2))
        console.print(f"\nSaved to {output_file}")


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI server."""
    import uvicorn

    console.print(f"\n[bold blue]Starting API server on port {args.port}[/bold blue]")
    console.print("Documentation: http://localhost:{}/docs".format(args.port))

    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def cmd_status(args: argparse.Namespace) -> None:
    """Show collection and system status."""
    from src.retriever import TherapySessionRetriever

    console.print("\n[bold blue]System Status[/bold blue]\n")

    # Check transcripts
    transcript_files = list(TRANSCRIPTS_DIR.glob("*.txt"))
    console.print(f"Transcripts directory: {TRANSCRIPTS_DIR}")
    console.print(f"Transcript files: {len(transcript_files)}")

    # Check ChromaDB
    console.print(f"\nChromaDB directory: {CHROMA_PERSIST_DIR}")
    console.print(f"ChromaDB exists: {CHROMA_PERSIST_DIR.exists()}")

    try:
        retriever = TherapySessionRetriever()
        count = retriever.collection_count
        console.print(f"Collection chunks: {count}")

        sessions = retriever.list_sessions()
        if sessions:
            table = Table(title="Ingested Sessions")
            table.add_column("Session ID", style="cyan")
            table.add_column("Client", style="green")
            table.add_column("Date")
            table.add_column("Chunks", justify="right", style="magenta")

            for s in sessions:
                table.add_row(
                    s["session_id"],
                    s["client_id"],
                    s.get("session_date", "N/A"),
                    str(s["chunk_count"]),
                )
            console.print(table)
    except Exception as e:
        console.print(f"[red]ChromaDB error: {e}[/red]")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Mental Health Session Summary RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest transcripts")
    ingest_parser.add_argument("--reset", action="store_true",
                                help="Clear and re-ingest")
    ingest_parser.add_argument("--file", type=str, default=None,
                                help="Ingest a single file")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query similar sessions")
    query_parser.add_argument("query", type=str, help="Search query text")
    query_parser.add_argument("--top-k", type=int, default=5,
                               help="Number of results")
    query_parser.add_argument("--client", type=str, default=None,
                               help="Filter by client ID")
    query_parser.add_argument("--output", type=str, default=None,
                               help="Save results to JSON file")

    # Summarize command
    summarize_parser = subparsers.add_parser("summarize",
                                              help="Summarize a transcript")
    summarize_parser.add_argument("file", type=str,
                                   help="Transcript filename (e.g., session_001.txt)")
    summarize_parser.add_argument("--no-similar", action="store_true",
                                   help="Skip similar session retrieval")
    summarize_parser.add_argument("--save", action="store_true",
                                   help="Save summary to outputs/")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0",
                               help="Server host")
    serve_parser.add_argument("--port", type=int, default=8000,
                               help="Server port")
    serve_parser.add_argument("--reload", action="store_true",
                               help="Enable auto-reload")

    # Status command
    subparsers.add_parser("status", help="Show system status")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "summarize": cmd_summarize,
        "serve": cmd_serve,
        "status": cmd_status,
    }

    cmd_fn = commands.get(args.command)
    if cmd_fn:
        cmd_fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
