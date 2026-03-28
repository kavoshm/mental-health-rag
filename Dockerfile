# =============================================================================
# Mental Health Session Summary RAG — Dockerfile
# =============================================================================
# Runs the FastAPI server for the RAG pipeline that ingests therapy session
# transcripts, stores them in a ChromaDB vector database, and generates
# structured clinical summaries with risk assessment and similar-session
# retrieval.
#
# Build:  docker build -t mental-health-rag .
# Run:    docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... mental-health-rag
# =============================================================================

# --- Stage 1: Base image ---
# Use Python 3.11 slim for a smaller image footprint while retaining
# the libraries needed for ChromaDB, FastAPI, and LangChain.
FROM python:3.11-slim AS base

# --- Stage 2: Set working directory ---
WORKDIR /app

# --- Stage 3: Install system dependencies ---
# ChromaDB and some LangChain dependencies require build tools and
# sqlite3 development headers for native extensions.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Stage 4: Install Python dependencies ---
# Copy requirements.txt first to leverage Docker layer caching.
# Dependencies change less frequently than source code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 5: Copy application source code ---
# Copy source, data (transcripts), and output directories.
COPY src/ src/
COPY data/ data/
COPY outputs/ outputs/

# --- Stage 6: Create non-root user for security ---
# Running as root inside a container is a security risk.
# The appuser needs write access to the ChromaDB persist directory
# and the outputs directory.
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# --- Stage 7: Expose the FastAPI port ---
# The API server listens on port 8000 by default.
EXPOSE 8000

# --- Stage 8: Configure environment ---
# OPENAI_API_KEY is passed at runtime via -e flag or docker-compose.
# Without it, the pipeline falls back to ChromaDB default embeddings
# and template-based summaries.
ENV PYTHONUNBUFFERED=1

# --- Stage 9: Launch the FastAPI server ---
# Use uvicorn to serve the FastAPI app. Bind to 0.0.0.0 so the
# server is accessible from outside the container.
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
