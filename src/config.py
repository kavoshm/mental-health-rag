"""
Configuration for the Mental Health Session Summary RAG Pipeline.

Centralizes all configurable parameters: model names, chunk sizes, paths,
and collection settings. Uses environment variables where appropriate,
with sensible defaults for development.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# OpenAI / LLM Configuration
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# ---------------------------------------------------------------------------
# Embedding Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS: int = 1536

# ---------------------------------------------------------------------------
# Text Splitting Configuration
# ---------------------------------------------------------------------------
# Therapy sessions have dialogue structure — we use larger chunks to preserve
# therapist-client exchange context, with custom separators for speaker turns.
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

# Custom separators tuned for therapy transcripts:
# 1. Blank lines (section boundaries, metadata separator)
# 2. Speaker turn boundaries (Therapist/Client lines)
# 3. Standard paragraph/sentence boundaries
CLINICAL_SEPARATORS: list[str] = [
    "\n\n",
    "\nTherapist:",
    "\nClient:",
    "\n",
    ". ",
    " ",
]

# ---------------------------------------------------------------------------
# ChromaDB Configuration
# ---------------------------------------------------------------------------
COLLECTION_NAME: str = "therapy_sessions"
SIMILARITY_METRIC: str = "cosine"  # cosine | l2 | ip

# ---------------------------------------------------------------------------
# Retrieval Configuration
# ---------------------------------------------------------------------------
DEFAULT_TOP_K: int = 5
MMR_DIVERSITY: float = 0.3  # 0.0 = pure relevance, 1.0 = pure diversity

# ---------------------------------------------------------------------------
# Risk Assessment Configuration
# ---------------------------------------------------------------------------
RISK_KEYWORDS: list[str] = [
    "suicidal", "suicide", "self-harm", "cutting", "kill myself",
    "not wanting to be alive", "want to disappear", "overdose",
    "end it", "better off without me", "passive death wishes",
    "harm myself", "no reason to live", "hopeless",
]

RISK_LEVELS: dict[str, str] = {
    "none": "No identified risk factors",
    "low": "Passive ideation without plan or intent",
    "moderate": "Active ideation without plan, or passive with risk factors",
    "high": "Active ideation with plan or access to means",
    "imminent": "Active ideation with plan, intent, and means",
}
