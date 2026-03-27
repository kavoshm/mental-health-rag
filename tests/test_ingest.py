"""
Tests for the ingestion pipeline: metadata extraction, transcript body
extraction, and dialogue-aware chunking.
"""

import pytest

# These modules require langchain and chromadb which may not be installed
# in all environments. Skip the entire module gracefully if unavailable.
langchain = pytest.importorskip("langchain", reason="langchain not installed")

from src.ingest import (
    create_therapy_splitter,
    extract_metadata,
    extract_transcript_body,
    split_transcript,
)
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


# =========================================================================
# Metadata Extraction
# =========================================================================

class TestExtractMetadata:
    """Tests for extract_metadata regex parsing."""

    def test_extracts_all_fields(self, sample_full_transcript):
        meta = extract_metadata(sample_full_transcript, "session_001.txt")
        assert meta["source_file"] == "session_001.txt"
        assert meta["session_date"] == "2025-01-06"
        assert meta["session_number"] == 1
        assert meta["client_id"] == "CLT-4401"
        assert meta["clinician"] == "Dr. Navarro, PsyD"
        assert meta["session_type"] == "Individual Therapy — Intake Assessment"
        assert meta["duration"] == "60 minutes"

    def test_session_number_is_int(self, sample_full_transcript):
        meta = extract_metadata(sample_full_transcript, "test.txt")
        assert isinstance(meta["session_number"], int)

    def test_missing_fields_are_absent(self):
        content = "No metadata here, just plain text dialogue."
        meta = extract_metadata(content, "bare.txt")
        assert meta["source_file"] == "bare.txt"
        assert "session_date" not in meta
        assert "client_id" not in meta
        assert "clinician" not in meta

    def test_partial_metadata(self):
        content = "Date: 2025-03-15\nClient ID: CLT-9999\nSome other stuff."
        meta = extract_metadata(content, "partial.txt")
        assert meta["session_date"] == "2025-03-15"
        assert meta["client_id"] == "CLT-9999"
        assert "session_number" not in meta
        assert "clinician" not in meta

    def test_date_format_regex(self):
        """Date extraction requires YYYY-MM-DD format."""
        valid = "Date: 2025-01-06"
        invalid = "Date: January 6, 2025"
        assert extract_metadata(valid, "f.txt").get("session_date") == "2025-01-06"
        assert "session_date" not in extract_metadata(invalid, "f.txt")

    def test_client_id_pattern(self):
        """Client ID must match word characters and hyphens."""
        content = "Client ID: CLT-4401"
        meta = extract_metadata(content, "f.txt")
        assert meta["client_id"] == "CLT-4401"

    def test_multiline_clinician(self):
        """Clinician field should stop at newline."""
        content = "Clinician: Dr. Chen, PhD\nSession Type: Individual"
        meta = extract_metadata(content, "f.txt")
        assert meta["clinician"] == "Dr. Chen, PhD"


# =========================================================================
# Transcript Body Extraction
# =========================================================================

class TestExtractTranscriptBody:
    """Tests for extract_transcript_body splitting."""

    def test_removes_metadata_header(self, sample_full_transcript):
        body = extract_transcript_body(sample_full_transcript)
        assert "SESSION METADATA" not in body
        assert "Date: 2025-01-06" not in body
        assert body.startswith("Therapist:")

    def test_removes_transcript_label(self, sample_full_transcript):
        body = extract_transcript_body(sample_full_transcript)
        assert not body.startswith("TRANSCRIPT")

    def test_no_separator_returns_full_content(self):
        content = "Just a transcript without any separator."
        assert extract_transcript_body(content) == content

    def test_preserves_dialogue(self, sample_full_transcript):
        body = extract_transcript_body(sample_full_transcript)
        assert "Therapist:" in body
        assert "Client:" in body


# =========================================================================
# Text Splitting / Chunking
# =========================================================================

class TestChunking:
    """Tests for dialogue-aware text splitting."""

    def test_splitter_uses_config_values(self):
        splitter = create_therapy_splitter()
        assert splitter._chunk_size == CHUNK_SIZE
        assert splitter._chunk_overlap == CHUNK_OVERLAP

    def test_short_text_produces_single_chunk(self):
        short_text = "Therapist: Hello.\n\nClient: Hi."
        metadata = {"source_file": "test.txt"}
        docs = split_transcript(short_text, metadata)
        assert len(docs) == 1
        assert docs[0].metadata["chunk_index"] == 0
        assert docs[0].metadata["total_chunks"] == 1

    def test_long_text_produces_multiple_chunks(self, sample_transcript_body):
        # Repeat the transcript body to ensure it exceeds CHUNK_SIZE
        long_text = sample_transcript_body * 10
        metadata = {"source_file": "long_test.txt"}
        docs = split_transcript(long_text, metadata)
        assert len(docs) > 1

    def test_chunk_metadata_is_propagated(self, sample_transcript_body):
        metadata = {
            "source_file": "session_001.txt",
            "client_id": "CLT-4401",
            "session_date": "2025-01-06",
        }
        docs = split_transcript(sample_transcript_body * 5, metadata)
        for doc in docs:
            assert doc.metadata["source_file"] == "session_001.txt"
            assert doc.metadata["client_id"] == "CLT-4401"
            assert "chunk_index" in doc.metadata
            assert "total_chunks" in doc.metadata
            assert "chunk_length" in doc.metadata

    def test_chunk_indices_are_sequential(self, sample_transcript_body):
        long_text = sample_transcript_body * 10
        metadata = {"source_file": "test.txt"}
        docs = split_transcript(long_text, metadata)
        indices = [d.metadata["chunk_index"] for d in docs]
        assert indices == list(range(len(docs)))

    def test_chunks_have_content(self, sample_transcript_body):
        long_text = sample_transcript_body * 5
        metadata = {"source_file": "test.txt"}
        docs = split_transcript(long_text, metadata)
        for doc in docs:
            assert len(doc.page_content.strip()) > 0

    def test_speaker_turns_preserved_in_chunks(self):
        """Verify that speaker turn boundaries are preferred split points."""
        # Build text where a natural split would occur at a speaker turn
        text = (
            "Therapist: " + "A" * 500 + "\n\n"
            "Client: " + "B" * 500 + "\n\n"
            "Therapist: " + "C" * 500
        )
        metadata = {"source_file": "test.txt"}
        docs = split_transcript(text, metadata)
        # At least some chunks should start with a speaker label
        starts_with_speaker = any(
            d.page_content.strip().startswith("Therapist:")
            or d.page_content.strip().startswith("Client:")
            for d in docs
        )
        assert starts_with_speaker
