"""
Shared fixtures for Mental Health RAG test suite.

Provides reusable transcript text, metadata, and configuration overrides
so individual test modules stay focused on their assertions.
"""

import pytest


# ---------------------------------------------------------------------------
# Sample transcript fixtures
# ---------------------------------------------------------------------------

SAMPLE_METADATA_HEADER = """\
SESSION METADATA
Date: 2025-01-06
Session Number: 1
Client ID: CLT-4401
Clinician: Dr. Navarro, PsyD
Session Type: Individual Therapy — Intake Assessment
Duration: 60 minutes
"""

SAMPLE_TRANSCRIPT_BODY = """\
Therapist: Thank you for coming in today. Can you tell me what brought you here?

Client: I've been feeling really down for about four months. I don't enjoy anything anymore. I used to love fishing but now I just sit on the couch after work.

Therapist: How long has this been going on?

Client: About four months. It started after I got passed over for a promotion.

Therapist: Have you had any thoughts of harming yourself or not wanting to be alive?

Client: I wouldn't do anything. I have two kids. But sometimes I think they'd be better off without me. I don't have a plan or anything.

Therapist: Do you have access to firearms at home?

Client: I have a hunting rifle in a safe.

Therapist: How about alcohol or drug use?

Client: I've been drinking more than usual. Maybe four or five beers a night, up from one or two.

Therapist: Your PHQ-9 is 19 which indicates moderately severe depression. I'd like to recommend weekly therapy sessions using behavioral activation and cognitive therapy.

Client: Yeah, I think I need the help. Let's do it.
"""


@pytest.fixture
def sample_full_transcript():
    """A complete transcript with metadata header and dialogue body."""
    return SAMPLE_METADATA_HEADER + "\n---\n\nTRANSCRIPT\n\n" + SAMPLE_TRANSCRIPT_BODY


@pytest.fixture
def sample_transcript_body():
    """Just the dialogue portion of a transcript."""
    return SAMPLE_TRANSCRIPT_BODY


@pytest.fixture
def sample_metadata_header():
    """Just the metadata header."""
    return SAMPLE_METADATA_HEADER


@pytest.fixture
def minimal_transcript():
    """A minimal transcript that satisfies the TranscriptInput min_length=100 constraint."""
    return (
        "SESSION METADATA\nDate: 2025-03-01\nClient ID: CLT-9999\n---\n\n"
        "Therapist: How are you feeling today?\n\n"
        "Client: I am feeling okay. Things have been stable this week. "
        "No major concerns to report."
    )


@pytest.fixture
def risk_transcript_passive():
    """Transcript with passive suicidal ideation (should trigger low risk)."""
    return (
        "SESSION METADATA\nDate: 2025-02-01\nClient ID: CLT-0001\n---\n\n"
        "Therapist: Any thoughts of self-harm?\n\n"
        "Client: Sometimes I feel like they would be better off without me. "
        "I have passive death wishes but no plan to act on them.\n\n"
        "Therapist: Do you have family support?\n\n"
        "Client: Yes, my children and wife are supportive."
    )


@pytest.fixture
def risk_transcript_active():
    """Transcript with active suicidal ideation (should trigger moderate risk)."""
    return (
        "SESSION METADATA\nDate: 2025-02-01\nClient ID: CLT-0002\n---\n\n"
        "Therapist: Tell me about your thoughts this week.\n\n"
        "Client: I have been having suicidal ideation. I think about ways to "
        "kill myself but I have not made a specific plan.\n\n"
        "Therapist: That is very important to address. Let's discuss safety planning."
    )


@pytest.fixture
def risk_transcript_denied():
    """Transcript where risk is discussed but explicitly denied."""
    return (
        "SESSION METADATA\nDate: 2025-02-01\nClient ID: CLT-0003\n---\n\n"
        "Therapist: Any thoughts of self-harm or suicide?\n\n"
        "Client: No, I denies suicidal ideation completely. No self-harm thoughts at all.\n\n"
        "Therapist: Good. Let's continue with our regular session."
    )


@pytest.fixture
def risk_transcript_none():
    """Transcript with no risk indicators at all."""
    return (
        "SESSION METADATA\nDate: 2025-02-01\nClient ID: CLT-0004\n---\n\n"
        "Therapist: How has your week been?\n\n"
        "Client: Great actually. I went hiking and spent time with friends. "
        "The breathing exercises really helped with my anxiety at work.\n\n"
        "Therapist: Wonderful progress. Let's keep building on that."
    )
