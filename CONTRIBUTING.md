# Contributing to Mental Health Session Summary RAG

Thank you for your interest in contributing. This project is a RAG pipeline that ingests therapy session transcripts, stores them in a vector database (ChromaDB), and generates structured clinical summaries with risk assessment via LLM. Contributions that improve retrieval quality, risk assessment accuracy, or API robustness are especially welcome.

## Development Setup

```bash
git clone https://github.com/kavosh-monfared/mental-health-rag.git
cd mental-health-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Optionally add your OpenAI API key: OPENAI_API_KEY=sk-...
```

The pipeline works without an API key using ChromaDB default embeddings and template-based summaries.

## Running Tests

```bash
# Full test suite
python -m pytest tests/ -v

# Specific modules
python -m pytest tests/test_models.py -v
python -m pytest tests/test_ingest.py -v
python -m pytest tests/test_risk_assessment.py -v
python -m pytest tests/test_retriever.py -v
python -m pytest tests/test_api.py -v
```

All tests must pass before submitting a pull request.

## Code Style

- **Type hints** on all function signatures and return types.
- **Pydantic v2 models** for all data structures (`src/models.py`). The same models define LLM output schemas and API contracts.
- **Structured JSON logging** via `src/logging_config.py`. Use the configured logger, not `print()`.
- **Anti-hallucination prompts** in `src/prompts.py`. Every prompt must instruct the LLM to only report information present in the transcript.
- Follow existing patterns -- read the module you are modifying before making changes.

## Submitting Changes

1. Fork the repository and create a feature branch (`git checkout -b feature/your-feature`).
2. Make your changes with tests covering new behavior.
3. Run the full test suite and confirm all tests pass.
4. If your change affects retrieval or summarization, test with `python -m src.main summarize session_001.txt` and verify output quality.
5. Open a pull request against `main` with a clear description of what changed and why.

## Clinical Safety Considerations

This project handles mental health session data, which carries heightened sensitivity. If your change modifies any of the following, take extra care:

- **Risk assessment** (`src/retriever.py`, keyword lists) -- The risk classification (none/low/moderate/high/imminent) follows Columbia Suicide Severity Rating Scale (C-SSRS) principles. Do not lower risk thresholds or remove risk keywords without clinical justification. A missed risk indicator in a therapy session can have life-threatening consequences.
- **Anti-hallucination prompts** (`src/prompts.py`) -- The LLM must never fabricate clinical details. Fields must default to "Not documented in this session" rather than generating plausible content. Do not weaken these guardrails.
- **Retrieval filtering** (`src/retriever.py`) -- Same-client exclusion and metadata filtering exist to prevent cross-contamination of clinical records. Do not bypass these safeguards.
- **Privacy** -- This project uses synthetic data only. Never add real patient data to the repository. Be mindful of 42 CFR Part 2 protections for substance use records and HIPAA requirements for mental health data.

When in doubt, err on the side of conservative risk classification and explicit "not documented" defaults.

## Questions

Open an issue with the `[QUESTION]` prefix if you have questions about the codebase or contribution process.
