"""
Structured Logging Configuration for the Mental Health RAG Pipeline.

Provides JSON-formatted structured logging for audit trails, which is
critical for healthcare applications handling sensitive mental health data.
All pipeline events (ingestion, retrieval, summarization, risk assessment,
API requests) are logged with structured context fields.

Usage:
    from src.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("event description", extra={"key": "value"})
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects.

    Each log entry contains:
    - timestamp (ISO 8601 UTC)
    - level
    - logger name
    - message
    - Any extra fields passed via the `extra` dict
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include any extra fields that are not standard LogRecord attributes
        standard_attrs = {
            "name", "msg", "args", "created", "relativeCreated",
            "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "filename", "module", "pathname", "thread", "threadName",
            "processName", "process", "message", "levelname", "levelno",
            "msecs", "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                log_entry[key] = value

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with structured JSON output.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(StructuredJsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger
