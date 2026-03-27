"""
Tests for the structured logging configuration.

Validates that the JSON formatter produces correct output and that
the logger factory returns properly configured loggers.
"""

import json
import logging

import pytest

from src.logging_config import StructuredJsonFormatter, get_logger


class TestStructuredJsonFormatter:
    """Tests for the JSON log formatter."""

    def test_output_is_valid_json(self):
        formatter = StructuredJsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_required_fields_present(self):
        formatter = StructuredJsonFormatter()
        record = logging.LogRecord(
            name="src.ingest",
            level=logging.WARNING,
            pathname="ingest.py",
            lineno=42,
            msg="something happened",
            args=None,
            exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        assert parsed["level"] == "WARNING"
        assert parsed["logger"] == "src.ingest"
        assert parsed["message"] == "something happened"
        assert "timestamp" in parsed

    def test_extra_fields_included(self):
        formatter = StructuredJsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="risk assessed",
            args=None,
            exc_info=None,
        )
        record.risk_level = "low"
        record.client_id = "CLT-4401"
        parsed = json.loads(formatter.format(record))
        assert parsed["risk_level"] == "low"
        assert parsed["client_id"] == "CLT-4401"

    def test_exception_info_included(self):
        formatter = StructuredJsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="error occurred",
            args=None,
            exc_info=exc_info,
        )
        parsed = json.loads(formatter.format(record))
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]


class TestGetLogger:
    """Tests for the logger factory function."""

    def test_returns_logger_instance(self):
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_matches(self):
        logger = get_logger("src.api")
        assert logger.name == "src.api"

    def test_logger_has_handler(self):
        logger = get_logger("test.handler_check")
        assert len(logger.handlers) > 0

    def test_handler_uses_json_formatter(self):
        logger = get_logger("test.formatter_check")
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, StructuredJsonFormatter)

    def test_logger_level_is_info(self):
        logger = get_logger("test.level_check")
        assert logger.level == logging.INFO

    def test_same_logger_returned_on_repeat_call(self):
        logger1 = get_logger("test.singleton")
        logger2 = get_logger("test.singleton")
        assert logger1 is logger2

    def test_no_duplicate_handlers(self):
        """Calling get_logger twice should not add a second handler."""
        name = "test.no_dupes"
        logger1 = get_logger(name)
        handler_count = len(logger1.handlers)
        logger2 = get_logger(name)
        assert len(logger2.handlers) == handler_count
