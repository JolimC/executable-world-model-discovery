"""Structured JSON diagnostics kept outside deterministic event payloads."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "structured_fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def configure_logging(level: str) -> None:
    """Install one process-level JSON log handler."""

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def structured_log(logger: logging.Logger, level: int, message: str, **fields: object) -> None:
    logger.log(level, message, extra={"structured_fields": fields})
