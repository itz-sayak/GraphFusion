"""Structured logging via structlog (console or JSON renderer)."""
from __future__ import annotations

import logging
import sys

import structlog

_CONFIGURED = False


def configure_logging(level: str = "INFO", json: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=getattr(logging, level.upper(), logging.INFO))
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    renderer = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer(colors=False)
    structlog.configure(
        processors=[*processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), logging.INFO)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str = "dfg"):
    if not _CONFIGURED:
        from backend.config import get_settings

        s = get_settings()
        configure_logging(s.dfg_log_level, s.dfg_log_json)
    return structlog.get_logger(name)
