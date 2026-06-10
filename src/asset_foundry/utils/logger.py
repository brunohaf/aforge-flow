"""Logging configuration and utilities."""

import logging
from pathlib import Path
from typing import Optional

import structlog


def configure_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = False,
) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(level=numeric_level)

    renderer = (
        structlog.processors.JSONRenderer()
        if json_format
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(
                fmt="%Y-%m-%d %H:%M:%S"
            ),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            numeric_level
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    """Get a logger instance."""
    return structlog.get_logger(name)
