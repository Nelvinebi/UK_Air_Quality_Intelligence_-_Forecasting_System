"""Central logging configuration for project command-line workflows."""

import logging
import os

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_LOG_LEVEL = "INFO"


def configure_logging(level: str | int | None = None) -> None:
    """Configure consistent application logging for CLI entrypoints."""
    resolved_level = level or os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)

    if isinstance(resolved_level, str):
        resolved_level = resolved_level.upper()

    logging.basicConfig(
        level=resolved_level,
        format=LOG_FORMAT,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )