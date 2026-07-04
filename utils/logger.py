# ============================================================
#  utils/logger.py
#  Configures the Python logging module to output through
#  Rich's handler for beautiful, structured console output.
#  All modules call get_logger(__name__) to get their logger.
# ============================================================

import logging
import os
from pathlib import Path

from rich.logging import RichHandler
from rich.console import Console

# Ensure logs/ directory exists
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
ERROR_LOG_PATH = LOGS_DIR / "error.log"

_configured = False


def configure_logging(level: str = "INFO") -> None:
    """
    Sets up the root logger with two handlers:
    1. RichHandler  → coloured, human-readable console output
    2. FileHandler  → full tracebacks written to logs/error.log
    Called once from main.py at startup.
    """
    global _configured
    if _configured:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # --- Rich console handler ---
    rich_handler = RichHandler(
        level=numeric_level,
        rich_tracebacks=True,
        tracebacks_show_locals=False,  # don't leak env vars in tracebacks
        markup=True,
        show_time=True,
        show_path=False,
    )

    # --- File handler (captures WARNING+ with full tracebacks) ---
    file_handler = logging.FileHandler(ERROR_LOG_PATH, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich_handler, file_handler],
        force=True,
    )

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger.  All agents and modules call this.
    Usage:
        from utils.logger import get_logger
        log = get_logger(__name__)
    """
    return logging.getLogger(name)
