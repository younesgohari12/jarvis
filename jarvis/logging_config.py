from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(path: Path, *, debug: bool = False) -> logging.Logger:
    logger = logging.getLogger("jarvis")
    if logger.handlers:
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        return logger

    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False
    try:
        handler: logging.Handler = RotatingFileHandler(
            path,
            maxBytes=262_144,
            backupCount=1,
            encoding="utf-8",
        )
    except OSError:
        handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    try:
        error_handler = RotatingFileHandler(
            path.with_name("errors.log"), maxBytes=262_144,
            backupCount=1, encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(error_handler)
    except OSError:
        pass
    if debug:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(console)
    return logger
