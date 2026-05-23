"""Centralized logging for oil_bot."""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.

    Usage:
        from oil_bot.utils.logging import get_logger
        logger = get_logger(__name__)
        logger.info("message")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger