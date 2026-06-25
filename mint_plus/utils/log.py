"""
Logging Utilities
==================

Provides consistent logging across the MINT package.

Usage:
    from mint_plus.utils import get_logger

    logger = get_logger(__name__)
    logger.info("Training started")
    logger.warning("Low memory")
    logger.error("Failed to load data")
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Add handler if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
