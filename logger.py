"""
logger.py - Centralized Logging Module
=========================================
Provides a unified logging facility for all project modules.
Logs to both console and file with configurable levels.
"""

import os
import logging
from config import get


def setup_logger(name=None):
    """
    Set up and return a logger with console and file handlers.

    Args:
        name (str): Logger name (typically __name__ of calling module).

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_level = get("logging.level", "INFO")
    log_file = get("logging.log_file", "logs/vm_migration.log")
    console_output = get("logging.console_output", True)
    log_format = get("logging.format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    logger = logging.getLogger(name or "vm_migration")

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    formatter = logging.Formatter(log_format)

    # File handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
