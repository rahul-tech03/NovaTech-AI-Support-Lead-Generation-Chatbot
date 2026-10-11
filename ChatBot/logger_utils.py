"""
Logger utility for NovaTech AI Support Agent.
Provides formatted console and file logging.
"""

import os
import sys
import logging
from typing import Optional

def setup_logger(name: str = "ChatBot", log_level: Optional[str] = None) -> logging.Logger:
    """
    Configure and return a structured logger.
    Logs to stdout and optionally to a log file inside the logs/ directory.
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger

    level_name = log_level or os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File Handler (Optional logs directory)
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    try:
        os.makedirs(logs_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(logs_dir, "chatbot.log"), encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    except Exception:
        pass  # Proceed with console logging if file system is restricted

    logger.propagate = False
    return logger
