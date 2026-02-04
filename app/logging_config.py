"""
Logging Configuration Module

Sets up application-wide logging using loguru.
Provides consistent log formatting and file output.
"""

import sys
from loguru import logger

from app.config import LOG_LEVEL, LOG_FILE


def setup_logging():
    """Configure application logging."""
    # Remove default handler
    logger.remove()
    
    # Console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=LOG_LEVEL,
        colorize=True
    )
    
    # File handler
    logger.add(
        LOG_FILE,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip"
    )
    
    logger.info("Logging configured successfully")
    return logger


# Initialize logging when module is imported
setup_logging()
