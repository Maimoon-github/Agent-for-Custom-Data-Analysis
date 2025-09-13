"""
Logging utility for the RAG Agent system
"""

import sys
from pathlib import Path
from loguru import logger
from config.settings import config

def setup_logging():
    """Configure logging for the RAG system"""
    # Remove default handler
    logger.remove()
    
    # Ensure log directory exists
    log_path = Path(config.logging.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Console handler with colored output
    logger.add(
        sys.stderr,
        level=config.logging.level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File handler for persistent logging
    logger.add(
        config.logging.file,
        level=config.logging.level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    return logger

# Initialize logger
log = setup_logging()

def get_logger(name: str = None):
    """Get a logger instance with optional name"""
    if name:
        return logger.bind(name=name)
    return logger
