"""
Logging configuration for Max Discord Bot.

This module provides a unified logging system with different verbosity levels
that can be controlled via command-line arguments.
"""

import logging
import sys
from typing import Optional

# Define logger levels
CRITICAL = logging.CRITICAL  # 50
ERROR = logging.ERROR        # 40
WARNING = logging.WARNING    # 30
INFO = logging.INFO          # 20
DEBUG = logging.DEBUG        # 10
TRACE = 5                    # Custom level below DEBUG

# Add custom TRACE level
logging.addLevelName(TRACE, "TRACE")

# Configure the root logger
logger = logging.getLogger("max_bot")

# Default format for log messages
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """
    Configure the logging system based on verbosity level.
    
    Args:
        verbose: Whether to enable detailed (DEBUG/TRACE) logging
        log_file: Optional file path to save logs to
    """
    # Set the base log level
    level = DEBUG if verbose else INFO
    
    # Configure the logger
    logger.setLevel(level)
    
    # Create console handler with appropriate level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console_handler.setFormatter(formatter)
    
    # Add the console handler to the logger
    logger.handlers = []  # Clear existing handlers
    logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized with {'verbose' if verbose else 'standard'} level")

def trace(self, message, *args, **kwargs):
    """Add trace log level (more detailed than debug)"""
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)

# Add trace method to Logger class
logging.Logger.trace = trace

# Create module-specific loggers
bot_logger = logging.getLogger("max_bot.bot")
handler_logger = logging.getLogger("max_bot.handler")
router_logger = logging.getLogger("max_bot.router")
rewriter_logger = logging.getLogger("max_bot.rewriter")
llm_logger = logging.getLogger("max_bot.llm")
history_logger = logging.getLogger("max_bot.history") 