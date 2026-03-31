"""
Centralized logging utility for the Lakshya AI Multi-Agent System.

This module provides logging configuration with dual output (console + file),
rotating file handlers, and helper functions for agent lifecycle logging.

Requirements: 8.1, 8.2, 8.3, 8.5, 8.6
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_file: str = "logs/lakshya_ai.log"
) -> logging.Logger:
    """
    Configure and return a logger instance with dual output.
    
    The logger writes to both console (INFO+) and file (DEBUG+) with
    structured formatting including timestamps, levels, and module names.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        log_level: Minimum log level for console output (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (default: logs/lakshya_ai.log)
    
    Returns:
        Configured logger instance
    
    Requirements:
        - 8.1: Centralized logging utility
        - 8.5: Configurable log levels
        - 8.6: Dual output (console + file)
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times if logger already exists
    if logger.handlers:
        return logger
    
    # Set base level to DEBUG so handlers can filter
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (INFO+)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation (DEBUG+)
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Rotating file handler: 10MB max, 5 backups
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def log_agent_start(
    logger: logging.Logger,
    agent_name: str,
    ticker: str,
    additional_context: Optional[dict] = None
) -> None:
    """
    Log the start of an agent's execution.
    
    Args:
        logger: Logger instance to use
        agent_name: Name of the agent starting execution
        ticker: Stock ticker being analyzed
        additional_context: Optional dictionary with additional context
    
    Requirements:
        - 8.2: Log agent start with name and input parameters
    """
    context_str = ""
    if additional_context:
        context_parts = [f"{k}={v}" for k, v in additional_context.items()]
        context_str = f" | Context: {', '.join(context_parts)}"
    
    logger.info(
        f"Agent '{agent_name}' started | Ticker: {ticker}{context_str}"
    )


def log_agent_complete(
    logger: logging.Logger,
    agent_name: str,
    execution_time: float,
    success: bool = True,
    additional_info: Optional[str] = None
) -> None:
    """
    Log the completion of an agent's execution with timing information.
    
    Args:
        logger: Logger instance to use
        agent_name: Name of the agent that completed
        execution_time: Execution time in seconds
        success: Whether the agent completed successfully
        additional_info: Optional additional information to log
    
    Requirements:
        - 8.3: Log agent completion with name and execution time
    """
    status = "completed successfully" if success else "failed"
    info_str = f" | {additional_info}" if additional_info else ""
    
    logger.info(
        f"Agent '{agent_name}' {status} | "
        f"Execution time: {execution_time:.2f}s{info_str}"
    )


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: str,
    additional_data: Optional[dict] = None
) -> None:
    """
    Log an error with full context and stack trace.
    
    Args:
        logger: Logger instance to use
        error: Exception that occurred
        context: Description of where/when the error occurred
        additional_data: Optional dictionary with additional context data
    
    Requirements:
        - 8.4: Log errors with full stack trace and context
    """
    error_msg = f"Error in {context}: {str(error)}"
    
    if additional_data:
        data_parts = [f"{k}={v}" for k, v in additional_data.items()]
        error_msg += f" | Context: {', '.join(data_parts)}"
    
    logger.error(error_msg, exc_info=True)


def log_system_event(
    logger: logging.Logger,
    event_type: str,
    message: str,
    level: str = "INFO"
) -> None:
    """
    Log a system-level event.
    
    Args:
        logger: Logger instance to use
        event_type: Type of event (e.g., "STARTUP", "SHUTDOWN", "CONFIG")
        message: Event message
        level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.log(log_level, f"[{event_type}] {message}")


def get_timestamp() -> str:
    """
    Get current timestamp in ISO 8601 format.
    
    Returns:
        ISO 8601 formatted timestamp string
    """
    from datetime import timezone
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
