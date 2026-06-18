"""
Structured logging configuration for the ML system.

Uses structlog for JSON-formatted, context-rich logging
that's easy to parse in production monitoring tools.
"""

import logging
import sys
from pathlib import Path

import structlog
import yaml


def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def setup_logging(level: str = "INFO") -> None:
    """
    Configure structured logging for the entire application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    config = load_config()
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", level).upper(), logging.INFO)
    log_dir = Path(log_config.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a named logger instance.
    
    Args:
        name: Logger name (typically module name)
        
    Returns:
        Configured structlog bound logger
    """
    return structlog.get_logger(name)
