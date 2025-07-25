import logging
import sys
import os
import warnings
from typing import Optional
from pathlib import Path


def suppress_verbose_warnings():
    """Suppress verbose warnings that clutter the terminal"""
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Suppress specific warning categories
    warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
    warnings.filterwarnings("ignore", message=".*streamlit.*")
    warnings.filterwarnings("ignore", message=".*KerasTensor.*")
    warnings.filterwarnings("ignore", message=".*inference_feedback_manager.*")

    # Set specific loggers to WARNING level to reduce noise
    logging.getLogger("streamlit").setLevel(logging.WARNING)
    logging.getLogger("streamlit.runtime").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path

    Returns:
        Configured logger
    """
    # Suppress verbose warnings first
    suppress_verbose_warnings()

    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    # Configure logging format with cleaner output
    log_format = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Configure handlers
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True
    )

    # Get logger for the calling module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at {log_level} level")

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name. If None, uses the calling module's name.

    Returns:
        Logger instance
    """
    if name is None:
        # Get the caller's module name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return logging.getLogger(name)