import logging
import os
import sys
from datetime import datetime
from typing import Optional

def setup_logging(level: int = logging.INFO, 
                 log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> None:
    """
    Setup centralized logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_string: Custom format string
    """
    
    # Default format
    if format_string is None:
        format_string = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
    
    # Create formatter
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"Warning: Failed to setup file logging: {e}")
    
    # Set levels for specific loggers to reduce noise
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

def setup_streamlit_logging() -> None:
    """Setup logging specifically for Streamlit applications"""
    
    # Streamlit apps should use INFO level by default
    setup_logging(
        level=logging.INFO,
        format_string='[%(levelname)s] %(name)s: %(message)s'
    )
    
    # Suppress some noisy loggers in web context
    logging.getLogger('streamlit').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

def setup_cli_logging(verbose: bool = False) -> None:
    """Setup logging specifically for CLI applications"""
    
    level = logging.DEBUG if verbose else logging.INFO
    
    setup_logging(
        level=level,
        format_string='[%(levelname)s] %(message)s'
    )

class LoggerMixin:
    """
    Mixin class to add logging capability to any class
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        return self._logger

def log_function_call(func):
    """
    Decorator to log function calls
    
    Usage:
        @log_function_call
        def my_function(arg1, arg2):
            pass
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    
    return wrapper

def log_execution_time(func):
    """
    Decorator to log function execution time
    
    Usage:
        @log_execution_time
        def slow_function():
            pass
    """
    def wrapper(*args, **kwargs):
        import time
        
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.3f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f} seconds: {e}")
            raise
    
    return wrapper

class ProgressLogger:
    """
    Helper class for logging progress of long-running operations
    """
    
    def __init__(self, total: int, name: str = "Operation", logger: Optional[logging.Logger] = None):
        self.total = total
        self.current = 0
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.last_percentage = 0
    
    def update(self, increment: int = 1) -> None:
        """Update progress"""
        self.current += increment
        percentage = int((self.current / self.total) * 100)
        
        # Log every 10% progress
        if percentage >= self.last_percentage + 10:
            self.logger.info(f"{self.name}: {percentage}% ({self.current}/{self.total})")
            self.last_percentage = percentage
    
    def complete(self) -> None:
        """Mark operation as complete"""
        self.logger.info(f"{self.name}: Completed ({self.total}/{self.total})")

