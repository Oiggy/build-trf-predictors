"""
LOGGING_CONFIG.PY - Generic logging configuration for predictor scripts

This module provides a simple way to set up logging for any predictor script.

Usage:
    from logging_config import setup_logging
    
    logger = setup_logging(
        log_filename="gammatone.log",
        output_dir=Path("path/to/output"),
        script_name="my_script.py"
    )
    
    logger.info("Processing started")
"""

import logging
import sys
from pathlib import Path


def setup_logging(log_filename, output_dir, script_name=None):
    """
    Setup logging configuration for predictor scripts
    
    Args:
        log_filename: Name of the log file (e.g., "gammatone.log", "wordonsets.log")
        output_dir: Path object for the output directory
        script_name: Optional name of the calling script for the logger
        
    Returns:
        Configured logger instance
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Full path to log file
    log_path = output_dir / log_filename
    
    # Create logger name
    logger_name = script_name if script_name else __name__
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    return logger