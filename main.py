#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for YouTube to MP3 downloader and voice cloning application.

This module initializes the application and starts the CLI interface.
"""

import logging
import sys
from pathlib import Path

from cli import ApplicationCLI
from config import LOG_DIR, LOG_FILE, LOG_FORMAT, LOG_LEVEL


def setup_application() -> None:
    """
    Perform initial application setup.
    
    Creates necessary directories and configures logging.
    """
    # Create log directory if it doesn't exist
    LOG_DIR.mkdir(exist_ok=True, parents=True)


def main() -> int:
    """
    Main entry point for the application.
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    # Setup application
    setup_application()
    
    # Initialize CLI
    cli = ApplicationCLI()
    
    # Run application
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
