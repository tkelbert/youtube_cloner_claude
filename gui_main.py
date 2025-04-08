#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI entry point for YouTube to MP3 downloader and voice cloning application.

This module initializes the GUI application.
"""

import logging
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from config import LOG_DIR, LOG_FILE, LOG_FORMAT, LOG_LEVEL
from gui import MainWindow


def setup_logging():
    """
    Configure logging for the application.
    """
    # Create log directory if it doesn't exist
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )


def main():
    """
    Main entry point for the GUI application.
    """
    # Set up logging
    setup_logging()
    
    # Create the Qt Application
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Use Fusion style as base
    
    # Set application metadata
    app.setApplicationName("YouTube to MP3 & Voice Cloning")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Example Corp")
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    # Start the event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
