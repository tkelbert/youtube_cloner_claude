#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration module for YouTube to MP3 downloader and voice cloning system.

This module centralizes all configuration settings, paths, and constants used
throughout the application to facilitate easier maintenance and adjustments.
"""

import os
import tempfile
from pathlib import Path

# Base directories
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = Path(os.path.join(BASE_DIR, "output"))
TEMP_DIR = Path(tempfile.gettempdir()) / "ytmp3voiceclone"

# Create necessary directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# YouTube downloader settings
YOUTUBE_DOWNLOAD_RETRIES = 3
YOUTUBE_TIMEOUT = 30  # seconds
YOUTUBE_MAX_FILESIZE = 500 * 1024 * 1024  # 500 MB

# Audio processing settings
DEFAULT_AUDIO_FORMAT = "mp3"
DEFAULT_BITRATE = "192k"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 2
SUPPORTED_AUDIO_FORMATS = ["mp3", "wav", "ogg", "m4a", "flac"]
FFMPEG_TIMEOUT = 600  # 10 minutes

# Voice cloning settings
VOICE_MODEL_DIR = Path(os.path.join(OUTPUT_DIR, "voice_models"))
VOICE_MODEL_DIR.mkdir(exist_ok=True)
VOICE_SAMPLE_MIN_DURATION = 10  # seconds
VOICE_SAMPLE_MAX_DURATION = 600  # 10 minutes
VOICE_SYNTHESIS_MAX_LENGTH = 1000  # characters

# Logging settings
LOG_DIR = Path(os.path.join(BASE_DIR, "logs"))
LOG_DIR.mkdir(exist_ok=True)
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = Path(os.path.join(LOG_DIR, "ytmp3voiceclone.log"))

# CLI settings
CLI_PROGRESS_BAR_WIDTH = 80
CLI_PROGRESS_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
