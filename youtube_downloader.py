#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube Downloader module for extracting audio from YouTube videos.

This module provides functionality to download audio from YouTube videos
using the yt-dlp library, with robust error handling and retry mechanisms.
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import yt_dlp

from config import (TEMP_DIR, YOUTUBE_DOWNLOAD_RETRIES, YOUTUBE_MAX_FILESIZE,
                   YOUTUBE_TIMEOUT)

# Configure module logger
logger = logging.getLogger(__name__)


class YouTubeDownloader:
    """
    Class for downloading audio from YouTube videos.
    
    Handles validation, downloading, and error management for YouTube audio extraction.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the YouTube downloader with configurable output directory.
        
        Args:
            output_dir: Directory where downloaded files will be saved.
                        If None, uses the temporary directory from config.
        """
        self.output_dir = Path(output_dir) if output_dir else TEMP_DIR
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Configure yt-dlp options
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'restrictfilenames': True,
            'noplaylist': True,
            'nocheckcertificate': True,
            'ignoreerrors': False,
            'logtostderr': False,
            'quiet': False,
            'no_warnings': False,
            'default_search': 'auto',
            'source_address': '0.0.0.0',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',  # Extract as WAV for highest quality
                'preferredquality': '192',
            }],
        }
    
    def validate_youtube_url(self, url: str) -> bool:
        """
        Validate if the given URL is a valid YouTube URL.
        
        Args:
            url: The URL to validate
            
        Returns:
            bool: True if valid YouTube URL, False otherwise
        """
        youtube_regex = (
            r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
            r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
        
        match = re.match(youtube_regex, url)
        return match is not None
    
    def get_video_info(self, url: str) -> Dict:
        """
        Retrieve metadata about a YouTube video.
        
        Args:
            url: YouTube URL to get information about
            
        Returns:
            Dict: Video metadata including title, duration, etc.
            
        Raises:
            ValueError: If URL is invalid or video info cannot be retrieved
        """
        if not self.validate_youtube_url(url):
            raise ValueError(f"Invalid YouTube URL: {url}")
        
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                if info.get('duration', 0) * 128 * 1000 / 8 > YOUTUBE_MAX_FILESIZE:
                    raise ValueError(f"Video is too large (exceeds {YOUTUBE_MAX_FILESIZE/1024/1024:.1f}MB limit)")
                return info
        except yt_dlp.utils.DownloadError as e:
            raise ValueError(f"Failed to retrieve video info: {str(e)}")
    
    def download_audio(self, url: str) -> Tuple[str, str]:
        """
        Download audio from a YouTube video.
        
        Args:
            url: YouTube URL to download audio from
            
        Returns:
            Tuple[str, str]: (File path to downloaded audio, Title of the video)
            
        Raises:
            ValueError: For invalid URLs, unavailable videos
            RuntimeError: For download failures after retries
        """
        if not self.validate_youtube_url(url):
            raise ValueError(f"Invalid YouTube URL: {url}")
        
        logger.info(f"Downloading audio from: {url}")
        
        # Try to download with retries
        for attempt in range(1, YOUTUBE_DOWNLOAD_RETRIES + 1):
            try:
                # Create a new YoutubeDL instance for each attempt
                with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                    # Set timeout
                    ydl.params['socket_timeout'] = YOUTUBE_TIMEOUT
                    
                    # Extract info and download
                    info = ydl.extract_info(url, download=True)
                    
                    # Get the downloaded file path
                    if 'entries' in info:
                        # Playlist not supported, but handle gracefully
                        info = info['entries'][0]
                        logger.warning("Playlist detected, only downloading the first video")
                    
                    title = info.get('title', 'unknown_video')
                    
                    # Find the downloaded file
                    filename = ydl.prepare_filename(info)
                    base_path = os.path.splitext(filename)[0]
                    
                    # Check for the WAV file (from the postprocessor)
                    wav_path = f"{base_path}.wav"
                    if os.path.exists(wav_path):
                        logger.info(f"Successfully downloaded audio: {wav_path}")
                        return wav_path, title
                    
                    # If WAV wasn't created for some reason, look for any audio file
                    for ext in ['wav', 'm4a', 'mp3', 'ogg', 'opus']:
                        audio_path = f"{base_path}.{ext}"
                        if os.path.exists(audio_path):
                            logger.info(f"Successfully downloaded audio: {audio_path}")
                            return audio_path, title
                    
                    raise FileNotFoundError("Downloaded file not found")
                    
            except yt_dlp.utils.DownloadError as e:
                logger.warning(f"Download attempt {attempt} failed: {str(e)}")
                if attempt == YOUTUBE_DOWNLOAD_RETRIES:
                    raise RuntimeError(f"Failed to download after {YOUTUBE_DOWNLOAD_RETRIES} attempts: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
            
            except Exception as e:
                logger.error(f"Unexpected error during download: {str(e)}")
                if attempt == YOUTUBE_DOWNLOAD_RETRIES:
                    raise RuntimeError(f"Failed to download after {YOUTUBE_DOWNLOAD_RETRIES} attempts: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        # This should never be reached due to the exceptions above, but added for completeness
        raise RuntimeError("Failed to download audio for unknown reasons")
