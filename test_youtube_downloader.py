#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the YouTube downloader module.

This module contains tests for the functionality of the YouTube downloader.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youtube_downloader import YouTubeDownloader


class TestYouTubeDownloader(unittest.TestCase):
    """Tests for the YouTubeDownloader class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.downloader = YouTubeDownloader(output_dir=self.temp_dir.name)
    
    def tearDown(self):
        """Clean up test environment after each test."""
        self.temp_dir.cleanup()
    
    def test_validate_youtube_url_valid(self):
        """Test URL validation with valid YouTube URLs."""
        valid_urls = [
            'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
            'https://youtu.be/dQw4w9WgXcQ',
            'https://www.youtube.com/embed/dQw4w9WgXcQ',
            'http://www.youtube.com/watch?v=dQw4w9WgXcQ',
            'youtube.com/watch?v=dQw4w9WgXcQ'
        ]
        
        for url in valid_urls:
            self.assertTrue(
                self.downloader.validate_youtube_url(url),
                f"URL validation failed for valid URL: {url}"
            )
    
    def test_validate_youtube_url_invalid(self):
        """Test URL validation with invalid URLs."""
        invalid_urls = [
            'https://www.example.com',
            'https://www.youtube.com/channel/UC38IQsAvIsxxjztdMZQtwHA',
            'not a url',
            'https://www.vimeo.com/123456',
            'https://www.youtube.com/watch?id=dQw4w9WgXcQ'  # Wrong parameter name
        ]
        
        for url in invalid_urls:
            self.assertFalse(
                self.downloader.validate_youtube_url(url),
                f"URL validation failed for invalid URL: {url}"
            )
    
    @patch('yt_dlp.YoutubeDL')
    def test_get_video_info(self, mock_ytdl):
        """Test retrieving video information."""
        # Configure mock
        mock_instance = MagicMock()
        mock_info = {
            'title': 'Test Video',
            'duration': 100,  # 100 seconds
            'uploader': 'Test Uploader'
        }
        mock_instance.extract_info.return_value = mock_info
        mock_ytdl.return_value.__enter__.return_value = mock_instance
        
        # Call method
        url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
        info = self.downloader.get_video_info(url)
        
        # Verify results
        self.assertEqual(info, mock_info)
        mock_instance.extract_info.assert_called_once_with(url, download=False)
    
    @patch('yt_dlp.YoutubeDL')
    def test_get_video_info_invalid_url(self, mock_ytdl):
        """Test retrieving info with invalid URL."""
        url = 'not a url'
        
        with self.assertRaises(ValueError):
            self.downloader.get_video_info(url)
    
    @patch('yt_dlp.YoutubeDL')
    def test_download_audio(self, mock_ytdl):
        """Test downloading audio from a video."""
        # Configure mock
        mock_instance = MagicMock()
        mock_info = {
            'title': 'Test Video',
            'id': 'dQw4w9WgXcQ',
            'ext': 'webm'
        }
        mock_instance.extract_info.return_value = mock_info
        
        # Setup for the prepare_filename method to return a valid path
        test_path = os.path.join(self.temp_dir.name, 'Test Video.webm')
        mock_instance.prepare_filename.return_value = test_path
        
        # Create a dummy file to simulate successful download
        wav_path = os.path.join(self.temp_dir.name, 'Test Video.wav')
        with open(wav_path, 'w') as f:
            f.write('dummy content')
        
        mock_ytdl.return_value.__enter__.return_value = mock_instance
        
        # Call method
        url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
        file_path, title = self.downloader.download_audio(url)
        
        # Verify results
        self.assertEqual(file_path, wav_path)
        self.assertEqual(title, 'Test Video')
        mock_instance.extract_info.assert_called_once_with(url, download=True)
    
    @patch('yt_dlp.YoutubeDL')
    def test_download_audio_invalid_url(self, mock_ytdl):
        """Test downloading with invalid URL."""
        url = 'not a url'
        
        with self.assertRaises(ValueError):
            self.downloader.download_audio(url)


if __name__ == '__main__':
    unittest.main()
