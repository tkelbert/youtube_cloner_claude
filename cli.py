#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line interface module for YouTube to MP3/voice cloning application.

This module provides a user-friendly CLI for interacting with the application,
coordinating the other modules, and handling user input.
"""

import argparse
import logging
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union

from tqdm import tqdm

from audio_processor import AudioProcessor
from config import (CLI_PROGRESS_BAR_FORMAT, CLI_PROGRESS_BAR_WIDTH, LOG_FILE,
                   LOG_FORMAT, LOG_LEVEL, OUTPUT_DIR)
from voice_cloner import VoiceCloner
from youtube_downloader import YouTubeDownloader

# Configure module logger
logger = logging.getLogger(__name__)


class CliMode(Enum):
    """Enum for different CLI operation modes."""
    DOWNLOAD_ONLY = "download"
    DOWNLOAD_AND_CONVERT = "convert"
    VOICE_CLONE = "clone"
    VOICE_SYNTHESIS = "synthesize"
    LIST_VOICES = "list-voices"


class ApplicationCLI:
    """
    Command-line interface for the YouTube to MP3/voice cloning application.
    
    Handles user interaction, command-line arguments, and coordinates the
    different modules of the application.
    """

    def __init__(self):
        """Initialize the CLI interface and component modules."""
        self._setup_logging()
        
        # Initialize components
        self.youtube_downloader = YouTubeDownloader()
        self.audio_processor = AudioProcessor()
        self.voice_cloner = None  # Initialize only when needed
        
        logger.info("Application initialized")
    
    def _setup_logging(self):
        """
        Set up logging for the application.
        
        Configures console and file logging with appropriate levels.
        """
        # Create a logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, LOG_LEVEL))
        
        # Remove existing handlers (in case this is called multiple times)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Create file handler
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(getattr(logging, LOG_LEVEL))
        file_formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        
        # Add handlers to logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
    
    def _init_voice_cloner(self) -> None:
        """
        Initialize the voice cloner module when needed.
        
        Lazy initialization to avoid importing heavy dependencies
        if voice cloning is not used.
        
        Raises:
            ImportError: If voice cloning dependencies are not installed
        """
        if self.voice_cloner is None:
            try:
                self.voice_cloner = VoiceCloner()
                # Check system compatibility
                is_compatible, reason = self.voice_cloner.check_system_compatibility()
                if not is_compatible:
                    logger.warning(f"Voice cloning compatibility issue: {reason}")
            except ImportError as e:
                logger.error(f"Failed to initialize voice cloner: {str(e)}")
                raise ImportError(
                    "Voice cloning dependencies not installed. "
                    "Install with: pip install tortoise-tts soundfile"
                ) from e
    
    def _parse_arguments(self) -> argparse.Namespace:
        """
        Parse command-line arguments.
        
        Returns:
            argparse.Namespace: Parsed command-line arguments
        """
        parser = argparse.ArgumentParser(
            description="YouTube to MP3 downloader and voice cloning application",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Download and convert a YouTube video to MP3
  python main.py convert https://www.youtube.com/watch?v=dQw4w9WgXcQ
  
  # Download YouTube audio and create a voice model
  python main.py clone https://www.youtube.com/watch?v=dQw4w9WgXcQ --name "my_voice"
  
  # List available voice models
  python main.py list-voices
  
  # Synthesize speech using a voice model
  python main.py synthesize "Hello, this is my cloned voice." --voice "my_voice"
            """
        )
        
        subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
        
        # Download only mode
        download_parser = subparsers.add_parser(
            "download", 
            help="Download audio from YouTube without converting"
        )
        download_parser.add_argument(
            "url", 
            help="YouTube URL to download audio from"
        )
        download_parser.add_argument(
            "--output-dir", "-o",
            help="Directory to save the downloaded audio",
            default=None
        )
        
        # Download and convert mode
        convert_parser = subparsers.add_parser(
            "convert", 
            help="Download audio from YouTube and convert to MP3"
        )
        convert_parser.add_argument(
            "url", 
            help="YouTube URL to download audio from"
        )
        convert_parser.add_argument(
            "--output-dir", "-o",
            help="Directory to save the converted audio",
            default=None
        )
        convert_parser.add_argument(
            "--format", "-f",
            help="Output audio format (mp3, wav, ogg, flac)",
            default="mp3"
        )
        convert_parser.add_argument(
            "--bitrate", "-b",
            help="Output audio bitrate (e.g., 128k, 192k, 320k)",
            default="192k"
        )
        
        # Voice cloning mode
        clone_parser = subparsers.add_parser(
            "clone", 
            help="Download audio from YouTube and create a voice model"
        )
        clone_parser.add_argument(
            "url", 
            help="YouTube URL to download audio from"
        )
        clone_parser.add_argument(
            "--name", "-n",
            help="Name for the voice model",
            default=None
        )
        clone_parser.add_argument(
            "--output-dir", "-o",
            help="Directory to save intermediate files",
            default=None
        )
        
        # Voice synthesis mode
        synthesize_parser = subparsers.add_parser(
            "synthesize", 
            help="Synthesize speech using a voice model"
        )
        synthesize_parser.add_argument(
            "text", 
            help="Text to synthesize into speech"
        )
        synthesize_parser.add_argument(
            "--voice", "-v",
            help="Name of the voice model to use",
            required=True
        )
        synthesize_parser.add_argument(
            "--output", "-o",
            help="Output file path for synthesized speech",
            default=None
        )
        synthesize_parser.add_argument(
            "--preset", "-p",
            help="Quality preset (ultra_fast, fast, standard, high_quality)",
            choices=["ultra_fast", "fast", "standard", "high_quality"],
            default="standard"
        )
        synthesize_parser.add_argument(
            "--seed", "-s",
            help="Random seed for reproducible synthesis",
            type=int,
            default=0
        )
        
        # List voices mode
        list_voices_parser = subparsers.add_parser(
            "list-voices", 
            help="List available voice models"
        )
        
        # Parse arguments
        args = parser.parse_args()
        
        # Validate mode
        if args.mode is None:
            parser.print_help()
            sys.exit(1)
        
        return args
    
    def download_youtube_audio(
        self, 
        url: str,
        output_dir: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Download audio from a YouTube video.
        
        Args:
            url: YouTube URL to download audio from
            output_dir: Directory to save the downloaded audio
            
        Returns:
            Tuple[str, str]: (File path to downloaded audio, Video title)
            
        Raises:
            ValueError: If download fails
        """
        logger.info(f"Downloading audio from YouTube: {url}")
        print(f"Downloading audio from YouTube: {url}")
        
        try:
            # Show a determinate progress bar if possible
            print("Starting download...")
            
            # Download audio
            audio_path, title = self.youtube_downloader.download_audio(url)
            
            print(f"Download complete: {os.path.basename(audio_path)}")
            return audio_path, title
            
        except Exception as e:
            logger.error(f"Failed to download YouTube audio: {str(e)}")
            print(f"Error: {str(e)}")
            raise
    
    def convert_audio_to_mp3(
        self,
        input_file: str,
        output_format: str = "mp3",
        bitrate: str = "192k",
        output_dir: Optional[str] = None
    ) -> str:
        """
        Convert audio file to MP3 or other format.
        
        Args:
            input_file: Path to input audio file
            output_format: Target audio format
            bitrate: Target audio bitrate
            output_dir: Directory to save converted file
            
        Returns:
            str: Path to converted audio file
            
        Raises:
            ValueError: If conversion fails
        """
        logger.info(f"Converting audio to {output_format}: {input_file}")
        print(f"Converting audio to {output_format}...")
        
        try:
            # Set output directory if provided
            if output_dir:
                self.audio_processor.output_dir = Path(output_dir)
                self.audio_processor.output_dir.mkdir(exist_ok=True, parents=True)
            
            # Convert audio
            converted_path = self.audio_processor.convert_audio(
                input_file,
                output_format=output_format,
                bitrate=bitrate
            )
            
            print(f"Conversion complete: {os.path.basename(converted_path)}")
            return converted_path
            
        except Exception as e:
            logger.error(f"Failed to convert audio: {str(e)}")
            print(f"Error: {str(e)}")
            raise
    
    def create_voice_model(
        self,
        audio_path: str,
        model_name: Optional[str] = None
    ) -> str:
        """
        Create a voice model from an audio file.
        
        Args:
            audio_path: Path to audio file with voice to clone
            model_name: Name for the voice model
            
        Returns:
            str: Name of created voice model
            
        Raises:
            ValueError: If voice model creation fails
        """
        # Initialize voice cloner
        self._init_voice_cloner()
        
        logger.info(f"Creating voice model from: {audio_path}")
        print(f"Creating voice model from: {os.path.basename(audio_path)}")
        print("This may take a few minutes...")
        
        try:
            # Validate audio for voice cloning
            is_valid, reason = self.voice_cloner.validate_audio_for_cloning(audio_path)
            if not is_valid:
                raise ValueError(f"Audio not suitable for voice cloning: {reason}")
            
            # Create voice model
            model = self.voice_cloner.create_voice_model(
                audio_path,
                model_name=model_name
            )
            
            print(f"Voice model created successfully: {model.name}")
            print(f"Model saved in: {model.path}")
            return model.name
            
        except Exception as e:
            logger.error(f"Failed to create voice model: {str(e)}")
            print(f"Error: {str(e)}")
            raise
    
    def list_voice_models(self) -> None:
        """
        List all available voice models.
        
        Prints voice model information to console.
        """
        # Initialize voice cloner
        self._init_voice_cloner()
        
        logger.info("Listing voice models")
        
        try:
            # Get voice models
            models = self.voice_cloner.list_voice_models()
            
            if not models:
                print("No voice models found")
                return
                
            print(f"Found {len(models)} voice models:")
            print("-" * 60)
            for i, model in enumerate(models, 1):
                print(f"{i}. {model.name}")
                print(f"   Created: {time.ctime(model.created_at)}")
                print(f"   Duration: {model.duration:.1f} seconds")
                if model.source_file:
                    print(f"   Source: {model.source_file}")
                print("-" * 60)
                
        except Exception as e:
            logger.error(f"Failed to list voice models: {str(e)}")
            print(f"Error: {str(e)}")
            raise
    
    def synthesize_speech(
        self,
        text: str,
        voice_model: str,
        output_path: Optional[str] = None,
        preset: str = "standard",
        seed: int = 0
    ) -> str:
        """
        Synthesize speech using a voice model.
        
        Args:
            text: Text to synthesize
            voice_model: Name of voice model to use
            output_path: Path to save synthesized audio
            preset: Quality preset
            seed: Random seed for synthesis
            
        Returns:
            str: Path to synthesized audio file
            
        Raises:
            ValueError: If speech synthesis fails
        """
        # Initialize voice cloner
        self._init_voice_cloner()
        
        logger.info(f"Synthesizing speech using voice model: {voice_model}")
        print(f"Synthesizing speech using voice model: {voice_model}")
        print("This may take a few minutes...")
        
        try:
            # Synthesize speech
            output_file = self.voice_cloner.synthesize_speech(
                text,
                voice_model,
                output_path=output_path,
                preset=preset,
                seed=seed
            )
            
            print(f"Speech synthesis complete: {os.path.basename(output_file)}")
            print(f"Saved to: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to synthesize speech: {str(e)}")
            print(f"Error: {str(e)}")
            raise
    
    def run(self) -> int:
        """
        Run the application based on command-line arguments.
        
        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        # Parse arguments
        args = self._parse_arguments()
        
        try:
            # Determine operation mode
            mode = CliMode(args.mode)
            
            # Execute the appropriate operation
            if mode == CliMode.DOWNLOAD_ONLY:
                audio_path, title = self.download_youtube_audio(
                    args.url,
                    output_dir=args.output_dir
                )
                print(f"Downloaded audio saved to: {audio_path}")
                
            elif mode == CliMode.DOWNLOAD_AND_CONVERT:
                audio_path, title = self.download_youtube_audio(
                    args.url,
                    output_dir=args.output_dir
                )
                converted_path = self.convert_audio_to_mp3(
                    audio_path,
                    output_format=args.format,
                    bitrate=args.bitrate,
                    output_dir=args.output_dir
                )
                print(f"Converted audio saved to: {converted_path}")
                
            elif mode == CliMode.VOICE_CLONE:
                audio_path, title = self.download_youtube_audio(
                    args.url,
                    output_dir=args.output_dir
                )
                # Use video title as model name if not provided
                model_name = args.name if args.name else title
                model_name = self.create_voice_model(audio_path, model_name)
                print(f"Voice model '{model_name}' created successfully")
                
            elif mode == CliMode.VOICE_SYNTHESIS:
                output_file = self.synthesize_speech(
                    args.text,
                    args.voice,
                    output_path=args.output,
                    preset=args.preset,
                    seed=args.seed
                )
                print(f"Synthesized speech saved to: {output_file}")
                
            elif mode == CliMode.LIST_VOICES:
                self.list_voice_models()
            
            return 0
            
        except ValueError as e:
            # User input errors
            print(f"Error: {str(e)}")
            logger.error(str(e))
            return 1
            
        except Exception as e:
            # Unexpected errors
            print(f"An unexpected error occurred: {str(e)}")
            logger.exception("Unexpected error")
            return 2
