#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio processing module for handling audio file operations.

This module provides functionality to convert audio formats, validate audio files,
and perform basic audio processing operations using ffmpeg.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pydub import AudioSegment
import soundfile as sf

from config import (DEFAULT_AUDIO_FORMAT, DEFAULT_BITRATE, DEFAULT_CHANNELS,
                   DEFAULT_SAMPLE_RATE, FFMPEG_TIMEOUT, OUTPUT_DIR,
                   SUPPORTED_AUDIO_FORMATS, TEMP_DIR)

# Configure module logger
logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Class for processing audio files.
    
    Handles audio conversion, validation, and basic manipulations.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the audio processor with configurable output directory.
        
        Args:
            output_dir: Directory where processed audio will be saved.
                        If None, uses the default output directory from config.
        """
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Verify ffmpeg is installed
        self._verify_ffmpeg()

    def _verify_ffmpeg(self) -> None:
        """
        Verify that ffmpeg is installed and accessible.
        
        Raises:
            RuntimeError: If ffmpeg is not found in the system path
        """
        try:
            subprocess.run(
                ['ffmpeg', '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=5
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg and ensure it's in your system PATH."
            )

    def is_valid_audio_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if a file is a valid audio file.
        
        Args:
            file_path: Path to the audio file to validate
            
        Returns:
            bool: True if valid audio file, False otherwise
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False
        
        # Check extension
        extension = file_path.suffix.lower().lstrip('.')
        if extension not in SUPPORTED_AUDIO_FORMATS:
            logger.warning(f"Unsupported audio format: {extension}")
            
        # Try to open with pydub to verify it's valid audio
        try:
            audio = AudioSegment.from_file(str(file_path))
            # Check if it has a reasonable duration
            if len(audio) < 100:  # Less than 100ms is suspicious
                logger.warning(f"Audio file too short: {len(audio)}ms")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to validate audio file: {str(e)}")
            return False

    def get_audio_info(self, file_path: Union[str, Path]) -> Dict:
        """
        Get information about an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dict: Audio metadata including format, duration, etc.
            
        Raises:
            ValueError: If the file is not a valid audio file
        """
        file_path = Path(file_path)
        
        if not self.is_valid_audio_file(file_path):
            raise ValueError(f"Invalid or corrupted audio file: {file_path}")
        
        try:
            audio = AudioSegment.from_file(str(file_path))
            return {
                'duration': len(audio) / 1000,  # in seconds
                'channels': audio.channels,
                'sample_rate': audio.frame_rate,
                'sample_width': audio.sample_width,
                'format': file_path.suffix.lower().lstrip('.'),
                'file_size': file_path.stat().st_size
            }
        except Exception as e:
            raise ValueError(f"Failed to get audio info: {str(e)}")

    def convert_audio(
        self, 
        input_file: Union[str, Path],
        output_format: str = DEFAULT_AUDIO_FORMAT,
        bitrate: str = DEFAULT_BITRATE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        output_file: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Convert an audio file to a different format or quality.
        
        Args:
            input_file: Path to the input audio file
            output_format: Target audio format (mp3, wav, etc.)
            bitrate: Target bitrate for the output (e.g., "192k")
            sample_rate: Target sample rate in Hz
            channels: Number of audio channels (1=mono, 2=stereo)
            output_file: Optional path to save the output file
                         If None, generates a path based on the input filename
            
        Returns:
            str: Path to the converted audio file
            
        Raises:
            ValueError: If input file is invalid or conversion fails
        """
        input_file = Path(input_file)
        
        if not self.is_valid_audio_file(input_file):
            raise ValueError(f"Invalid or corrupted input audio file: {input_file}")
        
        if output_format not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Generate output filename if not provided
        if output_file is None:
            output_file = self.output_dir / f"{input_file.stem}.{output_format}"
        else:
            output_file = Path(output_file)
        
        # Ensure output directory exists
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Converting {input_file} to {output_file}")
        
        try:
            # Build ffmpeg command
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-i', str(input_file),
                '-ar', str(sample_rate),
                '-ac', str(channels),
                '-b:a', bitrate
            ]
            
            # Add format-specific options
            if output_format == 'mp3':
                cmd.extend(['-codec:a', 'libmp3lame'])
            elif output_format == 'wav':
                cmd.extend(['-codec:a', 'pcm_s16le'])
            elif output_format == 'flac':
                cmd.extend(['-codec:a', 'flac'])
            elif output_format == 'ogg':
                cmd.extend(['-codec:a', 'libvorbis'])
                
            # Add output file path
            cmd.append(str(output_file))
            
            # Run ffmpeg process
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=FFMPEG_TIMEOUT
            )
            
            # Verify the output file exists and is valid
            if not output_file.exists():
                raise ValueError(f"Conversion failed: Output file not created")
            
            if not self.is_valid_audio_file(output_file):
                raise ValueError(f"Conversion produced invalid audio file")
                
            logger.info(f"Successfully converted to {output_file}")
            return str(output_file)
            
        except subprocess.SubprocessError as e:
            logger.error(f"FFMPEG conversion failed: {str(e)}")
            raise ValueError(f"Audio conversion failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during conversion: {str(e)}")
            raise ValueError(f"Audio conversion failed: {str(e)}")

    def normalize_audio(self, input_file: Union[str, Path], target_db: float = -3.0) -> str:
        """
        Normalize audio volume to a target decibel level.
        
        Args:
            input_file: Path to the input audio file
            target_db: Target loudness in dB
            
        Returns:
            str: Path to the normalized audio file
            
        Raises:
            ValueError: If input file is invalid or normalization fails
        """
        input_file = Path(input_file)
        
        if not self.is_valid_audio_file(input_file):
            raise ValueError(f"Invalid or corrupted input audio file: {input_file}")
        
        # Create output filename
        output_file = self.output_dir / f"{input_file.stem}_normalized{input_file.suffix}"
        
        logger.info(f"Normalizing {input_file} to {target_db}dB")
        
        try:
            # Use ffmpeg for normalization
            cmd = [
                'ffmpeg',
                '-y',
                '-i', str(input_file),
                '-filter:a', f'loudnorm=I={target_db}:LRA=11:TP=-1.5',
                str(output_file)
            ]
            
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=FFMPEG_TIMEOUT
            )
            
            if not output_file.exists():
                raise ValueError(f"Normalization failed: Output file not created")
            
            if not self.is_valid_audio_file(output_file):
                raise ValueError(f"Normalization produced invalid audio file")
                
            logger.info(f"Successfully normalized to {output_file}")
            return str(output_file)
            
        except subprocess.SubprocessError as e:
            logger.error(f"FFMPEG normalization failed: {str(e)}")
            raise ValueError(f"Audio normalization failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during normalization: {str(e)}")
            raise ValueError(f"Audio normalization failed: {str(e)}")

    def extract_audio_segment(
        self, 
        input_file: Union[str, Path],
        start_time: float,
        end_time: float
    ) -> str:
        """
        Extract a segment of audio between start and end times.
        
        Args:
            input_file: Path to the input audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            str: Path to the extracted audio segment
            
        Raises:
            ValueError: If input file is invalid or extraction fails
        """
        input_file = Path(input_file)
        
        if not self.is_valid_audio_file(input_file):
            raise ValueError(f"Invalid or corrupted input audio file: {input_file}")
        
        if start_time >= end_time:
            raise ValueError(f"Start time ({start_time}) must be less than end time ({end_time})")
        
        # Get audio info to validate times
        audio_info = self.get_audio_info(input_file)
        if end_time > audio_info['duration']:
            raise ValueError(f"End time ({end_time}) exceeds audio duration ({audio_info['duration']})")
        
        # Create output filename
        output_file = self.output_dir / f"{input_file.stem}_{start_time:.1f}-{end_time:.1f}{input_file.suffix}"
        
        logger.info(f"Extracting segment from {start_time}s to {end_time}s from {input_file}")
        
        try:
            # Use ffmpeg for extraction
            cmd = [
                'ffmpeg',
                '-y',
                '-i', str(input_file),
                '-ss', str(start_time),
                '-to', str(end_time),
                '-c', 'copy',  # Use copy to avoid re-encoding
                str(output_file)
            ]
            
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=FFMPEG_TIMEOUT
            )
            
            if not output_file.exists():
                raise ValueError(f"Extraction failed: Output file not created")
            
            if not self.is_valid_audio_file(output_file):
                raise ValueError(f"Extraction produced invalid audio file")
                
            logger.info(f"Successfully extracted segment to {output_file}")
            return str(output_file)
            
        except subprocess.SubprocessError as e:
            logger.error(f"FFMPEG extraction failed: {str(e)}")
            raise ValueError(f"Audio segment extraction failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during extraction: {str(e)}")
            raise ValueError(f"Audio segment extraction failed: {str(e)}")

    def load_audio_as_array(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load audio file as a numpy array for further processing.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple[np.ndarray, int]: Audio data as numpy array and sample rate
            
        Raises:
            ValueError: If file is invalid or loading fails
        """
        file_path = Path(file_path)
        
        if not self.is_valid_audio_file(file_path):
            raise ValueError(f"Invalid or corrupted audio file: {file_path}")
        
        try:
            # Use soundfile for high-quality audio loading
            data, sample_rate = sf.read(str(file_path))
            return data, sample_rate
        except Exception as e:
            logger.error(f"Failed to load audio as array: {str(e)}")
            raise ValueError(f"Failed to load audio as array: {str(e)}")
