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
                   SUPPORTED_AUDIO_FORMATS, TEMP_DIR, VOICE_SAMPLE_MAX_DURATION,
                   VOICE_SAMPLE_MIN_DURATION)

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
    
    def split_audio_for_voice_cloning(
        self, 
        input_file: Union[str, Path],
        max_duration: float = VOICE_SAMPLE_MAX_DURATION,
        min_duration: float = VOICE_SAMPLE_MIN_DURATION,
        target_duration: float = 300.0,  # 5 minutes
        overlap: float = 5.0,  # 5 seconds of overlap
        detect_silence: bool = True,
        silence_threshold: int = -40,  # dB
        min_silence_length: int = 1000  # ms
    ) -> List[str]:
        """
        Split a long audio file into smaller segments for voice cloning.
        
        Args:
            input_file: Path to the input audio file
            max_duration: Maximum duration for each segment (seconds)
            min_duration: Minimum duration for a usable segment (seconds)
            target_duration: Target duration for segments (seconds)
            overlap: Overlap between segments (seconds)
            detect_silence: Whether to try to split at silent parts
            silence_threshold: Threshold for silence detection (dB)
            min_silence_length: Minimum silence length to consider (ms)
            
        Returns:
            List[str]: Paths to the split audio segments
            
        Raises:
            ValueError: If input file is invalid or splitting fails
        """
        input_file = Path(input_file)
        
        if not self.is_valid_audio_file(input_file):
            raise ValueError(f"Invalid or corrupted input audio file: {input_file}")
        
        logger.info(f"Splitting audio file for voice cloning: {input_file}")
        
        try:
            # Load the audio file
            audio = AudioSegment.from_file(str(input_file))
            
            # Get duration in seconds
            duration = len(audio) / 1000.0
            
            # Check if splitting is necessary
            if duration <= max_duration:
                logger.info(f"Audio file is already within acceptable length: {duration:.1f}s")
                return [str(input_file)]
                
            logger.info(f"Audio duration: {duration:.1f}s, splitting into segments")
            
            # Calculate number of segments needed
            overlap_ms = int(overlap * 1000)
            target_duration_ms = int(target_duration * 1000)
            
            # Create output directory for segments
            output_dir = self.output_dir / f"{input_file.stem}_segments"
            output_dir.mkdir(exist_ok=True, parents=True)
            
            segments = []
            segment_paths = []
            
            if detect_silence:
                # Try to find silent parts for clean splitting
                logger.info("Detecting silence for intelligent splitting")
                
                # Find silent parts
                silence_ranges = pydub.silence.detect_silence(
                    audio,
                    min_silence_len=min_silence_length,
                    silence_thresh=silence_threshold
                )
                
                if len(silence_ranges) < 2:
                    logger.warning("Not enough silent parts detected, falling back to time-based splitting")
                    detect_silence = False
                else:
                    logger.info(f"Found {len(silence_ranges)} silent ranges")
                    
                    # Use the silent ranges to determine split points
                    current_start = 0
                    current_duration = 0
                    
                    for silence_start, silence_end in silence_ranges:
                        # If adding up to this silence would exceed our target, split here
                        if current_duration + (silence_start - current_start) > target_duration_ms:
                            # Use the middle of the silence as the split point
                            split_point = (silence_start + silence_end) // 2
                            
                            # Create segment
                            segment = audio[current_start:split_point]
                            segment_duration = len(segment) / 1000.0
                            
                            # Only keep segments that meet the minimum duration
                            if segment_duration >= min_duration:
                                segments.append(segment)
                                current_start = max(0, split_point - overlap_ms)
                                current_duration = 0
                        
                        current_duration += (silence_end - max(current_start, silence_start))
                    
                    # Add the last segment if it's long enough
                    if len(audio) - current_start >= min_duration * 1000:
                        segments.append(audio[current_start:])
            
            # Fall back to time-based splitting if silence detection didn't work or is disabled
            if not detect_silence or not segments:
                logger.info("Using time-based splitting")
                
                # Calculate segment length with overlap
                effective_segment_length = target_duration_ms
                
                # Split the audio at regular intervals
                start_time = 0
                while start_time < len(audio):
                    end_time = min(start_time + effective_segment_length, len(audio))
                    segment = audio[start_time:end_time]
                    
                    # Only keep segments that meet the minimum duration
                    if len(segment) / 1000.0 >= min_duration:
                        segments.append(segment)
                    
                    # Move start time for next segment, considering overlap
                    start_time = end_time - overlap_ms
            
            logger.info(f"Created {len(segments)} segments")
            
            # Save segments to files
            for i, segment in enumerate(segments):
                segment_path = output_dir / f"{input_file.stem}_segment_{i+1:03d}.wav"
                segment.export(str(segment_path), format="wav")
                segment_paths.append(str(segment_path))
                logger.info(f"Saved segment {i+1}: {segment_path} ({len(segment)/1000.0:.1f}s)")
            
            return segment_paths
            
        except Exception as e:
            logger.error(f"Failed to split audio file: {str(e)}")
            raise ValueError(f"Failed to split audio file: {str(e)}")
