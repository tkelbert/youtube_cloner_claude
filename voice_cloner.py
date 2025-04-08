#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice cloning module for creating voice models and synthesizing speech.

This module provides functionality to extract voice characteristics from audio samples
and synthesize speech in the cloned voice using Tortoise TTS.
"""

import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from torch import nn

# Import Tortoise TTS (with error handling for missing dependencies)
try:
    from tortoise.api import TextToSpeech
    from tortoise.utils.audio import load_audio, load_voice
    TORTOISE_AVAILABLE = True
except ImportError:
    TORTOISE_AVAILABLE = False

from config import (OUTPUT_DIR, TEMP_DIR, VOICE_MODEL_DIR,
                   VOICE_SAMPLE_MAX_DURATION, VOICE_SAMPLE_MIN_DURATION,
                   VOICE_SYNTHESIS_MAX_LENGTH)

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class VoiceModel:
    """Data class representing a cloned voice model."""
    name: str
    path: Path
    sample_rate: int
    created_at: float
    source_file: str
    duration: float


class VoiceCloner:
    """
    Class for voice cloning and speech synthesis operations.
    
    Handles creating voice models from audio samples and synthesizing
    speech using those models with Tortoise TTS.
    """

    def __init__(self, model_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the voice cloner with a model directory.
        
        Args:
            model_dir: Directory where voice models will be stored
                      If None, uses the default voice model directory from config
                      
        Raises:
            ImportError: If required dependencies are not installed
        """
        if not TORTOISE_AVAILABLE:
            raise ImportError(
                "Tortoise TTS not installed. Install with: "
                "pip install tortoise-tts"
            )
        
        self.model_dir = Path(model_dir) if model_dir else VOICE_MODEL_DIR
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize TTS model on first use, not during initialization
        self._tts = None
    
    def _load_tts_model(self) -> None:
        """
        Load the Tortoise TTS model.
        
        This is done lazily to avoid loading the large model on initialization.
        """
        if self._tts is None:
            logger.info("Loading Tortoise TTS model (this may take a moment)...")
            # Use lower quality preset by default for faster inference
            self._tts = TextToSpeech(use_deepspeed=False, kv_cache=True)
            logger.info("Tortoise TTS model loaded successfully")
    
    def check_system_compatibility(self) -> Tuple[bool, str]:
        """
        Check if the system is compatible with voice cloning operations.
        
        Returns:
            Tuple[bool, str]: (is_compatible, reason)
        """
        # Check if CUDA is available for GPU acceleration
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            return False, "Voice cloning works best with CUDA GPU acceleration, which is not available"
        
        # Check if enough GPU memory is available (if CUDA is available)
        if cuda_available:
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                # Need at least 4GB of VRAM
                if gpu_memory < 4 * 1024 * 1024 * 1024:
                    return False, f"GPU has only {gpu_memory/1024/1024/1024:.1f}GB VRAM, 4GB+ recommended"
            except Exception:
                pass
        
        return True, "System is compatible with voice cloning"
    
    def validate_audio_for_cloning(self, audio_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Validate if an audio file is suitable for voice cloning.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        audio_path = Path(audio_path)
        
        # Check if file exists
        if not audio_path.exists():
            return False, f"Audio file does not exist: {audio_path}"
        
        try:
            # Load audio using torchaudio
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            # Check duration
            duration = waveform.shape[1] / sample_rate
            if duration < VOICE_SAMPLE_MIN_DURATION:
                return False, f"Audio too short: {duration:.1f}s (min {VOICE_SAMPLE_MIN_DURATION}s)"
            
            if duration > VOICE_SAMPLE_MAX_DURATION:
                return False, f"Audio too long: {duration:.1f}s (max {VOICE_SAMPLE_MAX_DURATION}s)"
                
            # Check channels - convert stereo to mono if needed
            if waveform.shape[0] > 1:
                # Will be automatically converted to mono, just log it
                logger.info(f"Audio has {waveform.shape[0]} channels, will be converted to mono")
            
            # Check if audio contains speech (basic check for non-silent content)
            if waveform.abs().mean() < 0.01:
                return False, "Audio appears to contain no speech (too quiet)"
                
            return True, "Audio is suitable for voice cloning"
            
        except Exception as e:
            return False, f"Failed to validate audio: {str(e)}"
    
    def create_voice_model(
        self, 
        audio_path: Union[str, Path],
        model_name: Optional[str] = None,
        auto_split: bool = False,
        audio_processor = None
    ) -> VoiceModel:
        """
        Create a voice model from an audio sample.
        
        Args:
            audio_path: Path to the audio file containing the voice to clone
            model_name: Name for the voice model
                       If None, uses the audio filename
            auto_split: Whether to automatically split long audio files
            audio_processor: AudioProcessor instance for splitting (required if auto_split is True)
                       
        Returns:
            VoiceModel: The created voice model
            
        Raises:
            ValueError: If audio is invalid or model creation fails
        """
        audio_path = Path(audio_path)
        
        # Handle long audio by splitting if needed
        audio_paths = [audio_path]
        if auto_split:
            # Validate that we have an audio processor
            if audio_processor is None:
                raise ValueError("Audio processor is required for auto-splitting")
                
            # Check if the audio is too long
            try:
                waveform, sample_rate = torchaudio.load(str(audio_path))
                duration = waveform.shape[1] / sample_rate
                
                if duration > VOICE_SAMPLE_MAX_DURATION:
                    logger.info(f"Audio is too long ({duration:.1f}s), splitting into segments")
                    # Split audio into segments
                    audio_paths = [Path(p) for p in audio_processor.split_audio_for_voice_cloning(audio_path)]
                    logger.info(f"Split audio into {len(audio_paths)} segments")
            except Exception as e:
                logger.warning(f"Failed to check audio duration: {str(e)}")
                # Continue with single file approach
        
        # Generate model name if not provided
        if model_name is None:
            model_name = audio_path.stem
        
        # Sanitize model name (remove special characters)
        model_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in model_name)
        
        # Create model directory
        timestamp = time.time()
        model_path = self.model_dir / f"{model_name}_{int(timestamp)}"
        model_path.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Creating voice model '{model_name}' from {len(audio_paths)} audio segments")
        
        try:
            # Initialize variables to track the combined audio
            combined_duration = 0
            all_source_files = []
            
            # Process each audio segment
            for i, segment_path in enumerate(audio_paths):
                # Validate each segment
                is_valid, reason = self.validate_audio_for_cloning(segment_path)
                if not is_valid:
                    logger.warning(f"Skipping invalid segment {segment_path}: {reason}")
                    continue
                
                # Load audio for processing
                waveform, sample_rate = torchaudio.load(str(segment_path))
                
                # Convert to mono if needed (average channels)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # Resample to 22050Hz if needed (Tortoise's expected sample rate)
                if sample_rate != 22050:
                    resampler = torchaudio.transforms.Resample(sample_rate, 22050)
                    waveform = resampler(waveform)
                    sample_rate = 22050
                
                # Calculate duration after resampling
                duration = waveform.shape[1] / sample_rate
                combined_duration += duration
                
                # For the first segment or if we only have one segment, save it as the main voice
                if i == 0:
                    main_voice_path = model_path / "voice.wav"
                    torchaudio.save(str(main_voice_path), waveform, sample_rate)
                    logger.info(f"Saved main voice sample: {main_voice_path}")
                
                # Save each segment as an additional voice sample for more data
                sample_path = model_path / f"voice_sample_{i+1:03d}.wav"
                torchaudio.save(str(sample_path), waveform, sample_rate)
                logger.info(f"Saved voice sample {i+1}: {sample_path}")
                
                # Track source files
                all_source_files.append(str(segment_path))
            
            if combined_duration == 0:
                raise ValueError("No valid audio segments found for voice cloning")
            
            # Create metadata file with information about the voice
            with open(model_path / "metadata.txt", "w") as f:
                f.write(f"Source files: {', '.join(all_source_files)}\n")
                f.write(f"Created at: {time.ctime(timestamp)}\n")
                f.write(f"Total duration: {combined_duration:.2f} seconds\n")
                f.write(f"Number of segments: {len(audio_paths)}\n")
                f.write(f"Sample rate: {sample_rate} Hz\n")
            
            # Create and return the voice model object
            model = VoiceModel(
                name=model_name,
                path=model_path,
                sample_rate=sample_rate,
                created_at=timestamp,
                source_file=str(audio_path),  # Keep the original file as reference
                duration=combined_duration
            )
            
            logger.info(f"Voice model created successfully: {model_path}")
            return model
            
        except Exception as e:
            # Clean up if model creation fails
            if model_path.exists():
                shutil.rmtree(model_path)
            logger.error(f"Failed to create voice model: {str(e)}")
            raise ValueError(f"Failed to create voice model: {str(e)}")
    
    def list_voice_models(self) -> List[VoiceModel]:
        """
        List all available voice models.
        
        Returns:
            List[VoiceModel]: List of available voice models
        """
        models = []
        
        # Look for subdirectories in the model directory
        for dir_path in self.model_dir.iterdir():
            if not dir_path.is_dir():
                continue
                
            # Check if it has the required files for a voice model
            if not (dir_path / "voice.wav").exists():
                continue
                
            try:
                # Parse the name and timestamp from directory name
                dir_name = dir_path.name
                
                # Try to get name and timestamp from directory name
                if '_' in dir_name:
                    name, timestamp_str = dir_name.rsplit('_', 1)
                    try:
                        timestamp = float(timestamp_str)
                    except ValueError:
                        # If timestamp isn't a number, use the whole name
                        name = dir_name
                        timestamp = dir_path.stat().st_mtime
                else:
                    name = dir_name
                    timestamp = dir_path.stat().st_mtime
                
                # Get sample rate and metadata
                source_file = ""
                duration = 0.0
                
                # Try to load metadata if available
                if (dir_path / "metadata.txt").exists():
                    with open(dir_path / "metadata.txt", "r") as f:
                        for line in f:
                            if line.startswith("Source file:"):
                                source_file = line.split(":", 1)[1].strip()
                            elif line.startswith("Duration:"):
                                try:
                                    duration = float(line.split(":", 1)[1].split()[0])
                                except:
                                    pass
                
                # Get sample rate from audio file
                try:
                    waveform, sample_rate = torchaudio.load(str(dir_path / "voice.wav"))
                except:
                    sample_rate = 22050  # default
                
                # Create model object
                model = VoiceModel(
                    name=name,
                    path=dir_path,
                    sample_rate=sample_rate,
                    created_at=timestamp,
                    source_file=source_file,
                    duration=duration
                )
                
                models.append(model)
                
            except Exception as e:
                logger.warning(f"Skipping invalid voice model directory {dir_path}: {str(e)}")
        
        # Sort by creation time (newest first)
        models.sort(key=lambda m: m.created_at, reverse=True)
        return models
    
    def get_voice_model(self, model_name: str) -> Optional[VoiceModel]:
        """
        Get a specific voice model by name.
        
        Args:
            model_name: Name of the voice model to retrieve
            
        Returns:
            Optional[VoiceModel]: The voice model if found, None otherwise
        """
        models = self.list_voice_models()
        
        # Try exact match first
        for model in models:
            if model.name == model_name:
                return model
        
        # Try partial match
        for model in models:
            if model_name in model.name:
                return model
                
        return None
    
    def delete_voice_model(self, model_name: str) -> bool:
        """
        Delete a voice model.
        
        Args:
            model_name: Name of the voice model to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        model = self.get_voice_model(model_name)
        if model is None:
            return False
            
        try:
            shutil.rmtree(model.path)
            logger.info(f"Deleted voice model: {model.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete voice model {model.name}: {str(e)}")
            return False
    
    def synthesize_speech(
        self,
        text: str,
        voice_model: Union[str, VoiceModel],
        output_path: Optional[Union[str, Path]] = None,
        seed: int = 0,
        preset: str = 'standard'
    ) -> str:
        """
        Synthesize speech from text using a voice model.
        
        Args:
            text: Text to synthesize into speech
            voice_model: Voice model to use (name or VoiceModel object)
            output_path: Path to save the synthesized audio
                        If None, generates a path in the output directory
            seed: Random seed for synthesis (for reproducibility)
            preset: Quality preset ('ultra_fast', 'fast', 'standard', 'high_quality')
            
        Returns:
            str: Path to the synthesized audio file
            
        Raises:
            ValueError: If text is invalid, model not found, or synthesis fails
        """
        # Validate text
        if not text or len(text) > VOICE_SYNTHESIS_MAX_LENGTH:
            raise ValueError(
                f"Text must be between 1 and {VOICE_SYNTHESIS_MAX_LENGTH} characters"
            )
            
        # Get voice model
        if isinstance(voice_model, str):
            model = self.get_voice_model(voice_model)
            if model is None:
                raise ValueError(f"Voice model not found: {voice_model}")
        else:
            model = voice_model
            
        # Load TTS model if not already loaded
        self._load_tts_model()
        
        # Generate output path if not provided
        if output_path is None:
            # Create a safe filename from the first few words of text
            safe_name = '_'.join(text.split()[:5])
            safe_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in safe_name)
            if len(safe_name) > 50:
                safe_name = safe_name[:50]
                
            output_path = OUTPUT_DIR / f"{model.name}_{safe_name}_{int(time.time())}.wav"
        else:
            output_path = Path(output_path)
            
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Synthesizing speech from text using voice model {model.name}")
        
        try:
            # Create a temporary directory for Tortoise
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy voice model to temporary directory with expected structure
                voice_samples = []
                
                # Load voice sample
                voice_samples = [load_audio(str(model.path / "voice.wav"), 22050)]
                
                # Check for additional voice samples
                for i in range(1, 10):  # Check for up to 9 additional samples
                    sample_path = model.path / f"voice_sample_{i:03d}.wav"
                    if sample_path.exists():
                        voice_samples.append(load_audio(str(sample_path), 22050))
                
                # Set torch seed for reproducibility
                torch.manual_seed(seed)
                
                # Generate speech
                gen_result = self._tts.tts_with_preset(
                    text,
                    voice_samples=voice_samples,
                    preset=preset,
                    k=1,  # Only generate one candidate
                    use_deterministic_seed=seed
                )
                
                # Handle the result - could be tensor or ndarray, ensure it's ndarray
                audio_data = gen_result[0]
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.detach().cpu().numpy()
                
                # Ensure audio_data is properly shaped (should be 1D array)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.squeeze()
                
                # Save the generated audio
                torchaudio.save(
                    str(output_path),
                    torch.from_numpy(audio_data[None, :]),  # Add batch dimension
                    22050
                )
                
                logger.info(f"Speech synthesis complete: {output_path}")
                return str(output_path)
                
        except Exception as e:
            logger.error(f"Speech synthesis failed: {str(e)}")
            raise ValueError(f"Failed to synthesize speech: {str(e)}")
