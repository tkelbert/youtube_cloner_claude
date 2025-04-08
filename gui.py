#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI module for YouTube to MP3 downloader and voice cloning application.

This module provides a PyQt5-based graphical user interface for the application,
with dark mode support and user-friendly controls.
"""

import os
import sys
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PyQt5.QtCore import (QObject, QSize, Qt, QThread, QTimer, pyqtSignal,
                         pyqtSlot)
from PyQt5.QtGui import QColor, QFont, QIcon, QPalette
from PyQt5.QtWidgets import (QAction, QApplication, QCheckBox, QComboBox,
                            QFileDialog, QFrame, QGridLayout, QGroupBox,
                            QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                            QMessageBox, QProgressBar, QPushButton, QScrollArea,
                            QSizePolicy, QSlider, QSpacerItem, QSpinBox,
                            QStatusBar, QTabWidget, QTextEdit, QToolBar,
                            QVBoxLayout, QWidget)

# Import local modules
from audio_processor import AudioProcessor
from config import OUTPUT_DIR, VOICE_MODEL_DIR
from voice_cloner import VoiceCloner
from youtube_downloader import YouTubeDownloader


class Theme(Enum):
    """Enum for the application themes"""
    LIGHT = "Light"
    DARK = "Dark"
    SYSTEM = "System"


class WorkerSignals(QObject):
    """
    Signals for worker thread communication.
    """
    started = pyqtSignal()
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    result = pyqtSignal(object)
    status = pyqtSignal(str)


class Worker(QThread):
    """
    Worker thread for background tasks.
    """
    def __init__(self, fn, *args, **kwargs):
        """
        Initialize the worker thread.
        
        Args:
            fn: The function to execute in the thread
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
        """
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        
    def run(self):
        """
        Run the worker function and emit signals.
        """
        self.signals.started.emit()
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class MainWindow(QMainWindow):
    """
    Main window for the application.
    """
    def __init__(self):
        """
        Initialize the main window.
        """
        super(MainWindow, self).__init__()
        
        # Initialize components
        self.youtube_downloader = YouTubeDownloader()
        self.audio_processor = AudioProcessor()
        self.voice_cloner = None  # Initialize when needed
        
        # Set up the UI
        self.setup_ui()
        
        # Initialize variables
        self.workers = []  # Keep track of worker threads
        
        # Set theme from saved settings or default to dark
        self.set_theme(Theme.DARK)
    
    def setup_ui(self):
        """
        Set up the user interface.
        """
        # Basic window setup
        self.setWindowTitle("YouTube to MP3 & Voice Cloning")
        self.setMinimumSize(800, 600)
        
        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_download_tab()
        self.create_voice_clone_tab()
        self.create_voice_synthesis_tab()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        # Create progress bar in status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(150)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Create menu bar
        self.create_menu_bar()
    
    def create_menu_bar(self):
        """
        Create the application menu bar.
        """
        # File menu
        file_menu = self.menuBar().addMenu("&File")
        
        # Open output folder action
        open_output_action = QAction("Open Output Folder", self)
        open_output_action.triggered.connect(self.open_output_folder)
        file_menu.addAction(open_output_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Appearance menu
        appearance_menu = self.menuBar().addMenu("&Appearance")
        
        # Theme submenu
        theme_menu = appearance_menu.addMenu("Theme")
        
        # Theme actions
        self.theme_actions = {}
        for theme in Theme:
            action = QAction(theme.value, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, t=theme: self.set_theme(t))
            theme_menu.addAction(action)
            self.theme_actions[theme] = action
        
        # Set dark mode as checked by default
        self.theme_actions[Theme.DARK].setChecked(True)
        
        # Help menu
        help_menu = self.menuBar().addMenu("&Help")
        
        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
    
    def create_download_tab(self):
        """
        Create the YouTube download and conversion tab.
        """
        download_tab = QWidget()
        download_layout = QVBoxLayout(download_tab)
        
        # URL input group
        url_group = QGroupBox("YouTube URL")
        url_layout = QHBoxLayout(url_group)
        
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter YouTube video URL...")
        url_layout.addWidget(self.url_input)
        
        download_layout.addWidget(url_group)
        
        # Output options group
        output_group = QGroupBox("Output Options")
        output_layout = QGridLayout(output_group)
        
        output_layout.addWidget(QLabel("Format:"), 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["mp3", "wav", "ogg", "flac"])
        output_layout.addWidget(self.format_combo, 0, 1)
        
        output_layout.addWidget(QLabel("Bitrate:"), 1, 0)
        self.bitrate_combo = QComboBox()
        self.bitrate_combo.addItems(["128k", "192k", "256k", "320k"])
        self.bitrate_combo.setCurrentText("192k")
        output_layout.addWidget(self.bitrate_combo, 1, 1)
        
        output_layout.addWidget(QLabel("Output Directory:"), 2, 0)
        
        output_dir_layout = QHBoxLayout()
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setText(str(OUTPUT_DIR))
        output_dir_layout.addWidget(self.output_dir_input)
        
        self.output_dir_button = QPushButton("Browse...")
        self.output_dir_button.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(self.output_dir_button)
        
        output_layout.addLayout(output_dir_layout, 2, 1)
        
        download_layout.addWidget(output_group)
        
        # Action buttons
        actions_layout = QHBoxLayout()
        
        # Download only button
        self.download_button = QPushButton("Download Audio Only")
        self.download_button.clicked.connect(self.download_audio)
        actions_layout.addWidget(self.download_button)
        
        # Download and convert button
        self.convert_button = QPushButton("Download and Convert")
        self.convert_button.clicked.connect(self.download_and_convert)
        self.convert_button.setDefault(True)
        actions_layout.addWidget(self.convert_button)
        
        download_layout.addLayout(actions_layout)
        
        # Add spacer to push everything to the top
        download_layout.addStretch()
        
        # Set tab
        self.tabs.addTab(download_tab, "Download && Convert")
    
    def create_voice_clone_tab(self):
        """
        Create the voice cloning tab.
        """
        voice_clone_tab = QWidget()
        voice_clone_layout = QVBoxLayout(voice_clone_tab)
        
        # Source group - either YouTube URL or local file
        source_group = QGroupBox("Voice Source")
        source_layout = QVBoxLayout(source_group)
        
        # Radio button equivalent using a combobox for simplicity
        self.source_type_combo = QComboBox()
        self.source_type_combo.addItems(["YouTube URL", "Local Audio File"])
        self.source_type_combo.currentIndexChanged.connect(self.toggle_source_type)
        source_layout.addWidget(self.source_type_combo)
        
        # YouTube URL input (shown by default)
        self.clone_url_input = QLineEdit()
        self.clone_url_input.setPlaceholderText("Enter YouTube video URL...")
        source_layout.addWidget(self.clone_url_input)
        
        # Local file input (hidden by default)
        self.local_file_layout = QHBoxLayout()
        self.local_file_input = QLineEdit()
        self.local_file_input.setPlaceholderText("Select audio file...")
        self.local_file_layout.addWidget(self.local_file_input)
        
        self.browse_file_button = QPushButton("Browse...")
        self.browse_file_button.clicked.connect(self.browse_audio_file)
        self.local_file_layout.addWidget(self.browse_file_button)
        
        source_layout.addLayout(self.local_file_layout)
        self.local_file_input.setVisible(False)
        self.browse_file_button.setVisible(False)
        
        voice_clone_layout.addWidget(source_group)
        
        # Voice model options
        model_group = QGroupBox("Voice Model")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("Model Name:"), 0, 0)
        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText("Leave blank to use video/file name")
        model_layout.addWidget(self.model_name_input, 0, 1)
        
        voice_clone_layout.addWidget(model_group)
        
        # Create voice model button
        self.create_model_button = QPushButton("Create Voice Model")
        self.create_model_button.clicked.connect(self.create_voice_model)
        voice_clone_layout.addWidget(self.create_model_button)
        
        # Existing models group
        models_group = QGroupBox("Existing Voice Models")
        models_layout = QVBoxLayout(models_group)
        
        # Refresh and delete buttons
        models_button_layout = QHBoxLayout()
        
        self.refresh_models_button = QPushButton("Refresh")
        self.refresh_models_button.clicked.connect(self.refresh_voice_models)
        models_button_layout.addWidget(self.refresh_models_button)
        
        self.delete_model_button = QPushButton("Delete Selected")
        self.delete_model_button.clicked.connect(self.delete_voice_model)
        models_button_layout.addWidget(self.delete_model_button)
        
        models_layout.addLayout(models_button_layout)
        
        # Models list (as a text view for simplicity)
        self.models_text = QTextEdit()
        self.models_text.setReadOnly(True)
        models_layout.addWidget(self.models_text)
        
        voice_clone_layout.addWidget(models_group)
        
        # Set tab
        self.tabs.addTab(voice_clone_tab, "Voice Cloning")
    
    def create_voice_synthesis_tab(self):
        """
        Create the voice synthesis tab.
        """
        synthesis_tab = QWidget()
        synthesis_layout = QVBoxLayout(synthesis_tab)
        
        # Voice model selection
        model_group = QGroupBox("Voice Model")
        model_layout = QHBoxLayout(model_group)
        
        model_layout.addWidget(QLabel("Select Voice:"))
        self.voice_model_combo = QComboBox()
        # Will be populated when tab is activated
        model_layout.addWidget(self.voice_model_combo)
        
        self.refresh_synthesis_button = QPushButton("Refresh")
        self.refresh_synthesis_button.clicked.connect(self.refresh_synthesis_voices)
        model_layout.addWidget(self.refresh_synthesis_button)
        
        synthesis_layout.addWidget(model_group)
        
        # Text input
        text_group = QGroupBox("Text to Synthesize")
        text_layout = QVBoxLayout(text_group)
        
        self.synthesis_text = QTextEdit()
        self.synthesis_text.setPlaceholderText("Enter text to convert to speech...")
        text_layout.addWidget(self.synthesis_text)
        
        synthesis_layout.addWidget(text_group)
        
        # Synthesis options
        options_group = QGroupBox("Synthesis Options")
        options_layout = QGridLayout(options_group)
        
        options_layout.addWidget(QLabel("Quality:"), 0, 0)
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["ultra_fast", "fast", "standard", "high_quality"])
        self.quality_combo.setCurrentText("standard")
        options_layout.addWidget(self.quality_combo, 0, 1)
        
        options_layout.addWidget(QLabel("Random Seed:"), 1, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setMinimum(0)
        self.seed_spin.setMaximum(999999)
        self.seed_spin.setValue(0)
        options_layout.addWidget(self.seed_spin, 1, 1)
        
        options_layout.addWidget(QLabel("Output File:"), 2, 0)
        
        output_file_layout = QHBoxLayout()
        self.synthesis_output_input = QLineEdit()
        self.synthesis_output_input.setPlaceholderText("Leave blank for automatic filename")
        output_file_layout.addWidget(self.synthesis_output_input)
        
        self.synthesis_output_button = QPushButton("Browse...")
        self.synthesis_output_button.clicked.connect(self.browse_synthesis_output)
        output_file_layout.addWidget(self.synthesis_output_button)
        
        options_layout.addLayout(output_file_layout, 2, 1)
        
        synthesis_layout.addWidget(options_group)
        
        # Synthesize button
        self.synthesize_button = QPushButton("Synthesize Speech")
        self.synthesize_button.clicked.connect(self.synthesize_speech)
        synthesis_layout.addWidget(self.synthesize_button)
        
        # Add spacer to push everything to the top
        synthesis_layout.addStretch()
        
        # Set tab
        self.tabs.addTab(synthesis_tab, "Voice Synthesis")
        
        # Connect tab changed signal to refresh the voice models when synthesis tab is selected
        self.tabs.currentChanged.connect(self.on_tab_changed)
    
    def on_tab_changed(self, index):
        """
        Handle tab selection changes.
        
        Args:
            index: Index of the selected tab
        """
        if index == 2:  # Voice Synthesis tab
            self.refresh_synthesis_voices()
    
    def toggle_source_type(self, index):
        """
        Toggle between YouTube URL and local file inputs.
        
        Args:
            index: Index of the selected source type
        """
        if index == 0:  # YouTube URL
            self.clone_url_input.setVisible(True)
            self.local_file_input.setVisible(False)
            self.browse_file_button.setVisible(False)
        else:  # Local file
            self.clone_url_input.setVisible(False)
            self.local_file_input.setVisible(True)
            self.browse_file_button.setVisible(True)
    
    def browse_output_dir(self):
        """
        Open a dialog to select output directory.
        """
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", str(OUTPUT_DIR)
        )
        if directory:
            self.output_dir_input.setText(directory)
    
    def browse_audio_file(self):
        """
        Open a dialog to select an audio file.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", str(OUTPUT_DIR),
            "Audio Files (*.mp3 *.wav *.ogg *.flac *.m4a);;All Files (*)"
        )
        if file_path:
            self.local_file_input.setText(file_path)
    
    def browse_synthesis_output(self):
        """
        Open a dialog to select synthesis output file.
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Synthesized Speech", str(OUTPUT_DIR),
            "WAV Files (*.wav);;All Files (*)"
        )
        if file_path:
            if not file_path.endswith(".wav"):
                file_path += ".wav"
            self.synthesis_output_input.setText(file_path)
    
    def open_output_folder(self):
        """
        Open the output folder in the system file explorer.
        """
        output_dir = self.output_dir_input.text() or str(OUTPUT_DIR)
        
        # Platform-specific folder opening
        if sys.platform == 'win32':
            os.startfile(output_dir)
        elif sys.platform == 'darwin':  # macOS
            os.system(f'open "{output_dir}"')
        else:  # Linux/Unix
            os.system(f'xdg-open "{output_dir}"')
    
    def download_audio(self):
        """
        Download audio from YouTube without conversion.
        """
        url = self.url_input.text().strip()
        if not url:
            self.show_error("Please enter a YouTube URL.")
            return
        
        if not self.youtube_downloader.validate_youtube_url(url):
            self.show_error("Invalid YouTube URL.")
            return
        
        output_dir = self.output_dir_input.text()
        if output_dir:
            self.youtube_downloader.output_dir = Path(output_dir)
            self.youtube_downloader.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create worker thread for download
        worker = Worker(self.youtube_downloader.download_audio, url)
        worker.signals.started.connect(self.task_started)
        worker.signals.finished.connect(self.task_finished)
        worker.signals.result.connect(self.download_complete)
        worker.signals.error.connect(self.task_error)
        
        self.status_label.setText("Downloading audio...")
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setVisible(True)
        
        # Disable buttons during download
        self.download_button.setEnabled(False)
        self.convert_button.setEnabled(False)
        
        # Start worker
        self.workers.append(worker)
        worker.start()
    
    def download_and_convert(self):
        """
        Download audio from YouTube and convert to selected format.
        """
        url = self.url_input.text().strip()
        if not url:
            self.show_error("Please enter a YouTube URL.")
            return
        
        if not self.youtube_downloader.validate_youtube_url(url):
            self.show_error("Invalid YouTube URL.")
            return
        
        output_dir = self.output_dir_input.text()
        if output_dir:
            self.youtube_downloader.output_dir = Path(output_dir)
            self.youtube_downloader.output_dir.mkdir(exist_ok=True, parents=True)
            self.audio_processor.output_dir = Path(output_dir)
            self.audio_processor.output_dir.mkdir(exist_ok=True, parents=True)
        
        output_format = self.format_combo.currentText()
        bitrate = self.bitrate_combo.currentText()
        
        # Create worker thread for download
        worker = Worker(self.youtube_downloader.download_audio, url)
        worker.signals.started.connect(self.task_started)
        worker.signals.result.connect(
            lambda result: self.convert_downloaded_audio(result, output_format, bitrate)
        )
        worker.signals.error.connect(self.task_error)
        worker.signals.finished.connect(
            lambda: self.status_label.setText("Downloading complete, converting...")
        )
        
        self.status_label.setText("Downloading audio...")
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setVisible(True)
        
        # Disable buttons during download
        self.download_button.setEnabled(False)
        self.convert_button.setEnabled(False)
        
        # Start worker
        self.workers.append(worker)
        worker.start()
    
    def convert_downloaded_audio(
        self, download_result, output_format, bitrate
    ):
        """
        Convert downloaded audio to the specified format.
        
        Args:
            download_result: Tuple of (file_path, title) from download
            output_format: Target audio format
            bitrate: Target audio bitrate
        """
        audio_path, title = download_result
        
        # Create worker thread for conversion
        worker = Worker(
            self.audio_processor.convert_audio,
            audio_path,
            output_format=output_format,
            bitrate=bitrate
        )
        worker.signals.started.connect(lambda: self.status_label.setText("Converting audio..."))
        worker.signals.finished.connect(self.task_finished)
        worker.signals.result.connect(self.conversion_complete)
        worker.signals.error.connect(self.task_error)
        
        # Start worker
        self.workers.append(worker)
        worker.start()
    
    def download_complete(self, result):
        """
        Handle completed download.
        
        Args:
            result: Tuple of (file_path, title) from download
        """
        audio_path, title = result
        self.status_label.setText(f"Download complete: {os.path.basename(audio_path)}")
        
        # Show success message
        QMessageBox.information(
            self, "Download Complete",
            f"Successfully downloaded audio from:\n{title}\n\nSaved to:\n{audio_path}"
        )
        
        # Enable buttons
        self.download_button.setEnabled(True)
        self.convert_button.setEnabled(True)
    
    def conversion_complete(self, result):
        """
        Handle completed conversion.
        
        Args:
            result: Path to converted audio file
        """
        self.status_label.setText(f"Conversion complete: {os.path.basename(result)}")
        
        # Show success message
        QMessageBox.information(
            self, "Conversion Complete",
            f"Successfully converted audio to:\n{result}"
        )
        
        # Enable buttons
        self.download_button.setEnabled(True)
        self.convert_button.setEnabled(True)
    
    def init_voice_cloner(self):
        """
        Initialize the voice cloner if not already done.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.voice_cloner is None:
            try:
                self.voice_cloner = VoiceCloner()
                return True
            except ImportError as e:
                self.show_error(
                    "Voice cloning dependencies not installed.\n\n"
                    "Please install required packages with:\n"
                    "pip install torch torchaudio tortoise-tts"
                )
                return False
        return True
    
    def create_voice_model(self):
        """
        Create a voice model from the selected source.
        """
        # Check if voice cloner is initialized
        if not self.init_voice_cloner():
            return
        
        # Get model name
        model_name = self.model_name_input.text().strip() or None
        
        # Determine source type
        if self.source_type_combo.currentIndex() == 0:  # YouTube URL
            url = self.clone_url_input.text().strip()
            if not url:
                self.show_error("Please enter a YouTube URL.")
                return
            
            if not self.youtube_downloader.validate_youtube_url(url):
                self.show_error("Invalid YouTube URL.")
                return
            
            # Download first, then create model
            self.status_label.setText("Downloading audio...")
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            self.progress_bar.setVisible(True)
            
            # Create worker thread for download
            download_worker = Worker(self.youtube_downloader.download_audio, url)
            download_worker.signals.started.connect(self.task_started)
            download_worker.signals.result.connect(
                lambda result: self.process_downloaded_for_cloning(result, model_name)
            )
            download_worker.signals.error.connect(self.task_error)
            download_worker.signals.finished.connect(
                lambda: self.status_label.setText("Download complete, creating voice model...")
            )
            
            # Disable button during download
            self.create_model_button.setEnabled(False)
            
            # Start worker
            self.workers.append(download_worker)
            download_worker.start()
            
        else:  # Local file
            file_path = self.local_file_input.text().strip()
            if not file_path or not os.path.exists(file_path):
                self.show_error("Please select a valid audio file.")
                return
            
            # Create model directly
            self.create_voice_model_from_file(file_path, model_name)
    
    def process_downloaded_for_cloning(self, download_result, model_name):
        """
        Process downloaded audio for voice cloning.
        
        Args:
            download_result: Tuple of (file_path, title) from download
            model_name: Name for the voice model, or None to use video title
        """
        audio_path, title = download_result
        
        # Use video title as model name if none provided
        if model_name is None:
            model_name = title
        
        # Create voice model
        self.create_voice_model_from_file(audio_path, model_name)
    
    def create_voice_model_from_file(self, audio_path, model_name):
        """
        Create a voice model from an audio file.
        
        Args:
            audio_path: Path to the audio file
            model_name: Name for the voice model, or None to use filename
        """
        # Validate audio for voice cloning
        is_valid, reason = self.voice_cloner.validate_audio_for_cloning(audio_path)
        if not is_valid:
            self.show_error(f"Audio not suitable for voice cloning: {reason}")
            self.create_model_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            return
        
        # Create worker thread for model creation
        worker = Worker(
            self.voice_cloner.create_voice_model,
            audio_path,
            model_name=model_name
        )
        worker.signals.started.connect(lambda: self.status_label.setText("Creating voice model..."))
        worker.signals.finished.connect(self.task_finished)
        worker.signals.result.connect(self.voice_model_created)
        worker.signals.error.connect(self.task_error)
        
        # Start worker
        self.workers.append(worker)
        worker.start()
    
    def voice_model_created(self, model):
        """
        Handle completed voice model creation.
        
        Args:
            model: The created voice model
        """
        self.status_label.setText(f"Voice model created: {model.name}")
        
        # Show success message
        QMessageBox.information(
            self, "Voice Model Created",
            f"Successfully created voice model:\n{model.name}\n\nSaved to:\n{model.path}"
        )
        
        # Enable button
        self.create_model_button.setEnabled(True)
        
        # Refresh voice models list
        self.refresh_voice_models()
    
    def refresh_voice_models(self):
        """
        Refresh the list of available voice models.
        """
        # Check if voice cloner is initialized
        if not self.init_voice_cloner():
            return
        
        # Create worker thread for listing models
        worker = Worker(self.voice_cloner.list_voice_models)
        worker.signals.started.connect(lambda: self.status_label.setText("Refreshing voice models..."))
        worker.signals.finished.connect(lambda: self.status_label.setText("Ready"))
        worker.signals.result.connect(self.display_voice_models)
        worker.signals.error.connect(self.task_error)
        
        # Start worker
        self.workers.append(worker)
        worker.start()
    
    def display_voice_models(self, models):
        """
        Display the list of voice models in the text view.
        
        Args:
            models: List of voice models
        """
        if not models:
            self.models_text.setText("No voice models found.")
            return
        
        # Format the models list
        models_text = f"Found {len(models)} voice models:\n\n"
        
        for i, model in enumerate(models, 1):
            models_text += f"{i}. {model.name}\n"
            models_text += f"   Created: {time.ctime(model.created_at)}\n"
            models_text += f"   Duration: {model.duration:.1f} seconds\n"
            if model.source_file:
                models_text += f"   Source: {model.source_file}\n"
            models_text += "\n"
        
        self.models_text.setText(models_text)
    
    def delete_voice_model(self):
        """
        Delete the selected voice model.
        """
        # Check if voice cloner is initialized
        if not self.init_voice_cloner():
            return
        
        # Get selected model from text (this is a simplification)
        selected_text = self.models_text.textCursor().selectedText()
        if not selected_text:
            self.show_error("Please select a model name in the list to delete.")
            return
        
        # Try to extract model name from selection
        model_name = None
        for line in selected_text.split('\n'):
            if line.strip().endswith('.'):  # Model name line
                model_name = line.split('.', 1)[1].strip()
                break
        
        if not model_name:
            # Try another approach - just use the first word if it's not a number
            words = selected_text.split()
            if words and not words[0].isdigit():
                model_name = words[0]
        
        if not model_name:
            self.show_error("Could not determine which model to delete. Please select the model name.")
            return
        
        # Confirm deletion
        reply = QMessageBox.question(
            self, "Confirm Deletion",
            f"Are you sure you want to delete the voice model '{model_name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Create worker thread for deletion
            worker = Worker(self.voice_cloner.delete_voice_model, model_name)
            worker.signals.started.connect(lambda: self.status_label.setText(f"Deleting voice model: {model_name}..."))
            worker.signals.finished.connect(lambda: self.status_label.setText("Ready"))
            worker.signals.result.connect(
                lambda success: self.model_deletion_complete(success, model_name)
            )
            worker.signals.error.connect(self.task_error)
            
            # Start worker
            self.workers.append(worker)
            worker.start()
    
    def model_deletion_complete(self, success, model_name):
        """
        Handle completed voice model deletion.
        
        Args:
            success: Whether deletion was successful
            model_name: Name of the deleted model
        """
        if success:
            self.status_label.setText(f"Deleted voice model: {model_name}")
            # Refresh models list
            self.refresh_voice_models()
        else:
            self.show_error(f"Failed to delete voice model: {model_name}")
    
    def refresh_synthesis_voices(self):
        """
        Refresh the list of available voice models for synthesis.
        """
        # Check if voice cloner is initialized
        if not self.init_voice_cloner():
            return
        
        # Create worker thread for listing models
        worker = Worker(self.voice_cloner.list_voice_models)
        worker.signals.started.connect(lambda: self.status_label.setText("Refreshing voice models..."))
        worker.signals.finished.connect(lambda: self.status_label.setText("Ready"))
        worker.signals.result.connect(self.populate_synthesis_voices)
        worker.signals.error.connect(self.task_error)
        
        # Start worker
        self.workers.append(worker)
        worker.start()
    
    def populate_synthesis_voices(self, models):
        """
        Populate the voice model dropdown for synthesis.
        
        Args:
            models: List of voice models
        """
        self.voice_model_combo.clear()
        
        if not models:
            self.voice_model_combo.addItem("No voice models available")
            self.synthesize_button.setEnabled(False)
            return
        
        for model in models:
            self.voice_model_combo.addItem(model.name)
        
        self.synthesize_button.setEnabled(True)
    
    def synthesize_speech(self):
        """
        Synthesize speech from text using the selected voice model.
        """
        # Check if voice cloner is initialized
        if not self.init_voice_cloner():
            return
        
        # Get text to synthesize
        text = self.synthesis_text.toPlainText().strip()
        if not text:
            self.show_error("Please enter text to synthesize.")
            return
        
        # Get voice model
        voice_model = self.voice_model_combo.currentText()
        if not voice_model or voice_model == "No voice models available":
            self.show_error("Please select a voice model.")
            return
        
        # Get synthesis options
        preset = self.quality_combo.currentText()
        seed = self.seed_spin.value()
        output_path = self.synthesis_output_input.text().strip() or None
        
        # Create worker thread for synthesis
        worker = Worker(
            self.voice_cloner.synthesize_speech,
            text,
            voice_model,
            output_path=output_path,
            preset=preset,
            seed=seed
        )
        worker.signals.started.connect(self.task_started)
        worker.signals.finished.connect(self.task_finished)
        worker.signals.result.connect(self.synthesis_complete)
        worker.signals.error.connect(self.task_error)
        
        self.status_label.setText("Synthesizing speech...")
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setVisible(True)
        
        # Disable button during synthesis
        self.synthesize_button.setEnabled(False)
        
        # Start worker
        self.workers.append(worker)
        worker.start()
    
    def synthesis_complete(self, output_file):
        """
        Handle completed speech synthesis.
        
        Args:
            output_file: Path to synthesized audio file
        """
        self.status_label.setText(f"Synthesis complete: {os.path.basename(output_file)}")
        
        # Show success message
        QMessageBox.information(
            self, "Synthesis Complete",
            f"Successfully synthesized speech to:\n{output_file}"
        )
        
        # Enable button
        self.synthesize_button.setEnabled(True)
    
    def task_started(self):
        """
        Handle task start.
        """
        self.progress_bar.setVisible(True)
    
    def task_finished(self):
        """
        Handle task completion.
        """
        self.progress_bar.setVisible(False)
        
        # Re-enable buttons
        self.download_button.setEnabled(True)
        self.convert_button.setEnabled(True)
        self.create_model_button.setEnabled(True)
        self.synthesize_button.setEnabled(True)
        
        # Clean up completed workers
        self.workers = [w for w in self.workers if w.isRunning()]
    
    def task_error(self, error_message):
        """
        Handle task error.
        
        Args:
            error_message: Error message from the worker
        """
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {error_message}")
        
        # Show error message
        self.show_error(error_message)
        
        # Re-enable buttons
        self.download_button.setEnabled(True)
        self.convert_button.setEnabled(True)
        self.create_model_button.setEnabled(True)
        self.synthesize_button.setEnabled(True)
        
        # Clean up completed workers
        self.workers = [w for w in self.workers if w.isRunning()]
    
    def show_error(self, message):
        """
        Show an error message dialog.
        
        Args:
            message: Error message to display
        """
        QMessageBox.critical(self, "Error", message)
    
    def show_about_dialog(self):
        """
        Show the about dialog.
        """
        QMessageBox.about(
            self, "About YouTube to MP3 & Voice Cloning",
            "YouTube to MP3 & Voice Cloning\n\n"
            "An application for downloading YouTube audio, "
            "converting to different formats, and creating/using "
            "voice models for speech synthesis.\n\n"
            "© 2025 Example Corp."
        )
    
    def set_theme(self, theme):
        """
        Set the application theme.
        
        Args:
            theme: Theme to apply (from Theme enum)
        """
        # Update check marks in menu
        for t, action in self.theme_actions.items():
            action.setChecked(t == theme)
        
        app = QApplication.instance()
        
        if theme == Theme.LIGHT:
            app.setStyle("Fusion")
            app.setPalette(QApplication.style().standardPalette())
        elif theme == Theme.DARK:
            # Dark mode with Fusion style
            app.setStyle("Fusion")
            
            # Dark palette based on Qt Fusion dark
            dark_palette = QPalette()
            
            # Base colors
            dark_color = QColor(45, 45, 45)
            dark_color2 = QColor(53, 53, 53)
            light_color = QColor(200, 200, 200)
            
            # Base palette
            dark_palette.setColor(QPalette.Window, dark_color)
            dark_palette.setColor(QPalette.WindowText, light_color)
            dark_palette.setColor(QPalette.Base, dark_color2)
            dark_palette.setColor(QPalette.AlternateBase, dark_color)
            dark_palette.setColor(QPalette.ToolTipBase, dark_color)
            dark_palette.setColor(QPalette.ToolTipText, light_color)
            dark_palette.setColor(QPalette.Text, light_color)
            dark_palette.setColor(QPalette.Button, dark_color)
            dark_palette.setColor(QPalette.ButtonText, light_color)
            dark_palette.setColor(QPalette.BrightText, Qt.red)
            dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.HighlightedText, dark_color)
            
            # Disabled
            dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(120, 120, 120))
            dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 120, 120))
            dark_palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 80))
            dark_palette.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor(120, 120, 120))
            
            app.setPalette(dark_palette)
        else:  # System
            app.setStyle("")
            app.setPalette(QApplication.style().standardPalette())


def main():
    """
    Main entry point for the GUI application.
    """
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Use Fusion style as base
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
