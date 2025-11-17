# gui.py
# ==============================================================================
# PyQt Control Panel for Real-Time ASR (Tabbed Version)
# ==============================================================================
#
# Purpose:
# - Provides a tabbed GUI for changing application settings at runtime.
# - Tab 1 "Real-Time": Contains critical settings (Audio, WSL)
# - Tab 2 "General Settings": Contains behavior, hotkeys, and timeouts.
# ==============================================================================

import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QFormLayout,
    QComboBox, QCheckBox, QPushButton, QLabel, QLineEdit,
    QSpinBox, QGroupBox, QTabWidget
)
from PyQt5.QtCore import pyqtSlot, QObject

# Import logger
try:
    from utils.loggerInstance import uniLogger
except ImportError:
    # Fallback logger if utils are not available
    import logging

    uniLogger = logging.getLogger("GUI_Fallback")

# --- Dummy Classes for Type Hinting ---
try:
    from managers import ConfigurationManager
    from audioProcesses import AudioHandler
except ImportError:
    print("[GUI] Warning: Using dummy types for ConfigurationManager/AudioHandler.")


    class ConfigurationManager(QObject):
        def get(self, key, default=None): return default

        def set(self, key, value): pass


    class AudioHandler(QObject):
        def listAudioDevices(self): return {}


class ControlPanel(QWidget):
    """
    A PyQt Control Panel to change application settings.
    It interacts with the ConfigurationManager instance passed to it.
    """

    def __init__(self, config: ConfigurationManager, audio_handler: AudioHandler, parent=None):
        super().__init__(parent)
        self.config = config
        self.audio_handler = audio_handler

        if not hasattr(self.config, 'set') or not hasattr(self.config, 'get'):
            raise ValueError("GUI Error: 'config' object lacks 'set' or 'get' methods.")

        if not self.audio_handler:
            uniLogger.error(
                "[GUI] CRITICAL: AudioHandler object is None. Device list will be empty.")
            self.audio_handler = AudioHandler()  # Use dummy class to prevent crash
        elif not hasattr(self.audio_handler, 'listAudioDevices'):
            uniLogger.warning("[GUI] AudioHandler is missing 'listAudioDevices'. Using fallback.")

        self.initUI()
        self.load_initial_settings()
        uniLogger.info("[GUI] Control Panel initialized with Tab structure.")

    def initUI(self):
        """Initializes the user interface elements with a Tab Widget."""
        self.setWindowTitle("ASR Control Panel")
        main_layout = QVBoxLayout(self)

        # --- Create Tab Widget ---
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # --- Create "Real-Time" Tab ---
        self.realtime_tab = QWidget()
        realtime_layout = QVBoxLayout(self.realtime_tab)
        self.createRealTimeTab(realtime_layout)  # Populate this tab
        self.tab_widget.addTab(self.realtime_tab, "Real-Time")

        # --- Create "General Settings" Tab ---
        self.general_tab = QWidget()
        general_layout = QVBoxLayout(self.general_tab)
        self.createGeneralSettingsTab(general_layout)  # Populate this tab
        self.tab_widget.addTab(self.general_tab, "General Settings")

        # --- (Future) Add "Transcribe from File" Tab ---
        # self.file_tab = QWidget()
        # file_layout = QVBoxLayout(self.file_tab)
        # file_layout.addWidget(QLabel("Settings for file transcription will go here."))
        # self.tab_widget.addTab(self.file_tab, "Transcribe from File")

        # --- Apply Button (Outside the tabs) ---
        self.apply_button = QPushButton("Apply Settings")
        self.apply_button.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #4CAF50; color: white; }"
        )
        self.apply_button.clicked.connect(self.apply_settings)
        main_layout.addWidget(self.apply_button)

        self.setLayout(main_layout)
        self.resize(450, 450)  # Made it taller to fit all settings

    def createRealTimeTab(self, layout: QVBoxLayout):
        """Populates the "Real-Time" tab with critical settings."""
        # --- Audio Settings Group ---
        audio_group = QGroupBox("Audio Settings (Requires Stream Restart)")
        audio_layout = QFormLayout()
        self.device_combo = QComboBox()
        self.populateAudioDevices()
        audio_layout.addRow(QLabel("Audio Device:"), self.device_combo)
        self.rate_spinbox = QSpinBox()
        self.rate_spinbox.setRange(8000, 48000)
        self.rate_spinbox.setSingleStep(1000)
        self.rate_spinbox.setSuffix(" Hz")
        audio_layout.addRow(QLabel("Sample Rate:"), self.rate_spinbox)
        self.channels_spinbox = QSpinBox()
        self.channels_spinbox.setRange(1, 2)
        audio_layout.addRow(QLabel("Channels:"), self.channels_spinbox)
        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)

        # --- WSL & Model Settings Group ---
        wsl_group = QGroupBox("WSL & Model Settings (Requires Backend Restart)")
        wsl_layout = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems([
            "nvidia/canary-180m-flash",
            "nvidia/stt_en_fastconformer_ctc_small",
            "openai/whisper-large-v3",
            "openai/whisper-base",
            "openai/whisper-tiny"
        ])
        wsl_layout.addRow(QLabel("ASR Model:"), self.model_combo)
        self.wsl_url_edit = QLineEdit()
        wsl_layout.addRow(QLabel("WSL Server URL:"), self.wsl_url_edit)
        self.wsl_distro_edit = QLineEdit()
        wsl_layout.addRow(QLabel("WSL Distro Name:"), self.wsl_distro_edit)
        self.wsl_sudo_check = QCheckBox("Use 'sudo' for WSL Server")
        wsl_layout.addRow(self.wsl_sudo_check)
        wsl_group.setLayout(wsl_layout)
        layout.addWidget(wsl_group)

        layout.addStretch()  # Add stretch to push groups to the top

    def createGeneralSettingsTab(self, layout: QVBoxLayout):
        """Populates the "General Settings" tab with non-critical settings."""

        # --- Behavior Group ---
        behavior_group = QGroupBox("Behavior")
        behavior_layout = QFormLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["dictationMode", "constantIntervalMode"])
        behavior_layout.addRow(QLabel("Transcription Mode:"), self.mode_combo)
        self.lang_edit = QLineEdit()
        behavior_layout.addRow(QLabel("Language Code:"), self.lang_edit)
        self.typing_check = QCheckBox("Enable Typing/Clipboard Output")
        behavior_layout.addRow(self.typing_check)
        behavior_group.setLayout(behavior_layout)
        layout.addWidget(behavior_group)

        # --- Hotkeys Group ---
        hotkey_group = QGroupBox("Hotkeys (Requires App Restart to Apply)")
        hotkey_layout = QFormLayout()
        self.hotkey_rec = QLineEdit()
        hotkey_layout.addRow(QLabel("Toggle Recording:"), self.hotkey_rec)
        self.hotkey_out = QLineEdit()
        hotkey_layout.addRow(QLabel("Toggle Output:"), self.hotkey_out)
        self.hotkey_force = QLineEdit()
        hotkey_layout.addRow(QLabel("Force Transcribe:"), self.hotkey_force)
        hotkey_group.setLayout(hotkey_layout)
        layout.addWidget(hotkey_group)

        # --- Timeouts Group ---
        timeout_group = QGroupBox("Timeouts (seconds)")
        timeout_layout = QFormLayout()
        self.timeout_model = QSpinBox()
        self.timeout_model.setRange(0, 99999)  # 0 = disabled
        self.timeout_model.setSingleStep(60)
        timeout_layout.addRow(QLabel("Unload Model After:"), self.timeout_model)
        self.timeout_idle = QSpinBox()
        self.timeout_idle.setRange(0, 99999)  # 0 = disabled
        self.timeout_idle.setSingleStep(60)
        timeout_layout.addRow(QLabel("Stop Recording After Idle:"), self.timeout_idle)
        timeout_group.setLayout(timeout_layout)
        layout.addWidget(timeout_group)

        layout.addStretch()  # Add stretch to push groups to the top

    def populateAudioDevices(self):
        """Populates the audio device ComboBox by calling the AudioHandler."""
        self.device_combo.clear()
        try:
            devices = self.audio_handler.listAudioDevices()
            if devices:
                uniLogger.info(f"[GUI] Found {len(devices)} audio devices.")
                self.device_combo.addItem("Default Device", None)
                for idx, name in devices.items():
                    self.device_combo.addItem(f"{name}", idx)
                return
            else:
                uniLogger.warning("[GUI] listAudioDevices() returned empty.")
        except Exception as e:
            uniLogger.error(
                f"[GUI] Could not list audio devices from handler: {e}. Using fallback.",
                excInfo=True)

        if self.device_combo.count() == 0:
            uniLogger.warning("[GUI] Using fallback device list.")
            self.device_combo.addItem("Default Device", None)

    def load_initial_settings(self):
        """Loads the current settings from the config object into the widgets."""
        try:
            # --- Real-Time Tab ---
            device_id = self.config.get('deviceId', None)
            index = self.device_combo.findData(device_id)
            if index != -1: self.device_combo.setCurrentIndex(index)
            self.rate_spinbox.setValue(int(self.config.get('sampleRate', 16000)))
            self.channels_spinbox.setValue(int(self.config.get('channels', 1)))
            model_name = self.config.get('modelName', '')
            index = self.model_combo.findText(model_name)
            if index != -1:
                self.model_combo.setCurrentIndex(index)
            else:
                self.model_combo.addItem(model_name)
                self.model_combo.setCurrentText(model_name)
            self.wsl_url_edit.setText(self.config.get('wslServerUrl', 'http://localhost:5001'))
            self.wsl_distro_edit.setText(self.config.get('wslDistributionName', 'Ubuntu'))
            self.wsl_sudo_check.setChecked(bool(self.config.get('wslUseSudo', False)))

            # --- General Settings Tab ---
            self.mode_combo.setCurrentText(self.config.get('transcriptionMode', 'dictationMode'))
            self.lang_edit.setText(self.config.get('language', 'en'))
            self.typing_check.setChecked(bool(self.config.get('enableTypingOutput', True)))
            self.hotkey_rec.setText(self.config.get('recordingToggleKey', ''))
            self.hotkey_out.setText(self.config.get('outputToggleKey', ''))
            self.hotkey_force.setText(self.config.get('forceTranscriptionKey', ''))
            self.timeout_model.setValue(int(self.config.get('model_unloadTimeout', 600)))
            self.timeout_idle.setValue(int(self.config.get('consecutiveIdleTime', 120)))

            uniLogger.info("[GUI] Initial settings loaded into all tabs.")

        except Exception as e:
            uniLogger.error(f"[GUI] Error loading initial settings: {e}", excInfo=True)

    @pyqtSlot()
    def apply_settings(self):
        """
        Saves all widget values back to the shared config object.
        This will trigger the appropriate flags in ConfigurationManager.
        """
        uniLogger.info("[GUI] Apply button clicked. Saving settings from all tabs...")

        try:
            # --- Real-Time Tab Settings ---
            self.config.set('deviceId', self.device_combo.currentData())
            self.config.set('sampleRate', self.rate_spinbox.value())
            self.config.set('channels', self.channels_spinbox.value())
            self.config.set('modelName', self.model_combo.currentText())
            self.config.set('wslServerUrl', self.wsl_url_edit.text())
            self.config.set('wslDistributionName', self.wsl_distro_edit.text())
            self.config.set('wslUseSudo', self.wsl_sudo_check.isChecked())

            # --- General Settings Tab Settings ---
            self.config.set('transcriptionMode', self.mode_combo.currentText())
            self.config.set('language', self.lang_edit.text())
            self.config.set('enableTypingOutput', self.typing_check.isChecked())
            self.config.set('recordingToggleKey', self.hotkey_rec.text())
            self.config.set('outputToggleKey', self.hotkey_out.text())
            self.config.set('forceTranscriptionKey', self.hotkey_force.text())
            self.config.set('model_unloadTimeout', self.timeout_model.value())
            self.config.set('consecutiveIdleTime', self.timeout_idle.value())

            uniLogger.info("[GUI] Settings successfully applied to config object.")
            uniLogger.info("[GUI] Orchestrator will now detect changes in its main loop.")

        except Exception as e:
            uniLogger.error(f"[GUI] Error applying settings: {e}", excInfo=True)