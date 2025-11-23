# gui.py
# ==============================================================================
# QtPy Control Panel for Real-Time ASR (Ultimate Version + Dynamic Logic)
# ==============================================================================
#
# Purpose:
# - Provides a GUI for changing ALL application settings found in userSettings.
# - Features Hover Tooltips (?) for every single parameter.
# - Organized into 3 Tabs.
# - [NEW] Dynamic UI: Disables/Enables fields based on dependencies.
# ==============================================================================

import sys
from qtpy.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QGridLayout,
    QComboBox, QCheckBox, QPushButton, QLabel, QLineEdit,
    QSpinBox, QDoubleSpinBox, QGroupBox, QTabWidget, QScrollArea, QHBoxLayout
)
from qtpy.QtCore import Qt, Slot, QObject
from qtpy.QtGui import QCursor, QFont

# --- Import Logger ---
try:
    from utils.loggerInstance import uniLogger
except ImportError:
    import logging

    uniLogger = logging.getLogger("GUI_Fallback")
    logging.basicConfig(level=logging.INFO)

# --- Dummy Classes for Type Hinting ---
try:
    from managers import ConfigurationManager
    from audioProcesses import AudioHandler
except ImportError:
    class ConfigurationManager(QObject):
        def get(self, key, default=None): return default

        def set(self, key, value): pass


    class AudioHandler(QObject):
        def listAudioDevices(self): return {}


class ControlPanel(QWidget):
    def __init__(self, config: ConfigurationManager, audio_handler: AudioHandler, parent=None):
        super().__init__(parent)
        self.config = config
        self.audio_handler = audio_handler

        # --- Validation ---
        if not hasattr(self.config, 'set') or not hasattr(self.config, 'get'):
            uniLogger.warning("[GUI] Config object lacks 'set' or 'get' methods.")

        if not self.audio_handler:
            uniLogger.error(
                "[GUI] CRITICAL: AudioHandler object is None. Device list will be empty.")
            self.audio_handler = AudioHandler()
        elif not hasattr(self.audio_handler, 'listAudioDevices'):
            uniLogger.warning("[GUI] AudioHandler is missing 'listAudioDevices'. Using fallback.")

        # --- Styling ---
        self.setStyleSheet("""
            QToolTip {
                background-color: #212121;
                color: #ffffff;
                border: 1px solid #000;
                padding: 5px;
                border-radius: 4px;
                font-family: 'Segoe UI', sans-serif;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #bbb;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #333;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                min-height: 24px;
                padding-left: 4px;
            }
            /* Disabled widget styling */
            QWidget:disabled {
                color: #888;
                background-color: #f0f0f0;
            }
        """)

        self.initUI()
        self.connect_signals()  # [NEW] Connect events
        self.load_initial_settings()
        uniLogger.info("[GUI] Control Panel initialized with Dynamic Logic.")

    def initUI(self):
        self.setWindowTitle("ASR Control Panel")
        main_layout = QVBoxLayout(self)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # --- Tab 1: Connection & Audio ---
        self.tab_conn = QWidget()
        self.createConnectionTab(QVBoxLayout(self.tab_conn))
        self.tab_widget.addTab(self.tab_conn, "Connection & Audio")

        # --- Tab 2: Logic & Filtering ---
        self.tab_logic = QWidget()
        self.createLogicTab(QVBoxLayout(self.tab_logic))
        self.tab_widget.addTab(self.tab_logic, "Logic & Filtering")

        # --- Tab 3: General & Behavior ---
        self.tab_gen = QWidget()
        self.createGeneralTab(QVBoxLayout(self.tab_gen))
        self.tab_widget.addTab(self.tab_gen, "General & Behavior")

        # --- Apply Button ---
        self.apply_button = QPushButton("Apply All Settings")
        self.apply_button.setCursor(Qt.PointingHandCursor)
        self.apply_button.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #0078D7; color: white; padding: 10px; border-radius: 5px; } QPushButton:hover { background-color: #005a9e; }")
        self.apply_button.clicked.connect(self.apply_settings)
        main_layout.addWidget(self.apply_button)

        self.setLayout(main_layout)
        self.resize(550, 700)

    # ==========================================================================
    # Tab 1: Connection, Audio & Model
    # ==========================================================================
    def createConnectionTab(self, layout: QVBoxLayout):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)

        # --- Audio Settings ---
        grp_audio = QGroupBox("Audio Configuration")
        grid_audio = QGridLayout()

        self.device_combo = QComboBox()
        self.populateAudioDevices()
        self.add_row(grid_audio, 0, "Audio Device", "Input device ID.", self.device_combo)

        self.rate_spin = QSpinBox()
        self.rate_spin.setRange(8000, 48000)
        self.rate_spin.setSingleStep(1000)
        self.add_row(grid_audio, 1, "Sample Rate (Hz)", "e.g., 16000 for NeMo/Whisper.",
                     self.rate_spin)

        self.channels_spin = QSpinBox()
        self.channels_spin.setRange(1, 2)
        self.add_row(grid_audio, 2, "Channels", "1 = Mono (Recommended), 2 = Stereo.",
                     self.channels_spin)

        self.block_spin = QSpinBox()
        self.block_spin.setRange(256, 8192)
        self.block_spin.setValue(1024)
        self.add_row(grid_audio, 3, "Block Size", "Audio chunk size (latency vs CPU).",
                     self.block_spin)

        grp_audio.setLayout(grid_audio)
        content_layout.addWidget(grp_audio)

        # --- Model Settings ---
        grp_model = QGroupBox("Core Model Settings")
        grid_model = QGridLayout()

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems([
            "nvidia/canary-180m-flash",
            "nvidia/parakeet-rnnt-1.1b",
            "openai/whisper-large-v3",
            "openai/whisper-medium.en",
            "openai/whisper-tiny.en"
        ])
        self.add_row(grid_model, 0, "Model Name", "Remote NeMo or Local Whisper model name.",
                     self.model_combo)

        self.lang_edit = QLineEdit()
        self.lang_edit.setPlaceholderText("e.g. en, fa")
        self.add_row(grid_model, 1, "Language", "Target language code (or None for auto).",
                     self.lang_edit)

        self.chk_cpu = QCheckBox("Force CPU (Local Whisper only)")
        help_cpu = self.create_help_icon("Overrides GPU detection for local models.")
        grid_model.addWidget(self.chk_cpu, 2, 0, 1, 2)
        grid_model.addWidget(help_cpu, 2, 2)

        grp_model.setLayout(grid_model)
        content_layout.addWidget(grp_model)

        # --- Remote Server (WSL) ---
        self.grp_wsl = QGroupBox("Remote Server (WSL Settings)")  # Store reference to disable later
        grid_wsl = QGridLayout()

        self.wsl_url_edit = QLineEdit()
        self.add_row(grid_wsl, 0, "WSL Server URL", "URL of wslNemoServer.py.", self.wsl_url_edit)

        self.wsl_distro_edit = QLineEdit()
        self.add_row(grid_wsl, 1, "WSL Distro Name", "e.g., Ubuntu-22.04.", self.wsl_distro_edit)

        self.wsl_req_timeout = QDoubleSpinBox()
        self.wsl_req_timeout.setRange(1.0, 120.0)
        self.add_row(grid_wsl, 2, "Request Timeout (s)", "Max wait time for server response.",
                     self.wsl_req_timeout)

        self.wsl_ready_timeout = QDoubleSpinBox()
        self.wsl_ready_timeout.setRange(10.0, 300.0)
        self.add_row(grid_wsl, 3, "Ready Timeout (s)", "Max wait for server startup.",
                     self.wsl_ready_timeout)

        self.chk_sudo = QCheckBox("Use 'sudo'")
        self.chk_unload_exit = QCheckBox("Unload Remote Model on Exit")

        hbox_checks = QHBoxLayout()
        hbox_checks.addWidget(self.chk_sudo)
        hbox_checks.addWidget(self.create_help_icon("Run server with sudo?"))
        hbox_checks.addWidget(self.chk_unload_exit)
        hbox_checks.addWidget(self.create_help_icon("Free VRAM when app closes?"))
        grid_wsl.addLayout(hbox_checks, 4, 0, 1, 3)

        self.grp_wsl.setLayout(grid_wsl)
        content_layout.addWidget(self.grp_wsl)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    # ==========================================================================
    # Tab 2: Logic & Filtering
    # ==========================================================================
    def createLogicTab(self, layout: QVBoxLayout):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)

        # --- Transcription Modes ---
        grp_mode = QGroupBox("Transcription Mode")
        grid_mode = QGridLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["dictationMode", "constantIntervalMode"])
        self.add_row(grid_mode, 0, "Mode", "Dictation (pause-based) or Constant Interval.",
                     self.mode_combo)

        # Store refs for dynamic enabling
        self.lbl_dict_1 = QLabel("Dictation Silence Duration (s):")
        self.dict_silence_dur = QDoubleSpinBox();
        self.dict_silence_dur.setSingleStep(0.1)

        self.lbl_dict_2 = QLabel("Dictation Loudness Thresh:")
        self.dict_loudness = QDoubleSpinBox();
        self.dict_loudness.setDecimals(5);
        self.dict_loudness.setSingleStep(0.0001)

        self.lbl_int_1 = QLabel("Interval Duration (s):")
        self.const_interval = QDoubleSpinBox()

        # Add to grid
        grid_mode.addWidget(self.lbl_dict_1, 1, 0);
        grid_mode.addWidget(self.dict_silence_dur, 1, 1)
        grid_mode.addWidget(self.lbl_dict_2, 2, 0);
        grid_mode.addWidget(self.dict_loudness, 2, 1)
        grid_mode.addWidget(self.lbl_int_1, 3, 0);
        grid_mode.addWidget(self.const_interval, 3, 1)

        grp_mode.setLayout(grid_mode)
        content_layout.addWidget(grp_mode)

        # --- Silence Skipping & Filtering ---
        grp_filter = QGroupBox("Silence Skipping & Advanced Filtering")
        grid_filter = QGridLayout()

        self.min_loud_dur = QDoubleSpinBox()
        self.min_loud_dur.setSingleStep(0.1)
        self.add_row(grid_filter, 0, "Min Loud Duration (s)", "Segment must be loud for this long.",
                     self.min_loud_dur)

        self.skip_thresh = QDoubleSpinBox()
        self.skip_thresh.setDecimals(5)
        self.add_row(grid_filter, 1, "Silence Skip Threshold",
                     "Avg loudness below this is skipped.", self.skip_thresh)

        self.skip_before = QDoubleSpinBox()
        self.add_row(grid_filter, 2, "Check Start N Sec", "Check loudness at start of segment.",
                     self.skip_before)

        self.skip_after = QDoubleSpinBox()
        self.add_row(grid_filter, 3, "Check End N Sec", "Check loudness at end of segment.",
                     self.skip_after)

        self.word_thresh = QDoubleSpinBox()
        self.word_thresh.setDecimals(5)
        self.add_row(grid_filter, 4, "Common Word Loudness", "Threshold for filtering short words.",
                     self.word_thresh)

        self.common_words = QLineEdit()
        self.add_row(grid_filter, 5, "Common False Words", "Comma separated (e.g. uh, um, okay).",
                     self.common_words)

        self.banned_words = QLineEdit()
        self.add_row(grid_filter, 6, "Banned Words", "Always removed (e.g. <|endoftext|>).",
                     self.banned_words)

        grp_filter.setLayout(grid_filter)
        content_layout.addWidget(grp_filter)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    # ==========================================================================
    # Tab 3: General & Behavior
    # ==========================================================================
    def createGeneralTab(self, layout: QVBoxLayout):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)

        # --- Behavior Switches ---
        grp_beh = QGroupBox("General Behavior")
        grid_beh = QGridLayout()

        self.chk_dots = QCheckBox("Remove Trailing Dots")
        self.chk_start_out = QCheckBox("Output Enabled on Start")
        self.chk_start_rec = QCheckBox("Recording Active on Start")
        self.chk_typing = QCheckBox("Enable Typing Output")

        grid_beh.addWidget(self.chk_dots, 0, 0);
        grid_beh.addWidget(self.create_help_icon("Remove . or ... at end"), 0, 1)
        grid_beh.addWidget(self.chk_start_out, 1, 0);
        grid_beh.addWidget(self.create_help_icon("Start app with output active?"), 1, 1)
        grid_beh.addWidget(self.chk_start_rec, 2, 0);
        grid_beh.addWidget(self.create_help_icon("Start app with mic active?"), 2, 1)
        grid_beh.addWidget(self.chk_typing, 3, 0);
        grid_beh.addWidget(self.create_help_icon("Global typing switch"), 3, 1)

        self.combo_type_mode = QComboBox()
        self.combo_type_mode.addItems(["whole", "word", "letter"])
        self.add_row(grid_beh, 4, "Typing Mode", "How text is typed into windows.",
                     self.combo_type_mode)

        grp_beh.setLayout(grid_beh)
        content_layout.addWidget(grp_beh)

        # --- Audio Notifications ---
        grp_sound = QGroupBox("Audio Notifications")
        grid_sound = QGridLayout()
        self.chk_notify = QCheckBox("Enable Audio Notifications")
        self.chk_play_enable = QCheckBox("Play 'Enabled' Sounds")  # Depends on notify
        grid_sound.addWidget(self.chk_notify, 0, 0)
        grid_sound.addWidget(self.chk_play_enable, 1, 0)
        grp_sound.setLayout(grid_sound)
        content_layout.addWidget(grp_sound)

        # --- Hotkeys ---
        grp_hot = QGroupBox("Hotkeys (Global)")
        grid_hot = QGridLayout()
        self.key_rec = QLineEdit();
        self.add_row(grid_hot, 0, "Toggle Recording", "Win+Alt+L etc.", self.key_rec)
        self.key_out = QLineEdit();
        self.add_row(grid_hot, 1, "Toggle Output", "Ctrl+Q etc.", self.key_out)
        self.key_force = QLineEdit();
        self.add_row(grid_hot, 2, "Force Transcribe", "Ctrl+. etc.", self.key_force)
        grp_hot.setLayout(grid_hot)
        content_layout.addWidget(grp_hot)

        # --- Timeouts ---
        grp_time = QGroupBox("Timeouts & Limits (Seconds, 0=Disable)")
        grid_time = QGridLayout()
        self.time_unload = QSpinBox();
        self.time_unload.setRange(0, 99999)
        self.add_row(grid_time, 0, "Model Unload Timeout", "Unload model after N sec idle.",
                     self.time_unload)

        self.time_idle = QSpinBox();
        self.time_idle.setRange(0, 99999)
        self.add_row(grid_time, 1, "Stop Rec. Idle Time", "Stop mic after N sec silence.",
                     self.time_idle)

        self.max_rec = QSpinBox();
        self.max_rec.setRange(0, 99999)
        self.add_row(grid_time, 2, "Max Recording Duration", "Limit single session length.",
                     self.max_rec)

        self.max_prog = QSpinBox();
        self.max_prog.setRange(0, 99999)
        self.add_row(grid_time, 3, "Max Program Duration", "Close app after N seconds.",
                     self.max_prog)

        grp_time.setLayout(grid_time)
        content_layout.addWidget(grp_time)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    # ==========================================================================
    # Logic: Dynamic UI Updates
    # ==========================================================================
    def connect_signals(self):
        # Connect widgets to the update function
        self.model_combo.currentTextChanged.connect(self.update_ui_state)
        self.mode_combo.currentTextChanged.connect(self.update_ui_state)
        self.chk_notify.toggled.connect(self.update_ui_state)

    def update_ui_state(self):
        """Enables/Disables widgets based on current selections."""

        # 1. Model Name -> WSL Settings
        model = self.model_combo.currentText().lower()
        is_remote_nemo = "nvidia/" in model or "remote" in model

        # Enable WSL group only if remote model
        self.grp_wsl.setEnabled(is_remote_nemo)
        # Force CPU is only for local models (non-remote)
        self.chk_cpu.setEnabled(not is_remote_nemo)

        # 2. Transcription Mode -> Dictation/Interval params
        mode = self.mode_combo.currentText()
        is_dictation = mode == "dictationMode"

        self.dict_silence_dur.setEnabled(is_dictation)
        self.dict_loudness.setEnabled(is_dictation)
        self.lbl_dict_1.setEnabled(is_dictation)
        self.lbl_dict_2.setEnabled(is_dictation)

        self.const_interval.setEnabled(not is_dictation)
        self.lbl_int_1.setEnabled(not is_dictation)

        # 3. Notifications
        self.chk_play_enable.setEnabled(self.chk_notify.isChecked())

    # ==========================================================================
    # Helpers
    # ==========================================================================
    def create_help_icon(self, text):
        l = QLabel("(?)");
        l.setCursor(Qt.WhatsThisCursor)
        l.setStyleSheet("color: #0078D7; font-weight: bold; margin-left: 5px;")
        l.setToolTip(text);
        return l

    def add_row(self, grid, row, label, tooltip, widget):
        l = QLabel(label + ":");
        grid.addWidget(l, row, 0)
        grid.addWidget(widget, row, 1);
        grid.addWidget(self.create_help_icon(tooltip), row, 2)

    def populateAudioDevices(self):
        self.device_combo.clear()
        try:
            devs = self.audio_handler.listAudioDevices()
            self.device_combo.addItem("Default", None)
            if devs:
                for i, n in devs.items(): self.device_combo.addItem(str(n), i)
        except:
            self.device_combo.addItem("Default", None)

    def load_initial_settings(self):
        try:
            c = self.config
            # ... (Loading logic matches previous version) ...
            # Tab 1
            idx = self.device_combo.findData(c.get('deviceId'))
            if idx != -1: self.device_combo.setCurrentIndex(idx)
            self.rate_spin.setValue(int(c.get('sampleRate', 16000)))
            self.channels_spin.setValue(int(c.get('channels', 1)))
            self.block_spin.setValue(int(c.get('blockSize', 1024)))

            self.model_combo.setCurrentText(c.get('modelName', ''))
            self.lang_edit.setText(str(c.get('language', 'en')))
            self.chk_cpu.setChecked(c.get('CPU', False))

            self.wsl_url_edit.setText(c.get('wslServerUrl', ''))
            self.wsl_distro_edit.setText(c.get('wslDistributionName', ''))
            self.wsl_req_timeout.setValue(float(c.get('serverRequestTimeout', 15.0)))
            self.wsl_ready_timeout.setValue(float(c.get('wslServerReadyTimeout', 90.0)))
            self.chk_sudo.setChecked(c.get('wslUseSudo', False))
            self.chk_unload_exit.setChecked(c.get('unloadRemoteModelOnExit', True))

            # Tab 2
            self.mode_combo.setCurrentText(c.get('transcriptionMode', 'dictationMode'))
            self.dict_silence_dur.setValue(
                float(c.get('dictationMode_silenceDurationToOutput', 0.6)))
            self.dict_loudness.setValue(
                float(c.get('dictationMode_silenceLoudnessThreshold', 0.00035)))
            self.const_interval.setValue(
                float(c.get('constantIntervalMode_transcriptionInterval', 4.0)))

            self.min_loud_dur.setValue(float(c.get('minLoudDurationForTranscription', 0.3)))
            self.skip_thresh.setValue(float(c.get('silenceSkip_threshold', 0.0002)))
            self.skip_before.setValue(float(c.get('skipSilence_beforeNSecSilence', 0.3)))
            self.skip_after.setValue(float(c.get('skipSilence_afterNSecSilence', 0.3)))
            self.word_thresh.setValue(
                float(c.get('loudnessThresholdOf_commonFalseDetectedWords', 0.00065)))

            cw = c.get('commonFalseDetectedWords', [])
            self.common_words.setText(", ".join(cw) if isinstance(cw, list) else str(cw))
            bw = c.get('bannedWords', [])
            self.banned_words.setText(", ".join(bw) if isinstance(bw, list) else str(bw))

            # Tab 3
            self.chk_dots.setChecked(c.get('removeTrailingDots', True))
            self.chk_start_out.setChecked(c.get('outputEnabled', False))
            self.chk_start_rec.setChecked(c.get('isRecordingActive', True))
            self.chk_typing.setChecked(c.get('enableTypingOutput', True))
            self.combo_type_mode.setCurrentText(c.get('typingMode', 'whole'))

            self.chk_notify.setChecked(c.get('enableAudioNotifications', True))
            self.chk_play_enable.setChecked(c.get('playEnableSounds', False))

            self.key_rec.setText(c.get('recordingToggleKey', ''))
            self.key_out.setText(c.get('outputToggleKey', ''))
            self.key_force.setText(c.get('forceTranscriptionKey', ''))

            self.time_unload.setValue(int(c.get('model_unloadTimeout', 3600)))
            self.time_idle.setValue(int(c.get('consecutiveIdleTime', 1800)))
            self.max_rec.setValue(int(c.get('maxDurationRecording', 0)))
            self.max_prog.setValue(int(c.get('maxDurationProgramActive', 0)))

            # [NEW] Trigger UI update after loading
            self.update_ui_state()

        except Exception as e:
            uniLogger.error(f"[GUI] Load Error: {e}", excInfo=True)

    @Slot()
    def apply_settings(self):
        try:
            c = self.config
            # ... (Saving logic matches previous version) ...
            # Tab 1
            c.set('deviceId', self.device_combo.currentData())
            c.set('sampleRate', self.rate_spin.value())
            c.set('channels', self.channels_spin.value())
            c.set('blockSize', self.block_spin.value())

            c.set('modelName', self.model_combo.currentText())
            c.set('language', self.lang_edit.text())
            c.set('CPU', self.chk_cpu.isChecked())

            c.set('wslServerUrl', self.wsl_url_edit.text())
            c.set('wslDistributionName', self.wsl_distro_edit.text())
            c.set('serverRequestTimeout', self.wsl_req_timeout.value())
            c.set('wslServerReadyTimeout', self.wsl_ready_timeout.value())
            c.set('wslUseSudo', self.chk_sudo.isChecked())
            c.set('unloadRemoteModelOnExit', self.chk_unload_exit.isChecked())

            # Tab 2
            c.set('transcriptionMode', self.mode_combo.currentText())
            c.set('dictationMode_silenceDurationToOutput', self.dict_silence_dur.value())
            c.set('dictationMode_silenceLoudnessThreshold', self.dict_loudness.value())
            c.set('constantIntervalMode_transcriptionInterval', self.const_interval.value())

            c.set('minLoudDurationForTranscription', self.min_loud_dur.value())
            c.set('silenceSkip_threshold', self.skip_thresh.value())
            c.set('skipSilence_beforeNSecSilence', self.skip_before.value())
            c.set('skipSilence_afterNSecSilence', self.skip_after.value())
            c.set('loudnessThresholdOf_commonFalseDetectedWords', self.word_thresh.value())

            # List parsing
            c.set('commonFalseDetectedWords',
                  [x.strip() for x in self.common_words.text().split(',') if x.strip()])
            c.set('bannedWords',
                  [x.strip() for x in self.banned_words.text().split(',') if x.strip()])

            # Tab 3
            c.set('removeTrailingDots', self.chk_dots.isChecked())
            c.set('outputEnabled', self.chk_start_out.isChecked())
            c.set('isRecordingActive', self.chk_start_rec.isChecked())
            c.set('enableTypingOutput', self.chk_typing.isChecked())
            c.set('typingMode', self.combo_type_mode.currentText())

            c.set('enableAudioNotifications', self.chk_notify.isChecked())
            c.set('playEnableSounds', self.chk_play_enable.isChecked())

            c.set('recordingToggleKey', self.key_rec.text())
            c.set('outputToggleKey', self.key_out.text())
            c.set('forceTranscriptionKey', self.key_force.text())

            c.set('model_unloadTimeout', self.time_unload.value())
            c.set('consecutiveIdleTime', self.time_idle.value())
            c.set('maxDurationRecording', self.max_rec.value())
            c.set('maxDurationProgramActive', self.max_prog.value())

            uniLogger.info("[GUI] ALL settings saved successfully.")
        except Exception as e:
            uniLogger.error(f"[GUI] Save Error: {e}", excInfo=True)