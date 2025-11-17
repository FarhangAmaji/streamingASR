# ==============================================================================
# Configuration, State, and Model Lifecycle Managers
# ==============================================================================
import os
import sys
import time
from pathlib import Path

from utils.loggerInstance import uniLogger, uniDebugLogger


# ==================================
# Configuration Management
# ==================================
class ConfigurationManager:
    """Stores and provides access to all application settings."""

    def __init__(self, **kwargs):
        self._config = kwargs
        # --- NEW: REAL-TIME FLAGS (Required for Orchestrator to monitor changes) ---
        self.wsl_restart_required: bool = False
        self.audio_settings_changed: bool = False
        # --- END NEW ---

        self._config['scriptDir'] = self._findScriptDirectory()
        self._config['device'] = None
        self._config['actualSampleRate'] = self._config.get('sampleRate', 16000)
        self._config['actualChannels'] = self._config.get('channels', 1)
        uniDebugLogger.debug("Configuration initialized.")

    def _findScriptDirectory(self):
        try:
            import __main__
            if hasattr(__main__, '__file__') and __main__.__file__:
                mainFilePath = Path(os.path.abspath(__main__.__file__))
                return mainFilePath.parent
            else:
                return Path(os.path.dirname(os.path.abspath(__file__)))
        except (AttributeError, ImportError, TypeError):
            uniLogger.warning("Error determining script directory, using CWD as fallback.",
                              excInfo=True)
            return Path.cwd()

    def get(self, key, default=None):
        """Gets a configuration value by key, returning a default if the key is not found."""
        return self._config.get(key, default)

    def set(self, key, value):
        """Sets or updates a configuration value. Used for dynamic settings like the detected device."""

        # 1. CRITICAL WSL SETTING CHANGE CHECK (Requires Full Backend Restart)
        critical_wsl_keys = ['wslServerUrl', 'wslDistributionName', 'wslUseSudo', 'modelName']
        if key in critical_wsl_keys and self._config.get(key) != value:
            self._config[key] = value
            self.wsl_restart_required = True
            uniLogger.warning(f"WSL critical setting '{key}' changed. Restart required.")
            return

        # 2. AUDIO SETTING CHANGE CHECK (Requires Audio Stream Re-initialization)
        audio_keys = ['sampleRate', 'channels', 'blockSize', 'deviceId']
        if key in audio_keys and self._config.get(key) != value:
            self._config[key] = value
            self.audio_settings_changed = True
            uniDebugLogger.debug(f"Audio setting '{key}' changed. Real-time application pending.")
            return

        # 3. Default update for all other settings
        self._config[key] = value

    def getAll(self):
        """Returns a copy of the entire configuration dictionary to prevent external modification."""
        return self._config.copy()


# ==================================
# State Management
# ==================================
class StateManager:
    """Manages the dynamic state of the real-time transcriber application."""

    def __init__(self, config):
        if not isinstance(config, ConfigurationManager):
            uniLogger.error("StateManager requires a valid ConfigurationManager instance.")
            raise ValueError("StateManager requires a valid ConfigurationManager.")
        self.config = config
        self.isProgramActive = True
        self.isRecordingActive = self.config.get('isRecordingActive', True)
        self.outputEnabled = self.config.get('outputEnabled', False)
        now = time.time()
        self.programStartTime = now
        self.lastActivityTime = now
        self.recordingStartTime = now if self.isRecordingActive else 0
        self.lastValidTranscriptionTime = now
        uniDebugLogger.debug("StateManager initialized.")
        uniLogger.info(
            f"Initial State - Recording: {self.isRecordingActive}, Output Enabled: {self.outputEnabled}")

    # --- Getters ---
    def isRecording(self) -> bool:
        return self.isRecordingActive

    def isOutputEnabled(self) -> bool:
        return self.outputEnabled

    def shouldProgramContinue(self) -> bool:
        return self.isProgramActive

    # --- Setters ---
    def startRecording(self) -> bool:
        if not self.isRecordingActive:
            uniDebugLogger.debug("Setting state to Recording: ON")
            self.isRecordingActive = True
            now = time.time()
            self.recordingStartTime = now
            self.lastActivityTime = now
            self.lastValidTranscriptionTime = now
            return True
        uniDebugLogger.debug("startRecording called but already recording.")
        return False

    def stopRecording(self) -> bool:
        if self.isRecordingActive:
            uniDebugLogger.debug("Setting state to Recording: OFF")
            self.isRecordingActive = False
            self.recordingStartTime = 0
            self.lastActivityTime = time.time()
            return True
        uniDebugLogger.debug("stopRecording called but already stopped.")
        return False

    def toggleOutput(self) -> bool:
        self.outputEnabled = not self.outputEnabled
        status = 'ENABLED' if self.outputEnabled else 'DISABLED'
        uniLogger.info(f"Text Output {status}")
        uniDebugLogger.debug(f"Setting state Output Enabled: {self.outputEnabled}")
        self.updateLastActivityTime()
        return self.outputEnabled

    def stopProgram(self):
        if self.isProgramActive:
            uniDebugLogger.debug("Setting state Program Active: OFF")
            self.isProgramActive = False

    def updateLastActivityTime(self):
        now = time.time()
        uniDebugLogger.debug(f"Updating last activity time to {now:.2f}")
        self.lastActivityTime = now

    def updateLastValidTranscriptionTime(self):
        now = time.time()
        uniDebugLogger.debug(
            f"Updating last valid transcription time to {now:.2f} (idle timer reset).")
        self.lastValidTranscriptionTime = now

    # --- Timeout Checks ---
    def checkRecordingTimeout(self) -> bool:
        maxDuration = self.config.get('maxDurationRecording', 0)
        if maxDuration <= 0: return False
        if not self.isRecordingActive or self.recordingStartTime <= 0: return False
        elapsed = time.time() - self.recordingStartTime
        if elapsed >= maxDuration:
            uniDebugLogger.debug(
                f"Max recording duration check: Elapsed {elapsed:.1f}s >= Limit {maxDuration}s. Timeout.")
            return True
        return False

    def checkIdleTimeout(self) -> bool:
        if not self.isRecordingActive: return False
        idleTimeout = self.config.get('consecutiveIdleTime', 0)
        if idleTimeout <= 0: return False
        silentFor = time.time() - self.lastValidTranscriptionTime
        if silentFor >= idleTimeout:
            uniDebugLogger.debug(
                f"Idle timeout check: Silent for {silentFor:.1f}s >= Limit {idleTimeout}s. Timeout.")
            return True
        return False

    def timeSinceLastActivity(self) -> float:
        return time.time() - self.lastActivityTime

    def checkProgramTimeout(self):
        maxDuration = self.config.get('maxDurationProgramActive', 0)
        if maxDuration <= 0: return False
        elapsed = time.time() - self.programStartTime
        if elapsed >= maxDuration:
            uniDebugLogger.debug(
                f"Program timeout check: Elapsed {elapsed:.1f}s >= Limit {maxDuration}s. Timeout.")
            self.stopProgram()
            return True
        return False


# ==================================
# Model Lifecycle Management
# ==================================
class ModelLifecycleManager:
    """
    Handles automatic loading and unloading of the ASR model based on application activity
    to conserve system resources (especially VRAM).
    """

    def __init__(self, config, stateManager, asrModelHandler, systemInteractionHandler):
        if not isinstance(config, ConfigurationManager): raise ValueError(
            "Invalid ConfigurationManager")
        if not isinstance(stateManager, StateManager): raise ValueError("Invalid StateManager")
        if asrModelHandler is None: raise ValueError("AsrModelHandler cannot be None")
        if systemInteractionHandler is None: raise ValueError(
            "SystemInteractionHandler cannot be None")
        self.config = config
        self.stateManager = stateManager
        self.asrModelHandler = asrModelHandler
        self.systemInteractionHandler = systemInteractionHandler
        uniDebugLogger.debug("ModelLifecycleManager initialized.")

    def manageModelLifecycle(self):
        handlerType = type(self.asrModelHandler).__name__
        uniLogger.info(f"Starting Model Lifecycle Manager thread (Handler: {handlerType}).")
        checkInterval = 10
        while self.stateManager.shouldProgramContinue():
            try:
                isRecording = self.stateManager.isRecording()
                modelIsCurrentlyLoaded = self.asrModelHandler.isModelLoaded()
                unloadTimeout = self.config.get('model_unloadTimeout', 0)

                if unloadTimeout > 0 and not isRecording and modelIsCurrentlyLoaded:
                    timeInactive = self.stateManager.timeSinceLastActivity()
                    if timeInactive >= unloadTimeout:
                        uniLogger.info(
                            f"Model inactive for {timeInactive:.1f}s (>= {unloadTimeout}s), requesting unload...")
                        try:
                            unloadSuccess = self.asrModelHandler.unloadModel()
                            if unloadSuccess:
                                self.systemInteractionHandler.playNotification("modelUnloaded")
                                uniLogger.info("Model unload successful.")
                            else:
                                uniLogger.warning("Model unload request reported failure.")
                        except Exception as e:
                            uniLogger.error(f"Error during model unload request: {e}", excInfo=True)

                elif isRecording and not modelIsCurrentlyLoaded:
                    uniLogger.info(
                        "Recording active but model not loaded. Triggering model load/check...")
                    try:
                        loadSuccess = self.asrModelHandler.loadModel()
                        if not loadSuccess:
                            uniLogger.warning(
                                "Model load/check request reported failure. Will retry later.")
                            time.sleep(5)
                        self.stateManager.updateLastActivityTime()
                    except Exception as e:
                        uniLogger.error(f"Error during model load request: {e}", excInfo=True)
                        time.sleep(5)

            except Exception as loopError:
                uniLogger.error(f"Error in ModelLifecycleManager loop: {loopError}", excInfo=True)
                time.sleep(checkInterval)

            loopStartTime = time.time()
            while (
                    time.time() - loopStartTime < checkInterval) and self.stateManager.shouldProgramContinue():
                time.sleep(0.5)
        uniLogger.info("Model Lifecycle Manager thread stopping.")