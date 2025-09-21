# managers.py
# ==============================================================================
# Configuration, State, and Model Lifecycle Managers
# ==============================================================================
#
# Purpose:
# - ConfigurationManager: Holds and provides access to static application settings.
# - StateManager: Tracks dynamic application state (recording, output enabled, timings).
# - ModelLifecycleManager: Manages automatic loading/unloading of the ASR model
#   based on activity timeouts, interacting with the specific ASR handler.
# ==============================================================================
import os
import sys
import time
from pathlib import Path

# Import the new universal logger instances
from utils.loggerInstance import uniLogger, uniDebugLogger


# ==================================
# Configuration Management
# ==================================
class ConfigurationManager:
    """Stores and provides access to all application settings."""

    def __init__(self, **kwargs):
        """Initializes the configuration with user-provided settings and derives internal values."""
        self._config = kwargs
        # --- Derived/Internal Settings ---
        # Determine the directory of the running script robustly
        self._config['scriptDir'] = self._findScriptDirectory()
        # Initialize placeholders for settings that will be determined later by other components
        self._config['device'] = None  # Will be set by the AsrModelHandler
        # Set initial 'actual' audio values from requested config; these may be updated by AudioHandler
        self._config['actualSampleRate'] = self._config.get('sampleRate', 16000)
        self._config['actualChannels'] = self._config.get('channels', 1)
        uniDebugLogger.debug(
            f"Configuration initialized. Script directory: {self._config['scriptDir']}")

    def _findScriptDirectory(self):
        """Finds the directory of the main running script to locate assets like sound files."""
        try:
            import __main__
            if hasattr(__main__, '__file__') and __main__.__file__:
                # This is the most reliable method
                mainFilePath = Path(os.path.abspath(__main__.__file__))
                return mainFilePath.parent
            else:
                # Fallback for environments where __main__.__file__ is not set (e.g., interactive REPL)
                uniLogger.warning(
                    "Cannot determine main script path (__main__.__file__ missing/None), using managers.py directory as fallback.")
                return Path(os.path.dirname(os.path.abspath(__file__)))
        except (AttributeError, ImportError, TypeError):
            # Last resort if introspection fails
            uniLogger.warning("Error determining script directory, using CWD as fallback.",
                              excInfo=True)
            return Path.cwd()

    def get(self, key, default=None):
        """Gets a configuration value by key, returning a default if the key is not found."""
        return self._config.get(key, default)

    def set(self, key, value):
        """Sets or updates a configuration value. Used for dynamic settings like the detected device."""
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
        """Initializes the state manager based on default values from the configuration."""
        if not isinstance(config, ConfigurationManager):
            # This is a critical setup error, so log and raise an exception.
            uniLogger.error("StateManager requires a valid ConfigurationManager instance.")
            raise ValueError("StateManager requires a valid ConfigurationManager.")
        self.config = config
        # Initialize state flags from configuration defaults
        self.isProgramActive = True  # Overall application loop control flag
        self.isRecordingActive = self.config.get('isRecordingActive', True)
        self.outputEnabled = self.config.get('outputEnabled', False)
        # Initialize timing-related state variables
        now = time.time()
        self.programStartTime = now
        self.lastActivityTime = now  # Tracks user/system activity for model unload timeout
        self.recordingStartTime = now if self.isRecordingActive else 0
        self.lastValidTranscriptionTime = now  # Tracks last text output for idle timeout
        uniDebugLogger.debug("StateManager initialized.")
        uniLogger.info(
            f"Initial State - Recording: {self.isRecordingActive}, Output Enabled: {self.outputEnabled}")

    # --- Getters ---
    def isRecording(self) -> bool:
        """Returns True if recording is currently active, False otherwise."""
        return self.isRecordingActive

    def isOutputEnabled(self) -> bool:
        """Returns True if text output (typing/clipboard) is enabled, False otherwise."""
        return self.outputEnabled

    def shouldProgramContinue(self) -> bool:
        """Returns True if the main application loop should continue, False otherwise."""
        return self.isProgramActive

    # --- Setters ---
    def startRecording(self) -> bool:
        """Activates the recording state and updates relevant timers. Returns True if the state changed."""
        if not self.isRecordingActive:
            uniDebugLogger.debug("Setting state to Recording: ON")
            self.isRecordingActive = True
            now = time.time()
            self.recordingStartTime = now
            self.lastActivityTime = now  # Starting recording is considered an activity
            self.lastValidTranscriptionTime = now  # Reset the idle timer on recording start
            return True  # State changed
        uniDebugLogger.debug("startRecording called but already recording.")
        return False  # No state change

    def stopRecording(self) -> bool:
        """Deactivates the recording state and updates activity time. Returns True if the state changed."""
        if self.isRecordingActive:
            uniDebugLogger.debug("Setting state to Recording: OFF")
            self.isRecordingActive = False
            self.recordingStartTime = 0  # Reset the recording session start time
            self.lastActivityTime = time.time()  # Stopping is also an activity
            return True  # State changed
        uniDebugLogger.debug("stopRecording called but already stopped.")
        return False  # No state change

    def toggleOutput(self) -> bool:
        """Toggles the text output state. Returns the new state (True if enabled)."""
        self.outputEnabled = not self.outputEnabled
        status = 'ENABLED' if self.outputEnabled else 'DISABLED'
        uniLogger.info(f"Text Output {status}")
        uniDebugLogger.debug(f"Setting state Output Enabled: {self.outputEnabled}")
        self.updateLastActivityTime()  # Toggling output is a user activity
        return self.outputEnabled

    def stopProgram(self):
        """Signals the main application loop to stop by setting the active flag to False."""
        if self.isProgramActive:
            uniDebugLogger.debug("Setting state Program Active: OFF")
            self.isProgramActive = False

    def updateLastActivityTime(self):
        """Updates the timestamp of the last significant activity (used for the model unload timeout)."""
        now = time.time()
        uniDebugLogger.debug(f"Updating last activity time to {now:.2f}")
        self.lastActivityTime = now

    def updateLastValidTranscriptionTime(self):
        """Updates the timestamp of the last valid transcription output (used for the idle timeout)."""
        now = time.time()
        uniDebugLogger.debug(
            f"Updating last valid transcription time to {now:.2f} (idle timer reset).")
        self.lastValidTranscriptionTime = now

    # --- Timeout Checks ---
    def checkRecordingTimeout(self) -> bool:
        """
        Checks if the maximum recording session duration has been exceeded.
        Returns True if the timeout is reached. (A value of 0 means no limit).
        """
        maxDuration = self.config.get('maxDurationRecording', 0)
        if maxDuration <= 0:
            return False  # Timeout is disabled
        if not self.isRecordingActive or self.recordingStartTime <= 0:
            return False
        elapsed = time.time() - self.recordingStartTime
        if elapsed >= maxDuration:
            uniDebugLogger.debug(
                f"Max recording duration check: Elapsed {elapsed:.1f}s >= Limit {maxDuration}s. Timeout.")
            return True
        return False

    def checkIdleTimeout(self) -> bool:
        """
        Checks if the idle time (no valid transcription while recording) has been exceeded.
        Returns True if the timeout is reached.
        """
        if not self.isRecordingActive:
            return False
        idleTimeout = self.config.get('consecutiveIdleTime', 0)
        if idleTimeout <= 0:
            return False  # Timeout is disabled
        silentFor = time.time() - self.lastValidTranscriptionTime
        if silentFor >= idleTimeout:
            uniDebugLogger.debug(
                f"Idle timeout check: Silent for {silentFor:.1f}s >= Limit {idleTimeout}s. Timeout.")
            return True
        return False

    def timeSinceLastActivity(self) -> float:
        """Calculates the time elapsed since the last recorded user/system activity."""
        return time.time() - self.lastActivityTime

    def checkProgramTimeout(self):
        """
        Checks if the maximum program duration has been exceeded.
        If the timeout is reached, it signals the program to stop and returns True.
        """
        maxDuration = self.config.get('maxDurationProgramActive', 0)
        if maxDuration <= 0:
            return False  # Timeout is disabled
        elapsed = time.time() - self.programStartTime
        if elapsed >= maxDuration:
            uniDebugLogger.debug(
                f"Program timeout check: Elapsed {elapsed:.1f}s >= Limit {maxDuration}s. Timeout.")
            self.stopProgram()  # Signal the program to stop
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
        """Initializes the manager with all necessary application components."""
        # Validate inputs to ensure proper setup
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
        """
        Runs in a background thread to periodically check and manage the model's loaded state.
        """
        handlerType = type(self.asrModelHandler).__name__
        uniLogger.info(f"Starting Model Lifecycle Manager thread (Handler: {handlerType}).")
        checkInterval = 10  # Seconds to wait between checks
        while self.stateManager.shouldProgramContinue():
            try:
                # Get current application state
                isRecording = self.stateManager.isRecording()
                modelIsCurrentlyLoaded = self.asrModelHandler.isModelLoaded()
                unloadTimeout = self.config.get('model_unloadTimeout', 0)

                # --- Unload Condition ---
                # Unload if timeout is enabled, not recording, and model is loaded
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

                # --- Load Condition ---
                # Load if recording is active but the model is not currently loaded
                elif isRecording and not modelIsCurrentlyLoaded:
                    uniLogger.info(
                        "Recording active but model not loaded. Triggering model load/check...")
                    try:
                        loadSuccess = self.asrModelHandler.loadModel()
                        if not loadSuccess:
                            uniLogger.warning(
                                "Model load/check request reported failure. Will retry later.")
                            time.sleep(5)  # Wait a bit before the next check if loading failed
                    except Exception as e:
                        uniLogger.error(f"Error during model load request: {e}", excInfo=True)
                        time.sleep(5)  # Wait after an error
                    # A load attempt is considered an activity, resetting the unload timer
                    self.stateManager.updateLastActivityTime()
            except Exception as loopError:
                # Catch any unexpected errors within the management loop
                uniLogger.error(f"Error in ModelLifecycleManager loop: {loopError}", excInfo=True)
                time.sleep(checkInterval)

            # --- Periodic Check Interval ---
            # Sleep in small chunks to allow for a responsive shutdown
            loopStartTime = time.time()
            while (
                    time.time() - loopStartTime < checkInterval) and self.stateManager.shouldProgramContinue():
                time.sleep(0.5)
        uniLogger.info("Model Lifecycle Manager thread stopping.")
