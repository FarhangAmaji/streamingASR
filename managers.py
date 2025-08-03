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
import sys  # Needed for stderr print during early error
import time
from pathlib import Path

# Import logging helpers from utils
from utils import logWarning, logDebug, logInfo, logError


# Pygame needed for notifications in ModelLifecycleManager (indirectly via SystemInteractionHandler)
# PyAutoGUI is not directly used here.
# ==================================
# Configuration Management
# ==================================
class ConfigurationManager:
    """Stores and provides access to all application settings."""

    def __init__(self, **kwargs):
        self._config = kwargs
        # --- Derived/Internal Settings ---
        # Determine script directory robustly
        self._config['scriptDir'] = self._findScriptDirectory()
        # Initialize placeholders for settings derived later
        self._config['device'] = None  # Set by AsrModelHandler
        # Set initial 'actual' values from requested, may be updated by AudioHandler
        self._config['actualSampleRate'] = self._config.get('sampleRate', 16000)
        self._config['actualChannels'] = self._config.get('channels', 1)
        logDebug(f"Configuration initialized. Script directory: {self._config['scriptDir']}")

    def _findScriptDirectory(self):
        """Finds the directory of the main running script."""
        try:
            import __main__
            if hasattr(__main__, '__file__') and __main__.__file__:
                mainFilePath = Path(os.path.abspath(__main__.__file__))
                return mainFilePath.parent
            else:
                # Fallback for environments where __main__.__file__ is not set (e.g., interactive)
                # Try using the directory of this file (managers.py) as a less reliable fallback
                logWarning(
                    "Cannot determine main script path (__main__.__file__ missing/None), using managers.py directory as fallback.")
                return Path(os.path.dirname(os.path.abspath(__file__)))
        except (AttributeError, ImportError, TypeError):
            logWarning("Error determining script directory, using CWD as fallback.", exc_info=True)
            return Path.cwd()  # Current working directory as last resort

    def get(self, key, default=None):
        """Gets a configuration value by key, returning default if not found."""
        return self._config.get(key, default)

    def set(self, key, value):
        """Sets or updates a configuration value."""
        # Log if a key is being updated? Optional.
        # if key in self._config:
        #     logDebug(f"Configuration updated: '{key}' changed from '{self._config[key]}' to '{value}'")
        # else:
        #     logDebug(f"Configuration set: '{key}' = '{value}'")
        self._config[key] = value

    def getAll(self):
        """Returns a copy of the entire configuration dictionary."""
        # Return a copy to prevent external modification of the internal dict
        return self._config.copy()


# ==================================
# State Management
# ==================================
class StateManager:
    """Manages the dynamic state of the real-time transcriber."""

    def __init__(self, config):
        if not isinstance(config, ConfigurationManager):
            # Use basic print/log error as logger might not be fully set up if config fails early
            print("ERROR: StateManager requires a valid ConfigurationManager instance.",
                  file=sys.stderr)
            raise ValueError("StateManager requires a valid ConfigurationManager.")
        self.config = config
        # Initialize state based on configuration defaults
        self.isProgramActive = True  # Overall application loop control
        self.isRecordingActive = self.config.get('isRecordingActive', True)
        self.outputEnabled = self.config.get('outputEnabled', False)
        # Timing state initialization
        now = time.time()
        self.programStartTime = now
        self.lastActivityTime = now  # Tracks model/server interaction activity
        # Initialize recordingStartTime only if starting active
        self.recordingStartTime = now if self.isRecordingActive else 0
        # Tracks last time valid text was output, used for idle timeout
        self.lastValidTranscriptionTime = now
        logDebug("StateManager initialized.")
        logInfo(
            f"Initial State - Recording: {self.isRecordingActive}, Output Enabled: {self.outputEnabled}")

    # --- Getters ---
    def isRecording(self):
        """Returns True if recording is currently active, False otherwise."""
        return self.isRecordingActive

    def isOutputEnabled(self):
        """Returns True if text output (typing/clipboard) is enabled, False otherwise."""
        return self.outputEnabled

    def shouldProgramContinue(self):
        """Returns True if the main application loop should continue, False otherwise."""
        return self.isProgramActive

    # --- Setters ---
    def startRecording(self):
        """Activates recording state and updates relevant timers. Returns True if state changed."""
        if not self.isRecordingActive:
            logDebug("Setting state to Recording: ON")
            self.isRecordingActive = True
            now = time.time()
            self.recordingStartTime = now
            self.lastActivityTime = now  # Mark activity for model manager/timeout checks
            self.lastValidTranscriptionTime = now  # Reset idle timer on start
            return True  # State changed
        logDebug("startRecording called but already recording.")
        return False  # No state change

    def stopRecording(self):
        """Deactivates recording state and updates activity time. Returns True if state changed."""
        if self.isRecordingActive:
            logDebug("Setting state to Recording: OFF")
            self.isRecordingActive = False
            self.recordingStartTime = 0  # Reset session start time
            # Mark activity time when stopping recording as well
            self.lastActivityTime = time.time()
            return True  # State changed
        logDebug("stopRecording called but already stopped.")
        return False  # No state change

    def toggleOutput(self):
        """Toggles the text output state. Returns the new state (True if enabled, False if disabled)."""
        self.outputEnabled = not self.outputEnabled
        status = 'ENABLED' if self.outputEnabled else 'DISABLED'
        logInfo(f"Text Output {status}")
        logDebug(f"Setting state Output Enabled: {self.outputEnabled}")
        # Mark activity when toggling output, might interact with server or model state implicitly
        self.updateLastActivityTime()
        return self.outputEnabled  # Return the new state

    def stopProgram(self):
        """Signals the main application loop to stop."""
        if self.isProgramActive:
            logDebug("Setting state Program Active: OFF")
            self.isProgramActive = False
        # else: logDebug("stopProgram called but already stopped.")

    def updateLastActivityTime(self):
        """Updates the timestamp of the last significant activity (used for model unload timeout)."""
        now = time.time()
        logDebug(f"Updating last activity time to {now:.2f}")
        self.lastActivityTime = now

    def updateLastValidTranscriptionTime(self):
        """Updates the timestamp of the last valid transcription output (used for idle timeout)."""
        now = time.time()
        logDebug(f"Updating last valid transcription time to {now:.2f} (idle timer reset).")
        self.lastValidTranscriptionTime = now

    # --- Timeout Checks ---
    def checkRecordingTimeout(self):
        """
        Checks if the maximum recording session duration has been exceeded.
        Returns True if timeout reached, False otherwise. (Treats 0 or less as no limit).
        """
        maxDuration = self.config.get('maxDurationRecording', 0)  # Default 0 (no limit)
        # If maxDuration is 0 or negative, treat it as no limit
        if maxDuration <= 0:
            return False  # Never time out based on recording duration
        # Only check if currently recording and recording started properly
        if not self.isRecordingActive or self.recordingStartTime <= 0:
            return False
        elapsed = time.time() - self.recordingStartTime
        if elapsed >= maxDuration:
            logDebug(
                f"Max recording duration check: Elapsed {elapsed:.1f}s >= Limit {maxDuration}s. Timeout.")
            return True  # Timeout reached
        # else: logDebug(f"Max recording duration check: Elapsed {elapsed:.1f}s < Limit {maxDuration}s. OK.")
        return False  # Timeout not reached

    def checkIdleTimeout(self):
        """
        Checks if the consecutive idle time (no valid transcription output while recording)
        limit has been reached. Returns True if timeout reached, False otherwise.
        (Treats 0 or less as no limit).
        """
        # Only check if recording is supposed to be active
        if not self.isRecordingActive:
            return False
        idleTimeout = self.config.get('consecutiveIdleTime', 0)  # Default 0 (no limit)
        # If idleTimeout is 0 or negative, treat it as no limit
        if idleTimeout <= 0:
            return False  # Never time out based on idle time
        # Check time since the last *valid* transcription was output
        silentFor = time.time() - self.lastValidTranscriptionTime
        if silentFor >= idleTimeout:
            logDebug(
                f"Idle timeout check: Silent for {silentFor:.1f}s >= Limit {idleTimeout}s. Timeout.")
            return True  # Idle timeout reached
        # else: logDebug(f"Idle timeout check: Silent for {silentFor:.1f}s < Limit {idleTimeout}s. OK.")
        return False  # Idle timeout not reached

    def timeSinceLastActivity(self):
        """Calculates the time elapsed since the last recorded activity (model/server interaction)."""
        return time.time() - self.lastActivityTime

    def checkProgramTimeout(self):
        """
        Checks if the maximum program duration has been exceeded.
        If timeout is reached, it calls stopProgram() and returns True.
        Returns False otherwise. (Treats 0 or less as no limit).
        """
        maxDuration = self.config.get('maxDurationProgramActive', 0)  # Default 0 (no limit)
        if maxDuration <= 0:
            return False  # Never time out based on program duration
        elapsed = time.time() - self.programStartTime
        if elapsed >= maxDuration:
            logDebug(
                f"Program timeout check: Elapsed {elapsed:.1f}s >= Limit {maxDuration}s. Timeout.")
            self.stopProgram()  # Signal program stop
            return True  # Timeout reached
        # else: logDebug(f"Program timeout check: Elapsed {elapsed:.1f}s < Limit {maxDuration}s. OK.")
        return False  # Timeout not reached


# ==================================
# Model Lifecycle Management
# ==================================
class ModelLifecycleManager:
    """
    Handles automatic loading/unloading of the ASR model based on activity.
    Works for both local handlers and the remote client handler.
    """

    def __init__(self, config, stateManager, asrModelHandler, systemInteractionHandler):
        # Validate inputs
        if not isinstance(config, ConfigurationManager): raise ValueError(
            "Invalid ConfigurationManager")
        if not isinstance(stateManager, StateManager): raise ValueError("Invalid StateManager")
        # Check if asrModelHandler is an instance of the abstract base class if available
        # from modelHandlers import AbstractAsrModelHandler # Avoid circular import if possible
        # if not isinstance(asrModelHandler, AbstractAsrModelHandler): raise ValueError("Invalid AsrModelHandler")
        if asrModelHandler is None: raise ValueError("AsrModelHandler cannot be None")
        # systemInteractionHandler might be optional depending on features, but needed for sound here
        if systemInteractionHandler is None: raise ValueError(
            "SystemInteractionHandler cannot be None")
        self.config = config
        self.stateManager = stateManager
        self.asrModelHandler = asrModelHandler
        self.systemInteractionHandler = systemInteractionHandler
        logDebug("ModelLifecycleManager initialized.")

    def manageModelLifecycle(self):
        """
        Runs in a background thread to monitor activity and load/unload the model.
        """
        handlerType = type(self.asrModelHandler).__name__
        logInfo(f"Starting Model Lifecycle Manager thread (Handler: {handlerType}).")
        checkInterval = 10  # Seconds to wait between checks (adjust as needed)
        while self.stateManager.shouldProgramContinue():
            try:
                # Get current state
                isRecording = self.stateManager.isRecording()
                modelIsCurrentlyLoaded = self.asrModelHandler.isModelLoaded()
                unloadTimeout = self.config.get('model_unloadTimeout', 0)  # Default 0 (disabled)
                # --- Unload Condition ---
                # Unload if:
                # 1. Timeout is enabled (unloadTimeout > 0)
                # 2. Currently NOT recording
                # 3. Model IS currently loaded
                if unloadTimeout > 0 and not isRecording and modelIsCurrentlyLoaded:
                    timeInactive = self.stateManager.timeSinceLastActivity()
                    if timeInactive >= unloadTimeout:
                        logInfo(
                            f"Model inactive for {timeInactive:.1f}s (>= {unloadTimeout}s), requesting unload...")
                        try:
                            unloadSuccess = self.asrModelHandler.unloadModel()  # Request unload (local or remote)
                            if unloadSuccess:
                                # Play sound only if unload seemed successful
                                self.systemInteractionHandler.playNotification("modelUnloaded")
                                logInfo("Model unload successful.")
                            else:
                                logWarning("Model unload request reported failure.")
                        except Exception as e:
                            logError(f"Error during model unload request: {e}", exc_info=True)
                    # else: # Log inactivity duration periodically if debugging is needed
                    #      if logging.getLogger("SpeechToTextApp").isEnabledFor(logging.DEBUG):
                    #           logDebug(f"Model loaded but inactive. Time since last activity: {timeInactive:.1f}s / {unloadTimeout}s")
                # --- Load Condition ---
                # Load if:
                # 1. Currently IS recording
                # 2. Model IS NOT currently loaded
                elif isRecording and not modelIsCurrentlyLoaded:
                    logInfo("Recording active but model not loaded. Triggering model load/check...")
                    try:
                        loadSuccess = self.asrModelHandler.loadModel()  # Request load (local or remote check/load)
                        if not loadSuccess:
                            # loadModel logs details of failure (local or remote communication)
                            logWarning(
                                "Model load/check request reported failure. Will retry later.")
                            # Optional: Implement backoff strategy here?
                            time.sleep(5)  # Wait a bit before next check if load failed
                        # else: Load successful (logged within loadModel)
                    except Exception as e:
                        logError(f"Error during model load request: {e}", exc_info=True)
                        time.sleep(5)  # Wait after error
                    # Update activity time after a load attempt (success or fail) to reset unload timer
                    self.stateManager.updateLastActivityTime()
            except Exception as loopError:
                # Catch unexpected errors within the lifecycle loop itself
                logError(f"Error in ModelLifecycleManager loop: {loopError}", exc_info=True)
                # Avoid busy-looping if error persists
                time.sleep(checkInterval)
            # --- Periodic Check Interval ---
            # Use a timed sleep that checks the program state periodically for faster shutdown
            loopStartTime = time.time()
            while (
                    time.time() - loopStartTime < checkInterval) and self.stateManager.shouldProgramContinue():
                time.sleep(0.5)  # Sleep in smaller chunks to be responsive
        logInfo("Model Lifecycle Manager thread stopping.")
