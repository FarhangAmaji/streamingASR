# managers.py

# Pygame and PyAutoGUI are imported conditionally later where needed

import os
import time
from pathlib import Path

from utils import logWarning, logDebug, logInfo, logError


# ==================================
# Configuration Management
# ==================================
class ConfigurationManager:
    """Stores and provides access to all application settings."""

    def __init__(self, **kwargs):
        self._config = kwargs
        # --- Derived/Internal Settings ---
        # Ensure scriptDir uses the path of the *running* script (e.g., useRealtimeTranscription.py)
        try:
            # This might be fragile depending on how things are imported/run.
            # A more robust way might be to pass the script path explicitly during init.
            import __main__
            main_file_path = os.path.abspath(__main__.__file__)
            self._config['scriptDir'] = Path(os.path.dirname(main_file_path))
        except (AttributeError, ImportError):
            # Fallback if __main__.__file__ isn't available (e.g., interactive session)
            # This assumes mainTranscriberLogic.py is in the same directory as the sounds
            self._config['scriptDir'] = Path(os.path.dirname(os.path.abspath(__file__)))
            logWarning(
                f"Could not reliably determine main script directory, using fallback: {self._config['scriptDir']}")

        self._config['device'] = None  # Will be set by AsrModelHandler (local or remote info)
        self._config['actualSampleRate'] = self._config.get('sampleRate', 16000)  # Default/Initial
        self._config['actualChannels'] = self._config.get('channels', 1)  # Default/Initial

    def get(self, key, default=None):
        """Gets a configuration value."""
        return self._config.get(key, default)

    def set(self, key, value):
        """Sets or updates a configuration value."""
        self._config[key] = value

    def getAll(self):
        """Returns the entire configuration dictionary."""
        return self._config.copy()


# ==================================
# State Management
# ==================================
class StateManager:
    """Manages the dynamic state of the real-time transcriber."""

    def __init__(self, config):
        self.config = config
        self.isProgramActive = True  # Overall application loop control
        self.isRecordingActive = config.get('isRecordingActive', True)
        self.outputEnabled = config.get('outputEnabled', False)

        # Timing state
        self.programStartTime = time.time()
        self.lastActivityTime = time.time()  # Used for model unloading timeout (local models or server interaction)
        self.recordingStartTime = time.time() if self.isRecordingActive else 0
        self.lastValidTranscriptionTime = time.time()  # Used for consecutive idle timeout

    def _logDebug(self, message):
        logDebug(message, self.config.get('debugPrint'))

    # --- Getters ---
    def isRecording(self):
        return self.isRecordingActive

    def isOutputEnabled(self):
        return self.outputEnabled

    def shouldProgramContinue(self):
        return self.isProgramActive

    # --- Setters ---
    def startRecording(self):
        if not self.isRecordingActive:
            self._logDebug("Setting state to Recording: ON")
            self.isRecordingActive = True
            now = time.time()
            self.recordingStartTime = now
            self.lastActivityTime = now  # Mark activity for model manager/timeout checks
            self.lastValidTranscriptionTime = now  # Reset idle timer
            return True  # State changed
        return False  # No change

    def stopRecording(self):
        if self.isRecordingActive:
            self._logDebug("Setting state to Recording: OFF")
            self.isRecordingActive = False
            self.recordingStartTime = 0  # Reset session start time
            # Mark activity time when stopping recording as well, so server communication timeout resets
            self.lastActivityTime = time.time()
            return True  # State changed
        return False  # No change

    def toggleOutput(self):
        self.outputEnabled = not self.outputEnabled
        status = 'enabled' if self.outputEnabled else 'disabled'
        self._logDebug(f"Setting state Output: {status.upper()}")
        logInfo(f"Output {status}")
        # Mark activity when toggling output, might interact with server
        self.updateLastActivityTime()
        return self.outputEnabled  # Return new state

    def stopProgram(self):
        self._logDebug("Setting state Program Active: OFF")
        self.isProgramActive = False

    def updateLastActivityTime(self):
        """Updates the timestamp of the last significant activity (local processing or server interaction)."""
        self.lastActivityTime = time.time()
        # self._logDebug("Updated last activity time.") # Can be noisy

    def updateLastValidTranscriptionTime(self):
        """Updates the timestamp of the last valid transcription output."""
        self.lastValidTranscriptionTime = time.time()
        self._logDebug("Updated last valid transcription time (idle timer reset).")

    # --- Timeout Checks ---
    def checkRecordingTimeout(self):
        """Checks if the maximum recording session duration has been exceeded, treating 0 or less as no limit."""
        maxDuration = self.config.get('maxDurationRecording', 3600)
        # If maxDuration is 0 or negative, treat it as no limit
        if maxDuration <= 0:
            return False  # <<<--- THIS LINE IS CRITICAL ---

        # --- The rest only runs if maxDuration > 0 ---
        if not self.isRecordingActive or self.recordingStartTime == 0:
            return False

        elapsed = time.time() - self.recordingStartTime
        if elapsed >= maxDuration:
            # This log should now ONLY appear if maxDuration was > 0
            logInfo(f"Maximum recording session duration ({maxDuration}s) reached.")
            return True
        return False

    def checkIdleTimeout(self):
        """Checks if the consecutive idle time limit has been reached, treating 0 or less as no limit."""
        if not self.isRecordingActive:  # Only check if recording is supposed to be active
            return False

        idleTimeout = self.config.get('consecutiveIdleTime', 120)

        # <<<--- Start of Change (Optional but good practice) --->>>
        # If idleTimeout is 0 or negative, treat it as no limit
        if idleTimeout <= 0:
            return False  # Never time out
        # <<<--- End of Change --->>>

        silentFor = time.time() - self.lastValidTranscriptionTime
        if silentFor >= idleTimeout:
            logInfo(f"Consecutive idle time ({idleTimeout}s) reached.")
            # Action (stopping) should be triggered by the orchestrator
            return True
        return False

    def timeSinceLastActivity(self):
        """Calculates the time elapsed since the last recorded activity."""
        return time.time() - self.lastActivityTime

    def checkProgramTimeout(self):
        """Checks if the maximum program duration has been exceeded, treating 0 or less as no limit."""
        maxDuration = self.config.get('maxDurationProgramActive', 3600)
        if maxDuration <= 0:
            return False  # Never time out
        elapsed = time.time() - self.programStartTime
        if elapsed >= maxDuration:
            logInfo(f"Maximum program duration ({maxDuration}s) reached.")
            self.stopProgram()
            return True
        return False


# ==================================
# Model Lifecycle Management
# ==================================
class ModelLifecycleManager:
    """
    Handles automatic loading/unloading of the ASR model based on activity.
    Works for both local handlers (Whisper) and the remote client handler (NeMo).
    For remote, 'load'/'unload' interact with the server status/endpoints.
    """

    def __init__(self, config, stateManager, asrModelHandler, systemInteractionHandler):
        self.config = config
        self.stateManager = stateManager
        self.asrModelHandler = asrModelHandler  # Can be WhisperModelHandler or RemoteNemoClientHandler
        self.systemInteractionHandler = systemInteractionHandler
        self._logDebug = lambda msg: logDebug(msg, self.config.get('debugPrint'))

    def manageModelLifecycle(self):
        """
        Runs in a thread to monitor activity and load/unload the model (local or remote).
        """
        handlerType = type(self.asrModelHandler).__name__
        logInfo(f"Starting Model Lifecycle Manager thread (Handler: {handlerType}).")
        checkInterval = 10  # Seconds to wait between checks

        while self.stateManager.shouldProgramContinue():
            isRecording = self.stateManager.isRecording()
            # Use the handler's method to check loaded status (works for local/remote)
            modelIsCurrentlyLoaded = self.asrModelHandler.isModelLoaded()
            unloadTimeout = self.config.get('model_unloadTimeout', 1200)  # In seconds

            # --- Unload Condition ---
            # Unload if NOT recording AND model IS loaded AND timeout exceeded
            if not isRecording and modelIsCurrentlyLoaded and unloadTimeout > 0:
                timeInactive = self.stateManager.timeSinceLastActivity()
                if timeInactive >= unloadTimeout:
                    self._logDebug(
                        f"Model inactive for {timeInactive:.1f}s (>= {unloadTimeout}s), requesting unload...")
                    try:
                        self.asrModelHandler.unloadModel()  # Request unload (local or remote)
                        # Only play sound if unload seemed successful (modelLoaded becomes False)
                        if not self.asrModelHandler.isModelLoaded():
                            self.systemInteractionHandler.playNotification("modelUnloaded")
                        else:
                            logWarning(
                                "Unload requested, but handler still reports model as loaded.")
                    except Exception as e:
                        logError(f"Error during model unload request: {e}")
                # else:
                #     if self.config.get('debugPrint'):
                #          print(f"DEBUG: Model loaded but inactive. Time since last activity: {timeInactive:.1f}s / {unloadTimeout}s")

            # --- Load Condition ---
            # Load if recording IS active AND model IS NOT loaded
            elif isRecording and not modelIsCurrentlyLoaded:
                logInfo("Recording active but model not loaded. Triggering model load...")
                try:
                    self.asrModelHandler.loadModel()  # Request load (local or remote)
                    # If loading fails, modelLoaded will remain false, loop will retry later.
                except Exception as e:
                    logError(f"Error during model load request: {e}")
                # Update activity time after a load attempt (success or fail) to reset unload timer
                self.stateManager.updateLastActivityTime()

            # --- Periodic Check ---
            # Use a timed sleep that can be interrupted if the program stops
            startTime = time.time()
            while (
                    time.time() - startTime < checkInterval) and self.stateManager.shouldProgramContinue():
                time.sleep(0.5)  # Sleep in smaller chunks

        logInfo("Model Lifecycle Manager thread stopping.")
