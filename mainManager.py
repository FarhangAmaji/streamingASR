# mainManager.py
# ==============================================================================
# Real-Time Speech-to-Text - Main Orchestrator
# ==============================================================================
#
# Purpose:
# - Contains the main `SpeechToTextOrchestrator` class.
# - Manages the overall application lifecycle and coordination of components.
# - Determines ASR handler (local/remote) based on config.
# - Launches WSL server process if required for remote NeMo models.
# - Starts and manages background threads for transcription, keyboard monitoring,
#   and model lifecycle.
# - Handles main application loop, state transitions, and cleanup.
# ==============================================================================
import os
import platform
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse

# Import application components
from audioProcesses import AudioHandler, RealTimeAudioProcessor
from managers import ConfigurationManager, StateManager, ModelLifecycleManager
from systemInteractions import SystemInteractionHandler
from tasks import TranscriptionOutputHandler
# Import the specific model handlers from their new, separated locations
from modelHandlers import WhisperModelHandler
from wslServer.clientHandler import RemoteNemoClientHandler

# Import helper functions and the new logger instances
from utils.utils import convertWindowsPathToWsl
from utils.loggerInstance import uniLogger, uniDebugLogger


# ==================================
# Main Orchestrator Class
# ==================================
class SpeechToTextOrchestrator:
    """
    Main class orchestrating the real-time speech-to-text process.
    Connects and manages all components (Audio, State, Processing, Output, System, Model).
    Handles transcription processing in a separate thread and manages WSL server lifecycle if needed.
    """

    def __init__(self, **userConfig):
        """Initializes the orchestrator by setting up config and all components."""
        # --- Basic Setup ---
        self.config = ConfigurationManager(**userConfig)
        uniDebugLogger.debug("Initializing SpeechToText Orchestrator...")
        # Initialize attributes that will be set by helper methods
        self.stateManager = None
        self.systemInteractionHandler = None
        self.audioHandler = None
        self.realTimeProcessor = None
        self.outputHandler = None
        self.asrModelHandler = None
        self.modelLifecycleManager = None
        self.wslServerProcess = None  # Handle for the launched WSL process
        self.wslLaunchCommand = []  # Stores the command to launch the WSL server
        self.transcriptionRequestQueue = queue.Queue()
        self.threads = []

        # --- Instantiate Components & Handlers ---
        self._initializeCoreComponents()
        self._initializeAsrHandler()  # Sets self.asrModelHandler

        # Ensure ASR handler was successfully initialized before proceeding
        if not self.asrModelHandler:
            msg = "ASR Handler could not be initialized. Check configuration. Cannot continue."
            uniLogger.critical(msg)
            raise RuntimeError(msg)

        # Ensure SystemInteractionHandler is ready before initializing the model manager
        if not self.systemInteractionHandler:
            msg = "SystemInteractionHandler not initialized before ModelLifecycleManager."
            uniLogger.critical(msg)
            raise RuntimeError(msg)

        # Instantiate Model Lifecycle Manager (depends on ASR Handler)
        self.modelLifecycleManager = ModelLifecycleManager(
            self.config, self.stateManager, self.asrModelHandler, self.systemInteractionHandler
        )
        # --- Final Steps ---
        self._printInitialInstructions()
        uniLogger.info("Orchestrator initialization complete.")

    def _initializeCoreComponents(self):
        """Initializes the core state and interaction components."""
        uniDebugLogger.debug("Initializing core components...")
        self.stateManager = StateManager(self.config)
        self.systemInteractionHandler = SystemInteractionHandler(self.config)
        self.outputHandler = TranscriptionOutputHandler(self.config, self.stateManager,
                                                        self.systemInteractionHandler)
        self.audioHandler = AudioHandler(self.config, self.stateManager)
        self.realTimeProcessor = RealTimeAudioProcessor(self.config, self.stateManager)
        uniDebugLogger.debug("Core components initialized.")

    def _initializeAsrHandler(self):
        """Determines and initializes the appropriate ASR handler (local Whisper or remote NeMo)."""
        uniDebugLogger.debug("Initializing ASR handler...")
        modelName = self.config.get('modelName', '')
        modelNameLower = modelName.lower()
        self.wslServerProcess = None
        self.wslLaunchCommand = []
        # If model name is from Nvidia, use the remote client handler for WSL
        if modelNameLower.startswith("nvidia/"):
            uniLogger.info(
                f"Configured Nvidia model: '{modelName}'. Preparing RemoteNemoClientHandler.")
            if platform.system() == "Windows":
                self._prepareWslLaunchCommand(modelName)
            self.asrModelHandler = RemoteNemoClientHandler(self.config)
        # Otherwise, use the local Whisper handler
        elif modelName:
            uniLogger.info(
                f"Configured non-Nvidia model: '{modelName}'. Using local WhisperModelHandler.")
            self.asrModelHandler = WhisperModelHandler(self.config)
        else:
            # If no model name is specified, the application cannot run.
            uniLogger.critical(
                "No 'modelName' specified in configuration. Cannot initialize ASR handler.")
            self.asrModelHandler = None

        if self.asrModelHandler:
            uniDebugLogger.debug(f"ASR Handler initialized: {type(self.asrModelHandler).__name__}")
        else:
            uniLogger.error("ASR Handler initialization failed.")

    def _prepareWslLaunchCommand(self, modelName):
        """Prepares the command list needed to launch the wsl_server/server_app.py script in WSL."""
        uniDebugLogger.debug("Preparing WSL server launch command...")
        self.wslLaunchCommand = []
        wslServerUrl = self.config.get('wslServerUrl')
        wslDistro = self.config.get('wslDistributionName')
        useSudo = self.config.get('wslUseSudo', False)
        if not wslServerUrl or not wslDistro:
            uniLogger.error(
                "Config error: 'wslServerUrl' & 'wslDistributionName' required for auto WSL launch. Server will NOT be launched.")
            return
        if platform.system() != "Windows":
            uniLogger.warning("WSL launch command prep skipped: Not on Windows.")
            return
        try:
            # --- Robust Path Finding ---
            # Assume the project root is the directory where the main script is located.
            if hasattr(sys, 'argv') and sys.argv[0]:
                project_root = Path(sys.argv[0]).parent.resolve()
            else:
                # Fallback if sys.argv is not available, uses the current working directory
                project_root = Path.cwd().resolve()

            # --- START: SOLUTION FOR ModuleNotFoundError ---
            # Convert the project root path from Windows format to WSL format for PYTHONPATH
            project_root_wsl = convertWindowsPathToWsl(project_root)
            if not project_root_wsl:
                raise ValueError(f"Failed to convert project root to WSL path: {project_root}")
            uniDebugLogger.debug(f"Project root path for PYTHONPATH in WSL: {project_root_wsl}")
            # --- END: SOLUTION ---

            wslServerScriptFilename = "wslServer/serverApp.py"
            wslServerScriptPathWindows = project_root / wslServerScriptFilename
            uniDebugLogger.debug(
                f"Resolved absolute server script path: {wslServerScriptPathWindows}")

            if not wslServerScriptPathWindows.is_file():
                raise FileNotFoundError(
                    f"WSL script '{wslServerScriptPathWindows.name}' not found at the expected path: {wslServerScriptPathWindows}")

            # Convert the Windows path to a WSL-compatible path (e.g., C:\... -> /mnt/c/...)
            wslServerScriptPathWsl = convertWindowsPathToWsl(wslServerScriptPathWindows)
            if not wslServerScriptPathWsl:
                raise ValueError(
                    f"Failed to convert Windows path to WSL path: {wslServerScriptPathWindows}")

            parsedUrl = urlparse(wslServerUrl)
            wslServerPort = parsedUrl.port
            if not wslServerPort:
                raise ValueError(f"Could not extract port from wslServerUrl: {wslServerUrl}")

            commandBase = ["wsl.exe", "-d", wslDistro, "--"]

            # --- START: SOLUTION FOR ModuleNotFoundError ---
            # Use 'env' to set the PYTHONPATH environment variable for the python command.
            # This tells the Python interpreter inside WSL where to look for modules like 'utils'.
            commandInsideWsl = ["env", f"PYTHONPATH={project_root_wsl}", "/usr/bin/python3",
                                wslServerScriptPathWsl, "--model_name",
                                modelName, "--port", str(wslServerPort), "--load_on_start"]
            # --- END: SOLUTION ---

            if useSudo:
                uniLogger.warning(
                    "Config 'wslUseSudo' is True. This requires passwordless sudo in WSL.")
                # Insert 'sudo' after the 'env' command
                commandInsideWsl.insert(2, "sudo")

            self.wslLaunchCommand = commandBase + commandInsideWsl
            uniLogger.info("Prepared WSL server launch command successfully.")
            uniDebugLogger.debug(f"WSL Command List: {self.wslLaunchCommand}")

        except Exception as e:
            uniLogger.error(f"Error preparing WSL launch: {e}", excInfo=True)
            self.wslLaunchCommand = []

    def _printInitialInstructions(self):
        """Prints a summary of the application setup and user instructions based on configuration."""
        if not all([self.asrModelHandler, self.config, self.systemInteractionHandler]):
            uniLogger.warning("Cannot print initial instructions: components not initialized.")
            return

        handlerType = type(self.asrModelHandler).__name__
        maxRec = self.config.get('maxDurationRecording', 0)
        idleTime = self.config.get('consecutiveIdleTime', 0)
        unloadTimeout = self.config.get('model_unloadTimeout', 0)
        maxProgram = self.config.get('maxDurationProgramActive', 0)

        logMessage = (
            "\n--- Application Setup ---\n"
            f"Mode:                 {self.config.get('transcriptionMode', 'N/A')}\n"
            f"ASR Model:            {self.config.get('modelName', 'N/A')}\n"
            f"  Handler:            {handlerType}\n"
            f"  Target Device:      {self.asrModelHandler.getDevice()}\n"
        )
        if handlerType == 'RemoteNemoClientHandler':
            logMessage += (
                f"  WSL Server URL:     {self.config.get('wslServerUrl', 'Not Set!')}\n"
                f"  WSL Distro:         {self.config.get('wslDistributionName', 'Not Set!')}\n"
                f"  WSL Use Sudo:       {self.config.get('wslUseSudo', False)}\n"
            )
        logMessage += (
            f"Audio Device:         ID={self.config.get('deviceId', 'Default')}, Rate={self.config.get('actualSampleRate', 'N/A')}Hz, Channels={self.config.get('actualChannels', 'N/A')}\n"
            f"--- Hotkeys ---\n"
            f"Toggle Recording:     '{self.config.get('recordingToggleKey', 'N/A')}'\n"
            f"Toggle Text Output:   '{self.config.get('outputToggleKey', 'N/A')}' (Method: {self.systemInteractionHandler.textOutputMethod})\n"
            f"Force Transcription:  '{self.config.get('forceTranscriptionKey', 'N/A')}'\n"
            f"--- Timeouts ---\n"
            f"Max Recording:        {f'{maxRec} s' if maxRec > 0 else 'Unlimited'}\n"
            f"Stop Rec After Idle:  {f'{idleTime} s' if idleTime > 0 else 'Disabled'}\n"
            f"Unload Model Inactive: {f'{unloadTimeout} s' if unloadTimeout > 0 else 'Disabled'}\n"
            f"Program Auto-Exit:    {f'{maxProgram} s' if maxProgram > 0 else 'Unlimited'}\n"
            f"-------------------------"
        )
        uniLogger.info(logMessage, indicatorName="APP_SETUP_INFO")

    def _launchWslServer(self) -> bool:
        """Launches the server script in a subprocess and waits for it to become reachable."""
        if not self.wslLaunchCommand:
            uniLogger.warning("WSL server launch command not available. Skipping automatic launch.")
            return False

        if self.wslServerProcess and self.wslServerProcess.poll() is None:
            uniLogger.info(
                f"WSL server process (PID: {self.wslServerProcess.pid}) already running. Checking reachability...")
            if self._waitForServerReachable(self.config.get('wslServerReadyTimeout', 90.0),
                                            checkProcessFirst=False):
                uniLogger.info("Existing WSL server is reachable.")
                return True
            else:
                uniLogger.warning(
                    "Existing WSL server process did not become reachable. Terminating.")
                self._terminateWslServer()

        uniLogger.info("Attempting to launch WSL server...")
        uniDebugLogger.debug(f"Executing Popen with: {self.wslLaunchCommand}")
        try:
            creationFlags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
            self.wslServerProcess = subprocess.Popen(
                self.wslLaunchCommand,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding='utf-8', errors='replace', creationflags=creationFlags, bufsize=1
            )
            uniLogger.info(
                f"WSL server process launched (PID: {self.wslServerProcess.pid}). Waiting for reachability...")
            serverReadyTimeout = self.config.get('wslServerReadyTimeout', 90.0)

            if self._waitForServerReachable(serverReadyTimeout, checkProcessFirst=True):
                uniLogger.info(f"WSL server became reachable within {serverReadyTimeout}s.")
                return True
            else:
                uniLogger.error(
                    f"WSL server did not become reachable within {serverReadyTimeout}s or exited.")
                self._terminateWslServer()
                return False
        except (FileNotFoundError, PermissionError) as e:
            uniLogger.error(f"Error launching WSL server: {e}. Is WSL installed and in PATH?")
            self.wslServerProcess = None
            return False
        except Exception as e:
            uniLogger.error(f"Unexpected error launching WSL server: {e}", excInfo=True)
            self._terminateWslServer()
            return False

    def _waitForServerReachable(self, timeoutSeconds: float, checkProcessFirst: bool) -> bool:
        """Polls the WSL server's /status endpoint until it responds successfully or a timeout occurs."""
        startTime = time.time()
        if not isinstance(self.asrModelHandler,
                          RemoteNemoClientHandler) or not self.wslServerProcess:
            uniLogger.error(
                "Cannot wait for server: Incorrect ASR handler or no WSL process handle.")
            return False

        pid = self.wslServerProcess.pid or "N/A"
        uniDebugLogger.debug(
            f"Waiting up to {timeoutSeconds:.1f}s for server (PID: {pid}) to become reachable...")

        while time.time() - startTime < timeoutSeconds:
            if checkProcessFirst and self.wslServerProcess.poll() is not None:
                uniLogger.error(
                    f"WSL server process (PID: {pid}) exited prematurely (code: {self.wslServerProcess.returncode}).")
                self._logWslProcessOutputOnError()
                return False

            _ = self.asrModelHandler.checkServerStatus(forceCheck=True)
            if self.asrModelHandler.serverReachable:
                uniDebugLogger.debug(f"Server (PID: {pid}) is reachable.")
                return True

            time.sleep(2.0)

        uniLogger.warning(f"Timeout waiting for WSL server (PID: {pid}).")
        return False

    def _logWslProcessOutputOnError(self):
        """Reads and logs stdout/stderr from the WSL process, typically after an error or exit."""
        if not self.wslServerProcess: return
        pid = self.wslServerProcess.pid or "N/A"
        uniLogger.info(f"Attempting to read output from failed/exited WSL process (PID: {pid})...")
        try:
            # Use communicate() to get all output after the process has terminated
            stdoutData, _ = self.wslServerProcess.communicate(timeout=2.0)
            if stdoutData and stdoutData.strip():
                uniLogger.error(
                    f"--- Captured WSL Server Output (PID: {pid}) ---\n{stdoutData.strip()}",
                    indicatorName="WSL_SUBPROCESS_OUTPUT")
                # Provide hints for common errors found in the output
                if "sudo: a password is required" in stdoutData:
                    uniLogger.error("!!! Detected 'sudo password required'. Launch failed.")
                if "Address already in use" in stdoutData:
                    uniLogger.error(
                        "!!! Detected 'Address already in use'. Check port in WSL.")
        except Exception as readError:
            uniLogger.warning(f"Exception reading WSL process output for PID {pid}: {readError}")

    def _terminateWslServer(self):
        """Terminates the launched WSL server process if it exists and is running."""
        if not self.wslServerProcess or self.wslServerProcess.poll() is not None:
            return

        pid = self.wslServerProcess.pid
        uniLogger.info(f"Attempting to terminate running WSL server process (PID: {pid})...")
        try:
            self.wslServerProcess.terminate()
            self.wslServerProcess.wait(timeout=3.0)
            uniLogger.info(f"WSL server process (PID: {pid}) terminated gracefully.")
        except subprocess.TimeoutExpired:
            uniLogger.warning(f"WSL server (PID: {pid}) did not terminate. Forcing kill...")
            self.wslServerProcess.kill()
        except Exception as e:
            uniLogger.error(f"Error during termination of WSL server (PID: {pid}): {e}",
                            excInfo=True)

        self._logWslProcessOutputOnError()
        self.wslServerProcess = None

    def _transcriptionWorkerLoop(self):
        """Worker loop running in a separate thread for transcription."""
        uniLogger.info(
            f"Starting Transcription Worker thread (Handler: {type(self.asrModelHandler).__name__}).")
        while self.stateManager.shouldProgramContinue():
            try:
                queueItem = self.transcriptionRequestQueue.get(timeout=1.0)
                if queueItem is None:
                    uniDebugLogger.debug("Transcription worker received None sentinel, stopping.")
                    break

                audioData, sampleRate = queueItem
                if self.asrModelHandler.isModelLoaded():
                    transcriptionResult = self.asrModelHandler.transcribeAudioSegment(audioData,
                                                                                      sampleRate)
                    self.outputHandler.processTranscriptionResult(transcriptionResult, audioData)
                else:
                    uniLogger.warning("Transcription worker skipped segment: ASR model not loaded.")
                self.transcriptionRequestQueue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                uniLogger.error(f"!!! ERROR in Transcription Worker: {e}", excInfo=True)
        uniLogger.info("Transcription Worker thread stopping.")

    def _startBackgroundThreads(self):
        """Starts all necessary background threads."""
        uniDebugLogger.debug("Starting background threads...")

        # A wrapper to catch and log exceptions that occur inside threads
        def threadWrapper(targetFunc, threadName, *args, **kwargs):
            uniDebugLogger.debug(f"Thread '{threadName}' starting...")
            try:
                targetFunc(*args, **kwargs)
                uniDebugLogger.debug(f"Thread '{threadName}' finished normally.")
            except Exception as e:
                uniLogger.critical(f"!!! EXCEPTION in thread '{threadName}': {e}", excInfo=True)
                if threadName in ["KeyboardMonitorThread", "TranscriptionWorkerThread"]:
                    self.stateManager.stopProgram()

        threadTargets = {
            "KeyboardMonitorThread": (
                self.systemInteractionHandler.monitorKeyboardShortcuts, (self,)),
            "ModelManagerThread": (self.modelLifecycleManager.manageModelLifecycle, ()),
            "TranscriptionWorkerThread": (self._transcriptionWorkerLoop, ()),
        }
        self.threads = []
        for name, (target, args) in threadTargets.items():
            thread = threading.Thread(target=threadWrapper, args=(target, name) + args, name=name,
                                      daemon=True)
            self.threads.append(thread)
            thread.start()

        time.sleep(0.1)  # Give threads a moment to start
        if any(not t.is_alive() for t in self.threads):
            uniLogger.warning(
                "One or more background threads may have failed to start or exited immediately.")

    def toggleRecording(self):
        """Toggles the recording state."""
        if self.stateManager.isRecording():
            if self.stateManager.stopRecording():
                self.systemInteractionHandler.playNotification("recordingOff")
                uniLogger.info("Recording stopped.")
        else:
            if self.stateManager.startRecording():
                self.systemInteractionHandler.playNotification("recordingOn")
                uniLogger.info("Recording started.")

    def toggleOutput(self):
        """Toggles the text output state."""
        newState = self.stateManager.toggleOutput()
        self.systemInteractionHandler.playNotification(
            "outputEnabled" if newState else "outputDisabled")
        if not newState:
            self.realTimeProcessor.clearBufferIfOutputDisabled()

    def forceTranscribeCurrentBuffer(self):
        """Forces the current audio buffer to be sent for transcription."""
        uniLogger.info("Force transcription action triggered by hotkey.")
        if not self.stateManager.isOutputEnabled():
            uniLogger.info("Force transcription skipped: Output is disabled.")
            return

        audioData = self.realTimeProcessor.getAudioBufferCopyAndClear()
        if audioData is not None and audioData.size > 0:
            sampleRate = self.config.get('actualSampleRate')
            try:
                # Put the audio data onto the queue for the worker thread
                self.transcriptionRequestQueue.put((audioData, sampleRate), block=True, timeout=0.5)
                uniLogger.info(f"Forced transcription: Queued audio.")
                self.stateManager.updateLastActivityTime()
            except queue.Full:
                uniLogger.warning("Forced transcription: Queue is full. Audio dropped.")
        else:
            uniLogger.info("Force transcription: Audio buffer was empty.")

    def _cleanup(self):
        """Cleans up all resources."""
        uniLogger.info("Initiating orchestrator cleanup...")
        if self.stateManager: self.stateManager.stopProgram()
        try:
            self.transcriptionRequestQueue.put(None, block=False)
        except queue.Full:
            pass

        if self.audioHandler: self.audioHandler.stopStream()
        self._terminateWslServer()
        if self.asrModelHandler: self.asrModelHandler.cleanup()
        if self.systemInteractionHandler: self.systemInteractionHandler.cleanup()

        uniLogger.info("Joining background threads...")
        for t in self.threads:
            if t.is_alive(): t.join(timeout=2.0)
        uniLogger.info("Cleanup complete.")

    def _runInitialSetup(self):
        """Handles the initial setup sequence."""
        uniLogger.info("Running initial setup...")
        serverReachable = True
        # If using a remote model, launch and check the WSL server first
        if isinstance(self.asrModelHandler, RemoteNemoClientHandler):
            uniLogger.info("Remote handler detected, checking WSL server...")
            serverReachable = self._launchWslServer()
            if not serverReachable:
                uniLogger.error("WSL NeMo server check failed.")

        # If the server is reachable (or if using a local model), attempt to load the model
        if serverReachable:
            uniLogger.info("Attempting initial ASR model load...")
            if not self.asrModelHandler.loadModel():
                uniLogger.error("Initial ASR model load failed.")
                if isinstance(self.asrModelHandler, WhisperModelHandler):
                    uniLogger.critical("Local Whisper model failed to load. Aborting.")
                    self.stateManager.stopProgram()
                    return

        # Start background threads if no critical errors have occurred
        if self.stateManager.shouldProgramContinue():
            self._startBackgroundThreads()

        if self.stateManager.isRecording() and self.stateManager.shouldProgramContinue():
            uniLogger.info("Initial state is recording, starting audio stream...")
            if not self.audioHandler.startStream():
                uniLogger.critical("Failed to start audio stream initially. Disabling recording.")
                self.stateManager.stopRecording()
        uniLogger.info("Initial setup phase complete.")

    def _mainLoop(self):
        """The core processing loop of the orchestrator."""
        while self.stateManager.shouldProgramContinue():
            if self.stateManager.checkProgramTimeout():
                uniLogger.info("Program timeout reached. Stopping.")
                break

            self.realTimeProcessor.clearBufferIfOutputDisabled()
            if self.stateManager.isRecording():
                if self.stateManager.checkRecordingTimeout() or self.stateManager.checkIdleTimeout():
                    uniLogger.info("A recording timeout was reached. Stopping recording.")
                    self.toggleRecording()

            shouldRecord = self.stateManager.isRecording()
            isStreamActive = self.audioHandler.stream is not None and self.audioHandler.stream.active
            if shouldRecord and not isStreamActive:
                if not self.audioHandler.startStream():
                    uniLogger.error("Failed to restart audio stream. Disabling recording.")
                    self.stateManager.stopRecording()
            elif not shouldRecord and isStreamActive:
                self.audioHandler.stopStream()

            if self.stateManager.isRecording():
                maxChunks = 50
                for _ in range(maxChunks):
                    chunk = self.audioHandler.getAudioChunk()
                    if chunk is None: break
                    self.realTimeProcessor.processIncomingChunk(chunk)

            if self.stateManager.isOutputEnabled():
                audioToTranscribe = self.realTimeProcessor.checkTranscriptionTrigger()
                if audioToTranscribe is not None and audioToTranscribe.size > 0:
                    try:
                        self.transcriptionRequestQueue.put(
                            (audioToTranscribe, self.config.get('actualSampleRate')), timeout=0.5)
                    except queue.Full:
                        uniLogger.warning("Transcription queue is full. Audio segment dropped.")

            time.sleep(0.01)

    def run(self):
        """Main execution entry point that wraps the setup and main loop."""
        uniLogger.info("Starting main orchestrator...")
        try:
            self._runInitialSetup()
            if self.stateManager.shouldProgramContinue():
                uniLogger.info("Entering main processing loop...")
                self._mainLoop()
        except KeyboardInterrupt:
            uniLogger.info("\nKeyboardInterrupt received. Stopping application...")
        except Exception as e:
            uniLogger.critical(f"\n!!! CRITICAL UNHANDLED ERROR IN MAIN LOOP: {e}", excInfo=True)
        finally:
            if self.stateManager: self.stateManager.stopProgram()
            uniLogger.info("Exiting main loop.")
            self._cleanup()
            uniLogger.info("Orchestrator run finished.")
