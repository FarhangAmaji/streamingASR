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
import threading
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse

# Import application components
from audioProcesses import AudioHandler, RealTimeAudioProcessor
from managers import ConfigurationManager, StateManager, ModelLifecycleManager
from modelHandlers import WhisperModelHandler, RemoteNemoClientHandler
from systemInteractions import SystemInteractionHandler
from tasks import TranscriptionOutputHandler
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
            if platform.system() != "Windows":
                uniLogger.error(
                    "Remote Nvidia model selected, but not running on Windows. Automatic WSL server launching disabled.")
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
        """Prepares the command list needed to launch the wslNemoServer.py script in WSL."""
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
            # Extract the port from the server URL
            parsedUrl = urlparse(wslServerUrl)
            wslServerPort = parsedUrl.port
            if not wslServerPort:
                raise ValueError(f"Could not extract port from wslServerUrl: {wslServerUrl}")

            # Find the directory of the main script to locate wslNemoServer.py
            import __main__
            scriptDir = None
            if hasattr(__main__, '__file__') and __main__.__file__:
                scriptDir = Path(os.path.abspath(__main__.__file__)).parent
            else:
                uniLogger.warning(
                    "Could not reliably determine main script path, falling back to mainManager.py directory.")
                scriptDir = Path(os.path.dirname(os.path.abspath(__file__)))

            # Locate the WSL server script, with a fallback to the current working directory
            wslServerScriptFilename = "wslNemoServer.py"
            wslServerScriptPathWindows = scriptDir / wslServerScriptFilename
            uniDebugLogger.debug(f"Looking for WSL server script at: {wslServerScriptPathWindows}")
            if not wslServerScriptPathWindows.is_file():
                fallbackPath = Path.cwd() / wslServerScriptFilename
                uniDebugLogger.debug(f"Script not found, checking CWD: {fallbackPath}")
                if not fallbackPath.is_file():
                    raise FileNotFoundError(
                        f"WSL script '{wslServerScriptFilename}' not found in script dir or CWD.")
                wslServerScriptPathWindows = fallbackPath
                uniLogger.warning(f"Using WSL server script from CWD: {wslServerScriptPathWindows}")

            # Convert the Windows path to a WSL-compatible path (e.g., C:\... -> /mnt/c/...)
            wslServerScriptPathWsl = convertWindowsPathToWsl(wslServerScriptPathWindows)
            if not wslServerScriptPathWsl:
                raise ValueError(
                    f"Failed to convert Windows path to WSL path: {wslServerScriptPathWindows}")

            # Assemble the full command to be executed by subprocess.Popen
            commandBase = ["wsl.exe", "-d", wslDistro, "--"]
            commandInsideWsl = ["/usr/bin/python3", wslServerScriptPathWsl, "--model_name",
                                modelName, "--port", str(wslServerPort), "--load_on_start"]
            if useSudo:
                uniLogger.warning(
                    "Config 'wslUseSudo' is True. This requires passwordless sudo in WSL.")
                commandInsideWsl.insert(0, "sudo")

            self.wslLaunchCommand = commandBase + commandInsideWsl
            uniLogger.info("Prepared WSL server launch command successfully.")
            uniDebugLogger.debug(f"WSL Command List: {self.wslLaunchCommand}")
        except (FileNotFoundError, ValueError) as e:
            uniLogger.error(
                f"Error preparing WSL launch: {e}. Server will NOT be launched automatically.")
            self.wslLaunchCommand = []
        except Exception as e:
            uniLogger.error(f"Unexpected error preparing WSL launch command: {e}", excInfo=True)
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

        # Build a multi-line string for a clean log output
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
            f"Unload Model Inactive:{f'{unloadTimeout} s' if unloadTimeout > 0 else 'Disabled'}\n"
            f"Program Auto-Exit:    {f'{maxProgram} s' if maxProgram > 0 else 'Unlimited'}\n"
            f"-------------------------"
        )
        uniLogger.info(logMessage, indicatorName="APP_SETUP_INFO")

    def _launchWslServer(self) -> bool:
        """Launches the wslNemoServer.py script in a subprocess and waits for it to become reachable."""
        if not self.wslLaunchCommand:
            uniLogger.warning("WSL server launch command not available. Skipping automatic launch.")
            return False
        # If a process handle exists and the process is running, check if it's reachable
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

        if platform.system() != "Windows":
            uniLogger.error(
                "WSL server launch skipped: Cannot execute wsl.exe on non-Windows platform.")
            return False

        uniLogger.info("Attempting to launch WSL server...")
        uniDebugLogger.debug(f"Executing Popen with: {self.wslLaunchCommand}")
        try:
            # On Windows, create the process without a visible console window
            creationFlags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
            self.wslServerProcess = subprocess.Popen(
                self.wslLaunchCommand,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding='utf-8', errors='replace', creationflags=creationFlags, bufsize=1
            )
            uniLogger.info(
                f"WSL server process launched (PID: {self.wslServerProcess.pid}). Waiting for reachability...")
            serverReadyTimeout = self.config.get('wslServerReadyTimeout', 90.0)
            # Wait for the server to respond to status checks
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
            f"Waiting up to {timeoutSeconds:.1f}s for WSL server (PID: {pid}) to become reachable...")

        # Loop until timeout is reached
        while time.time() - startTime < timeoutSeconds:
            # Check if the process has exited prematurely
            if checkProcessFirst and self.wslServerProcess.poll() is not None:
                uniLogger.error(
                    f"WSL server process (PID: {pid}) exited prematurely (exit code: {self.wslServerProcess.returncode}).")
                self._logWslProcessOutputOnError()
                return False

            # Force a status check on the server
            _ = self.asrModelHandler.checkServerStatus(forceCheck=True)
            if self.asrModelHandler.serverReachable:
                uniDebugLogger.debug(f"Server (PID: {pid}) is reachable.")
                return True

            time.sleep(2.0)  # Wait before polling again

        uniLogger.warning(f"Timeout waiting for WSL server (PID: {pid}).")
        return False

    def _logWslProcessOutputOnError(self):
        """Reads and logs stdout/stderr from the WSL process, typically after an error or exit."""
        if not self.wslServerProcess: return
        pid = self.wslServerProcess.pid or "N/A"
        uniLogger.info(f"Attempting to read output from failed/exited WSL process (PID: {pid})...")
        try:
            # Communicate() waits for the process to terminate and reads all output
            stdoutData, _ = self.wslServerProcess.communicate(timeout=2.0)
            if stdoutData and stdoutData.strip():
                uniLogger.error(
                    f"--- Captured WSL Server Output (PID: {pid}) ---\n{stdoutData.strip()}",
                    indicatorName="WSL_SUBPROCESS_OUTPUT")
                # Provide hints for common errors found in the output
                if "sudo: a password is required" in stdoutData:
                    uniLogger.error(
                        "!!! Detected 'sudo password required'. Automatic launch failed. Configure passwordless sudo or set wslUseSudo=False.")
                if "Address already in use" in stdoutData:
                    uniLogger.error(
                        "!!! Detected 'Address already in use'. Check port 5001 in WSL.")
        except Exception as readError:
            uniLogger.warning(f"Exception reading WSL process output for PID {pid}: {readError}")

    def _terminateWslServer(self):
        """Terminates the launched WSL server process if it exists and is running."""
        if not self.wslServerProcess: return
        pid = self.wslServerProcess.pid or "N/A"
        # Check if the process is still running
        if self.wslServerProcess.poll() is None:
            uniLogger.info(f"Attempting to terminate running WSL server process (PID: {pid})...")
            try:
                # First, try a graceful shutdown
                self.wslServerProcess.terminate()
                self.wslServerProcess.wait(timeout=3.0)
                uniLogger.info(f"WSL server process (PID: {pid}) terminated gracefully.")
            except subprocess.TimeoutExpired:
                # If it doesn't terminate, force kill it
                uniLogger.warning(
                    f"WSL server (PID: {pid}) did not terminate gracefully. Forcing kill...")
                self.wslServerProcess.kill()
            except Exception as e:
                uniLogger.error(f"Error during termination of WSL server (PID: {pid}): {e}",
                                excInfo=True)
        # Log any final output from the process
        self._logWslProcessOutputOnError()
        self.wslServerProcess = None

    def _transcriptionWorkerLoop(self):
        """
        Worker loop running in a separate thread. Waits for audio data on the queue,
        calls the ASR handler, and processes the result via the output handler.
        """
        if not all([self.asrModelHandler, self.stateManager, self.outputHandler]):
            uniLogger.critical("Transcription worker cannot start: Missing required components.")
            return

        uniLogger.info(
            f"Starting Transcription Worker thread (Handler: {type(self.asrModelHandler).__name__}).")
        while self.stateManager.shouldProgramContinue():
            try:
                # Wait for an item on the transcription queue
                queueItem = self.transcriptionRequestQueue.get(timeout=1.0)
                if queueItem is None:  # A None item is a sentinel to stop the thread
                    uniDebugLogger.debug("Transcription worker received None sentinel, stopping.")
                    break

                audioData, sampleRate = queueItem
                # Only transcribe if the model is loaded
                if self.asrModelHandler.isModelLoaded():
                    duration = len(audioData) / sampleRate
                    uniDebugLogger.debug(f"Worker processing {duration:.2f}s audio segment...")
                    startTime = time.time()
                    transcriptionResult = self.asrModelHandler.transcribeAudioSegment(audioData,
                                                                                      sampleRate)
                    uniDebugLogger.debug(f"ASR inference took {time.time() - startTime:.3f}s.")
                    # Send the result to the output handler for filtering and formatting
                    self.outputHandler.processTranscriptionResult(transcriptionResult, audioData)
                else:
                    uniLogger.warning("Transcription worker skipped segment: ASR model not loaded.")
                self.transcriptionRequestQueue.task_done()
            except queue.Empty:
                # This is expected when no audio is ready for transcription
                continue
            except Exception as e:
                uniLogger.error(f"!!! ERROR in Transcription Worker: {e}", excInfo=True)
                time.sleep(1)  # Pause briefly after an error
        uniLogger.info("Transcription Worker thread stopping.")

    def _startBackgroundThreads(self):
        """Starts all necessary background threads for hotkeys, model management, and transcription."""
        uniDebugLogger.debug("Starting background threads...")
        self.threads = []

        # A wrapper to catch and log exceptions that occur inside threads
        def threadWrapper(targetFunc, threadName, *args, **kwargs):
            uniDebugLogger.debug(f"Thread '{threadName}' starting...")
            try:
                targetFunc(*args, **kwargs)
                uniDebugLogger.debug(f"Thread '{threadName}' finished normally.")
            except Exception as e:
                uniLogger.critical(f"!!! EXCEPTION in thread '{threadName}': {e}", excInfo=True)
                # If a critical thread fails, signal the whole program to stop
                if threadName in ["KeyboardMonitorThread", "TranscriptionWorkerThread"]:
                    self.stateManager.stopProgram()

        threadTargets = {
            "KeyboardMonitorThread": (
                self.systemInteractionHandler.monitorKeyboardShortcuts, (self,)),
            "ModelManagerThread": (self.modelLifecycleManager.manageModelLifecycle, ()),
            "TranscriptionWorkerThread": (self._transcriptionWorkerLoop, ()),
        }
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
        """Toggles the recording state. Called by the hotkey monitor."""
        if self.stateManager.isRecording():
            if self.stateManager.stopRecording():
                self.systemInteractionHandler.playNotification("recordingOff")
                uniLogger.info("Recording stopped.")
                self.realTimeProcessor.clearBuffer()
                self.audioHandler.clearQueue()
        else:
            if self.stateManager.startRecording():
                self.systemInteractionHandler.playNotification("recordingOn")
                uniLogger.info("Recording started.")

    def toggleOutput(self):
        """Toggles the text output state. Called by the hotkey monitor."""
        newState = self.stateManager.toggleOutput()
        if newState:
            self.systemInteractionHandler.playNotification("outputEnabled")
        else:
            self.systemInteractionHandler.playNotification("outputDisabled")
            self.realTimeProcessor.clearBufferIfOutputDisabled()

    def forceTranscribeCurrentBuffer(self):
        """Forces the current audio buffer to be sent for transcription. Called by the hotkey monitor."""
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
                duration = len(audioData) / sampleRate
                uniLogger.info(f"Forced transcription: Queued {duration:.2f}s of audio.")
                self.stateManager.updateLastActivityTime()
            except queue.Full:
                uniLogger.warning("Forced transcription: Queue is full. Audio dropped.")
        else:
            uniLogger.info("Force transcription: Audio buffer was empty.")

    def _cleanup(self):
        """Cleans up all resources: stops threads, audio stream, model, and WSL server."""
        uniLogger.info("Initiating orchestrator cleanup...")
        if self.stateManager: self.stateManager.stopProgram()

        # Send a sentinel value to the transcription queue to unblock the worker thread
        try:
            self.transcriptionRequestQueue.put(None, block=False)
        except queue.Full:
            pass  # If full, the worker will time out and stop anyway

        # Stop all components
        if self.audioHandler: self.audioHandler.stopStream()
        self._terminateWslServer()
        if self.asrModelHandler: self.asrModelHandler.cleanup()
        if self.systemInteractionHandler: self.systemInteractionHandler.cleanup()

        # Wait for all background threads to finish
        uniLogger.info("Joining background threads...")
        for t in self.threads:
            if t.is_alive(): t.join(timeout=2.0)
        uniLogger.info("Cleanup complete.")

    def _runInitialSetup(self):
        """Handles the initial setup sequence: launching WSL server, loading the model, starting threads, and the audio stream."""
        uniLogger.info("Running initial setup...")
        serverReachable = True
        # If using a remote model, launch and check the WSL server first
        if isinstance(self.asrModelHandler, RemoteNemoClientHandler):
            uniLogger.info("Remote handler detected, checking WSL server...")
            serverReachable = self._launchWslServer()
            if not serverReachable:
                uniLogger.error(
                    "WSL NeMo server check failed. Remote transcription requires manual server start.")

        # If the server is reachable (or if using a local model), attempt to load the model
        if serverReachable:
            uniLogger.info("Attempting initial ASR model load...")
            if not self.asrModelHandler.loadModel():
                uniLogger.error("Initial ASR model load failed.")
                # If a local model fails to load, it's a critical error, so stop the program
                if isinstance(self.asrModelHandler, WhisperModelHandler):
                    uniLogger.critical("Local Whisper model failed to load. Aborting.")
                    self.stateManager.stopProgram()
                    return

        # Start background threads if no critical errors have occurred
        if self.stateManager.shouldProgramContinue():
            self._startBackgroundThreads()

        # If configured to start recording immediately, start the audio stream
        if self.stateManager.isRecording() and self.stateManager.shouldProgramContinue():
            uniLogger.info("Initial state is recording, starting audio stream...")
            if not self.audioHandler.startStream():
                uniLogger.critical("Failed to start audio stream initially. Disabling recording.")
                self.stateManager.stopRecording()
        uniLogger.info("Initial setup phase complete.")

    def _mainLoop(self):
        """The core processing loop of the orchestrator, handling state checks and audio processing."""
        while self.stateManager.shouldProgramContinue():
            # Check for program-level timeouts
            if self.stateManager.checkProgramTimeout():
                uniLogger.info("Program timeout reached. Stopping.")
                break

            # Clear audio buffer if text output has been disabled
            self.realTimeProcessor.clearBufferIfOutputDisabled()
            # Check for recording-specific timeouts (max duration, idle)
            if self.stateManager.isRecording():
                if self.stateManager.checkRecordingTimeout():
                    uniLogger.info("Max recording duration reached. Stopping recording.")
                    self.toggleRecording()
                elif self.stateManager.checkIdleTimeout():
                    uniLogger.info("Idle timeout reached. Stopping recording.")
                    self.toggleRecording()

            # Manage the audio stream's lifecycle based on the recording state
            shouldRecord = self.stateManager.isRecording()
            isStreamActive = self.audioHandler.stream is not None and self.audioHandler.stream.active
            if shouldRecord and not isStreamActive:
                if not self.audioHandler.startStream():
                    uniLogger.error("Failed to restart audio stream. Disabling recording.")
                    self.stateManager.stopRecording()
            elif not shouldRecord and isStreamActive:
                self.audioHandler.stopStream()

            # Process audio chunks from the input queue
            if self.stateManager.isRecording():
                maxChunks = 50  # Process up to 50 chunks per loop to stay responsive
                for _ in range(maxChunks):
                    chunk = self.audioHandler.getAudioChunk()
                    if chunk is None: break  # No more chunks in the queue
                    self.realTimeProcessor.processIncomingChunk(chunk)

            # Check if a transcription should be triggered
            if self.stateManager.isOutputEnabled():
                audioToTranscribe = self.realTimeProcessor.checkTranscriptionTrigger()
                if audioToTranscribe is not None and audioToTranscribe.size > 0:
                    try:
                        self.transcriptionRequestQueue.put(
                            (audioToTranscribe, self.config.get('actualSampleRate')), timeout=0.5)
                    except queue.Full:
                        uniLogger.warning("Transcription queue is full. Audio segment dropped.")

            time.sleep(0.01)  # Brief sleep to prevent 100% CPU usage

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
            # Ensure cleanup is always called
            self.stateManager.stopProgram()
            uniLogger.info("Exiting main loop.")
            self._cleanup()
            uniLogger.info("Orchestrator run finished.")
