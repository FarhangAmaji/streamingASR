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
import traceback  # Retained for explicit traceback.format_exc()
from pathlib import Path
from urllib.parse import urlparse

# Import application components
from audioProcesses import AudioHandler, RealTimeAudioProcessor
from managers import ConfigurationManager, StateManager, ModelLifecycleManager
from modelHandlers import WhisperModelHandler, RemoteNemoClientHandler
from systemInteractions import SystemInteractionHandler
from tasks import TranscriptionOutputHandler
# Import logging helpers from utils
from utils import logWarning, logDebug, convertWindowsPathToWsl, logInfo, logError, logCritical


# ==================================
# Main Orchestrator Class
# ==================================
class SpeechToTextOrchestrator:
    """
    Main class orchestrating the real-time speech-to-text process.
    Connects and manages all components (Audio, State, Processing, Output, System, Model).
    Determines ASR handler (local/remote) based on config.
    Handles transcription processing in a separate thread.
    Manages WSL server lifecycle if needed.
    """

    def __init__(self, **userConfig):
        """Initializes the orchestrator by setting up config and calling helper methods."""
        # --- Basic Setup ---
        self.config = ConfigurationManager(**userConfig)
        logDebug("Initializing SpeechToText Orchestrator...")
        # Initialize attributes that might be set later or during setup methods
        self.stateManager = None
        self.systemInteractionHandler = None
        self.audioHandler = None
        self.realTimeProcessor = None
        self.outputHandler = None
        self.asrModelHandler = None
        self.modelLifecycleManager = None
        self.wslServerProcess = None  # Handle for the launched WSL process
        self.wslLaunchCommand = []  # Store the command to launch WSL server
        self.transcriptionRequestQueue = queue.Queue()
        self.threads = []
        # --- Instantiate Components & Handlers ---
        self._initializeCoreComponents()
        self._initializeAsrHandler()  # Sets self.asrModelHandler and prepares self.wslLaunchCommand
        # Ensure ASR handler was successfully initialized before proceeding
        if not self.asrModelHandler:
            logCritical(
                "ASR Handler could not be initialized. Check configuration and logs. Cannot continue.")
            raise RuntimeError(
                "ASR Handler could not be initialized. Check configuration and logs. Cannot continue.")
        # --- Instantiate Model Lifecycle Manager (depends on ASR Handler) ---
        if not self.systemInteractionHandler:
            logCritical("SystemInteractionHandler not initialized before ModelLifecycleManager.")
            raise RuntimeError(
                "SystemInteractionHandler not initialized before ModelLifecycleManager.")
        self.modelLifecycleManager = ModelLifecycleManager(
            self.config, self.stateManager, self.asrModelHandler, self.systemInteractionHandler
        )
        # --- Final Steps ---
        self._printInitialInstructions()
        logInfo("Orchestrator initialization complete.")

    def _initializeCoreComponents(self):
        """Initializes the core state and interaction components."""
        logDebug("Initializing core components...")
        self.stateManager = StateManager(self.config)
        self.systemInteractionHandler = SystemInteractionHandler(self.config)
        self.outputHandler = TranscriptionOutputHandler(self.config, self.stateManager,
                                                        self.systemInteractionHandler)
        self.audioHandler = AudioHandler(self.config, self.stateManager)
        self.realTimeProcessor = RealTimeAudioProcessor(self.config, self.stateManager)
        logDebug("Core components initialized.")

    def _initializeAsrHandler(self):
        """Determines and initializes the appropriate ASR handler (local or remote client)."""
        logDebug("Initializing ASR handler...")
        modelName = self.config.get('modelName', '')
        modelNameLower = modelName.lower()
        self.wslServerProcess = None
        self.wslLaunchCommand = []
        if modelNameLower.startswith("nvidia/"):
            logInfo(f"Configured Nvidia model: '{modelName}'. Preparing RemoteNemoClientHandler.")
            if platform.system() != "Windows":
                logError("Remote Nvidia model selected, but application is not running on Windows.")
                logError("Automatic WSL server launching is only supported from Windows.")
            self._prepareWslLaunchCommand(modelName)
            self.asrModelHandler = RemoteNemoClientHandler(self.config)
        elif modelName:
            logInfo(f"Configured non-Nvidia model: '{modelName}'. Using local WhisperModelHandler.")
            self.asrModelHandler = WhisperModelHandler(self.config)
            self.wslServerProcess = None
            self.wslLaunchCommand = []
        else:
            logCritical("No 'modelName' specified in configuration. Cannot initialize ASR handler.")
            self.asrModelHandler = None
        if self.asrModelHandler:
            logDebug(f"ASR Handler initialized: {type(self.asrModelHandler).__name__}")
        else:
            logError(
                "ASR Handler initialization failed.")

    def _prepareWslLaunchCommand(self, modelName):
        """Prepares the command list needed to launch the WSL NeMo server script."""
        logDebug("Preparing WSL server launch command...")
        self.wslLaunchCommand = []  # Reset command list
        wslServerUrl = self.config.get('wslServerUrl')
        wslDistro = self.config.get('wslDistributionName')
        useSudo = self.config.get('wslUseSudo', False)
        if not wslServerUrl or not wslDistro:
            logError(
                "Config error: 'wslServerUrl' & 'wslDistributionName' are required for automatic WSL server launch.")
            logError("WSL server will NOT be launched automatically.")
            return
        if platform.system() != "Windows":
            logWarning("WSL launch command preparation skipped: Not running on Windows.")
            return
        try:
            parsedUrl = urlparse(wslServerUrl)
            wslServerPort = parsedUrl.port
            if not wslServerPort:
                raise ValueError(f"Could not extract port from wslServerUrl: {wslServerUrl}")

            import __main__
            mainFilePath = None
            scriptDir = None
            if hasattr(__main__, '__file__') and __main__.__file__:
                mainFilePath = Path(os.path.abspath(__main__.__file__))
                scriptDir = mainFilePath.parent
            else:
                logWarning(
                    "Could not reliably determine main script path (__main__.__file__ missing). Falling back to mainManager.py directory.")
                scriptDir = Path(os.path.dirname(os.path.abspath(__file__)))

            wslServerScriptFilename = "wslNemoServer.py"
            wslServerScriptPathWindows = scriptDir / wslServerScriptFilename
            logDebug(f"Looking for WSL server script at: {wslServerScriptPathWindows}")
            if not wslServerScriptPathWindows.is_file():
                cwd = Path.cwd()
                fallbackPath = cwd / wslServerScriptFilename
                logDebug(
                    f"Script not found in script dir, checking fallback CWD path: {fallbackPath}")
                if not fallbackPath.is_file():
                    raise FileNotFoundError(
                        f"WSL script '{wslServerScriptFilename}' not found in script dir ({scriptDir}) or CWD ({cwd}).")
                wslServerScriptPathWindows = fallbackPath
                logWarning(f"Using WSL server script from CWD: {wslServerScriptPathWindows}")
            logDebug(f"Found WSL server script (Windows path): {wslServerScriptPathWindows}")

            wslServerScriptPathWsl = convertWindowsPathToWsl(wslServerScriptPathWindows)
            if not wslServerScriptPathWsl:
                raise ValueError(
                    f"Failed to convert Windows path to WSL path: {wslServerScriptPathWindows}")
            logDebug(f"Converted WSL server script path (WSL path): {wslServerScriptPathWsl}")

            pythonExecutable = "/usr/bin/python3"
            commandBase = ["wsl.exe", "-d", wslDistro, "--"]
            commandInsideWsl = []
            if useSudo:
                logWarning("Config 'wslUseSudo' is True. Preparing command with 'sudo'.")
                logWarning(
                    "--> CRITICAL: This requires passwordless sudo configured in WSL for the *exact* following command, otherwise launch WILL fail.")
                commandInsideWsl.append("sudo")
            commandInsideWsl.append(pythonExecutable)
            commandInsideWsl.append(wslServerScriptPathWsl)
            commandInsideWsl.extend([
                "--model_name", modelName,
                "--port", str(wslServerPort),
                "--load_on_start"
            ])
            preparedCommand = commandBase + commandInsideWsl
            self.wslLaunchCommand = preparedCommand
            logInfo("Prepared WSL server launch command successfully.")
            logDebug(f"WSL Command List (for Popen): {self.wslLaunchCommand}")
            try:
                cmdString = subprocess.list2cmdline(self.wslLaunchCommand)
                logInfo(
                    f"WSL Command String (for manual testing in CMD - verify quoting): {cmdString}")
                if useSudo:
                    logWarning(
                        "Verify the command string above matches your passwordless sudo configuration exactly.")
            except Exception as eCmdline:
                logWarning(f"Could not generate command string representation: {eCmdline}")
        except FileNotFoundError as e:
            logError(f"Error preparing WSL launch (FileNotFound): {e}")
            logError("WSL server will NOT be launched automatically.")
            self.wslLaunchCommand = []
        except ValueError as e:
            logError(f"Error preparing WSL launch (ValueError): {e}")
            logError("WSL server will NOT be launched automatically.")
            self.wslLaunchCommand = []
        except Exception as e:
            logError(f"Unexpected error preparing WSL launch command: {type(e).__name__} - {e}")
            logError(traceback.format_exc())
            logError("WSL server will NOT be launched automatically.")
            self.wslLaunchCommand = []

    def _printInitialInstructions(self):
        """Prints setup info and user instructions based on configuration."""
        if not self.asrModelHandler or not self.config or not self.systemInteractionHandler:
            logWarning("Cannot print initial instructions: Required components not initialized.")
            return

        modelName = self.config.get('modelName', 'N/A')
        mode = self.config.get('transcriptionMode', 'N/A')
        recKey = self.config.get('recordingToggleKey', 'N/A')
        outKey = self.config.get('outputToggleKey', 'N/A')
        forceKey = self.config.get('forceTranscriptionKey', 'N/A')  # New hotkey
        devId = self.config.get('deviceId', 'Default')
        rate = self.config.get('actualSampleRate', 'N/A')
        ch = self.config.get('actualChannels', 'N/A')
        maxRec = self.config.get('maxDurationRecording', 0)
        idleTime = self.config.get('consecutiveIdleTime', 0)
        unloadTimeout = self.config.get('model_unloadTimeout', 0)
        maxProgram = self.config.get('maxDurationProgramActive', 0)

        handlerType = type(self.asrModelHandler).__name__
        deviceStr = self.asrModelHandler.getDevice()

        maxRecStr = f"{maxRec} s" if maxRec > 0 else "Unlimited"
        idleTimeStr = f"{idleTime} s" if idleTime > 0 else "Disabled"
        unloadTimeoutStr = f"{unloadTimeout} s" if unloadTimeout > 0 else "Disabled"
        maxProgramStr = f"{maxProgram} s" if maxProgram > 0 else "Unlimited"

        logMessage = "\n--- Application Setup ---\n"
        logMessage += f"Mode:                 {mode}\n"
        logMessage += f"ASR Model:            {modelName}\n"
        logMessage += f"  Handler:            {handlerType}\n"
        logMessage += f"  Target Device:      {deviceStr}\n"
        if handlerType == 'RemoteNemoClientHandler':
            logMessage += f"  WSL Server URL:     {self.config.get('wslServerUrl', 'Not Set!')}\n"
            logMessage += f"  WSL Distro:         {self.config.get('wslDistributionName', 'Not Set!')}\n"
            logMessage += f"  WSL Use Sudo:       {self.config.get('wslUseSudo', False)}\n"
        logMessage += f"Audio Device:         ID={devId}, Rate={rate}Hz, Channels={ch}\n"
        logMessage += f"--- Hotkeys ---\n"
        logMessage += f"Toggle Recording:     '{recKey}'\n"
        logMessage += f"Toggle Text Output:   '{outKey}' (Method: {self.systemInteractionHandler.textOutputMethod})\n"
        logMessage += f"Force Transcription:  '{forceKey}'\n"  # New hotkey info
        logMessage += f"--- Timeouts ---\n"
        logMessage += f"Max Recording:        {maxRecStr}\n"
        logMessage += f"Stop Rec After Idle:  {idleTimeStr}\n"
        logMessage += f"Unload Model Inactive:{unloadTimeoutStr}\n"
        logMessage += f"Program Auto-Exit:    {maxProgramStr}\n"
        logMessage += f"-------------------------"
        logInfo(logMessage, indicatorName="APP_SETUP_INFO")

    def _launchWslServer(self) -> bool:
        """
        Launches the wslNemoServer.py script in WSL using subprocess if configured.
        Waits for the server process to become REACHABLE by polling its /status endpoint
        or until a timeout occurs. Logs errors.
        Returns:
            bool: True if the server process was started and became reachable
                  within the timeout, False otherwise.
        """
        if not self.wslLaunchCommand:
            logWarning("WSL server launch command not available. Skipping automatic launch.")
            return False
        if self.wslServerProcess and self.wslServerProcess.poll() is None:
            logInfo(
                f"WSL server process (PID: {self.wslServerProcess.pid}) appears to be already running. Checking reachability...")
            serverReadyTimeout = self.config.get('wslServerReadyTimeout', 90.0)
            isReachable = self._waitForServerReachable(serverReadyTimeout, checkProcessFirst=False)
            if isReachable:
                logInfo("Existing WSL server is reachable.")
                return True
            else:
                logWarning(
                    "Existing WSL server process did not become reachable within timeout or exited.")
                self._terminateWslServer()
        if platform.system() != "Windows":
            logError("WSL server launch skipped: Cannot execute wsl.exe on non-Windows platform.")
            return False
        logInfo(f"Attempting to launch WSL server...")
        logDebug(f"Executing Popen with command list: {self.wslLaunchCommand}")
        try:
            creationFlags = 0
            startupinfo = None
            if platform.system() == "Windows":
                creationFlags = subprocess.CREATE_NO_WINDOW
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            self.wslServerProcess = subprocess.Popen(
                self.wslLaunchCommand,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                creationflags=creationFlags,
                startupinfo=startupinfo,
                bufsize=1
            )
            logInfo(
                f"WSL server process launched (PID: {self.wslServerProcess.pid}). Waiting for server reachability...")
            serverReadyTimeout = self.config.get('wslServerReadyTimeout', 90.0)
            if serverReadyTimeout <= 0:
                logWarning(
                    "Config 'wslServerReadyTimeout' <= 0. Skipping server reachability polling.")
                logWarning("Waiting fixed 5 seconds (unreliable)...")
                time.sleep(5.0)
                processExitCode = self.wslServerProcess.poll()
                if processExitCode is None:
                    logWarning(
                        "WSL server process still running after fixed wait (reachability unknown). Assuming OK.")
                    return True
                else:
                    logError(
                        f"WSL server process exited quickly (code: {processExitCode}) during fixed wait (polling disabled).")
                    self._logWslProcessOutputOnError()
                    self.wslServerProcess = None
                    return False
            isReachable = self._waitForServerReachable(serverReadyTimeout, checkProcessFirst=True)
            if isReachable:
                logInfo(f"WSL server became reachable within {serverReadyTimeout}s timeout.")
                return True
            else:
                logError(
                    f"WSL server did not become reachable within {serverReadyTimeout}s timeout or exited.")
                if self.wslServerProcess and self.wslServerProcess.poll() is None:
                    logWarning(
                        "Terminating WSL server process as it failed reachability check but is still running.")
                    self._terminateWslServer()
                elif self.wslServerProcess:
                    logDebug(
                        "WSL server process already exited (confirmed after reachability check).")
                    self._logWslProcessOutputOnError()
                    self.wslServerProcess = None
                return False
        except FileNotFoundError:
            logError(
                f"Error launching WSL server: 'wsl.exe' not found. Is WSL installed and configured in system PATH?")
            self.wslServerProcess = None
            return False
        except PermissionError as pe:
            logError(f"Permission error launching WSL server process: {pe}")
            logError(
                "Hint: Check Windows permissions for running wsl.exe or accessing related resources.")
            self.wslServerProcess = None
            return False
        except Exception as e:
            logError(f"Unexpected error launching or monitoring WSL server process: {e}")
            logError(traceback.format_exc())
            if self.wslServerProcess and self.wslServerProcess.poll() is None:
                logWarning("Terminating WSL process due to unexpected launch error.")
                self._terminateWslServer()
            self.wslServerProcess = None
            return False

    def _waitForServerReachable(self, timeoutSeconds: float, checkProcessFirst: bool) -> bool:
        """
        Polls the WSL server's /status endpoint until it responds successfully (is reachable)
        or the timeout is reached, or the server process exits. Handles logging.
        Args:
            timeoutSeconds (float): Maximum time to wait for the server to be reachable.
            checkProcessFirst (bool): If True, checks if the process exited before polling network.
        Returns:
            bool: True if the server responded successfully to /status, False otherwise.
        """
        startTime = time.time()
        pollingInterval = 2.0
        if not isinstance(self.asrModelHandler, RemoteNemoClientHandler):
            logError("Cannot wait for server reachable: Incorrect ASR handler type.")
            return False
        if not self.wslServerProcess:
            logError(
                "Cannot wait for server reachable: WSL process handle is None (launch may have failed).")
            return False
        try:
            pid = self.wslServerProcess.pid
        except Exception:
            pid = "N/A (process exited)"
            logWarning("Could not get PID, WSL process may have exited very quickly.")
        logDebug(
            f"Waiting up to {timeoutSeconds:.1f}s for WSL server (PID: {pid}) to become reachable...")
        while True:
            elapsedTime = time.time() - startTime
            if elapsedTime >= timeoutSeconds:
                logWarning(
                    f"Timeout ({timeoutSeconds}s) waiting for WSL server reachability (PID: {pid}).")
                return False
            if checkProcessFirst:
                processExitCode = self.wslServerProcess.poll()
                if processExitCode is not None:
                    logError(
                        f"WSL server process (PID: {pid}) exited prematurely during reachability wait (exit code: {processExitCode}).")
                    self._logWslProcessOutputOnError()
                    self.wslServerProcess = None
                    return False
            logDebug(
                f"Polling WSL server /status for reachability (PID: {pid})... (Elapsed: {elapsedTime:.1f}s)")
            _ = self.asrModelHandler.checkServerStatus(forceCheck=True)
            if self.asrModelHandler.serverReachable is True:
                logDebug(f"Server /status check successful (server is reachable) (PID: {pid}).")
                return True
            elif self.asrModelHandler.serverReachable is False:
                logWarning(f"Server confirmed unreachable during reachability check (PID: {pid}).")
                processExitCode = self.wslServerProcess.poll()
                if processExitCode is not None:
                    logError(
                        f"WSL server process (PID: {pid}) confirmed exited (code: {processExitCode}) after becoming unreachable.")
                    self._logWslProcessOutputOnError()
                    self.wslServerProcess = None
                return False
            if not checkProcessFirst:
                processExitCode = self.wslServerProcess.poll()
                if processExitCode is not None:
                    logError(
                        f"WSL server process (PID: {pid}) exited after network poll (exit code: {processExitCode}).")
                    self._logWslProcessOutputOnError()
                    self.wslServerProcess = None
                    return False
            logDebug(f"Server not reachable yet (PID: {pid}), waiting {pollingInterval}s...")
            time.sleep(pollingInterval)

    def _logWslProcessOutputOnError(self):
        """Reads and logs stdout/stderr from the WSL process, typically after an error or exit."""
        if not self.wslServerProcess:
            logDebug("Skipping reading WSL output: process handle is None.")
            return
        try:
            pid = self.wslServerProcess.pid
        except Exception:
            pid = "N/A (process exited)"
        logInfo(f"Attempting to read stdout/stderr from failed/exited WSL process (PID: {pid})...")
        outputLogged = False
        try:
            stdoutData, _ = self.wslServerProcess.communicate(timeout=2.0)
            if stdoutData and stdoutData.strip():
                logError(
                    f"--- Captured WSL Server stdout/stderr (PID: {pid}) ---\n{stdoutData.strip()}",
                    indicatorName="WSL_SUBPROCESS_OUTPUT")
                if "sudo: a password is required" in stdoutData:
                    logError(
                        "!!! Detected 'sudo password required' error. Automatic launch failed.")
                    logError("!!! Configure passwordless sudo in WSL or set wslUseSudo=False.")
                if "Traceback (most recent call last)" in stdoutData:
                    logError("!!! Detected Python Traceback in WSL output. Check server script.")
                if "Address already in use" in stdoutData:
                    logError(
                        "!!! Detected 'Address already in use' in WSL output. Check port 5001 in WSL (`netstat -tulnp | grep 5001`).")
                outputLogged = True
            else:
                logDebug(f"WSL process {pid} communicate() returned no stdout/stderr data.")
        except subprocess.TimeoutExpired:
            logWarning(
                f"Timeout waiting for WSL process {pid} output via communicate(). Output may be incomplete.")
        except ValueError:
            logDebug(f"WSL process {pid} streams closed before/during communicate().")
        except Exception as readError:
            logWarning(
                f"Exception reading WSL process stdout/stderr for PID {pid} via communicate(): {readError}")

        if not outputLogged and self.wslServerProcess and self.wslServerProcess.stdout and not self.wslServerProcess.stdout.closed:
            logDebug(f"Attempting fallback non-blocking read for WSL process {pid}...")
            try:
                remainingOutput = self.wslServerProcess.stdout.read()
                if remainingOutput and remainingOutput.strip():
                    logError(
                        f"--- Fallback Read WSL Server stdout/stderr (PID: {pid}) ---\n{remainingOutput.strip()}",
                        indicatorName="WSL_SUBPROCESS_OUTPUT_FALLBACK")
                    outputLogged = True
            except Exception as fallbackReadError:
                logWarning(f"Fallback read for WSL process {pid} failed: {fallbackReadError}")
        if not outputLogged:
            logWarning(f"No stdout/stderr captured or logged from WSL process {pid}.")

    def _terminateWslServer(self):
        """Terminates the launched WSL server process if it exists and is running."""
        if not self.wslServerProcess:
            logDebug("No WSL server process handle found to terminate.")
            return
        try:
            pid = self.wslServerProcess.pid
        except Exception:
            pid = "N/A (process exited before term)"
        processExitCode = self.wslServerProcess.poll()
        if processExitCode is None:
            logInfo(f"Attempting to terminate running WSL server process (PID: {pid})...")
            try:
                self.wslServerProcess.terminate()
                logDebug(f"Sent terminate signal (SIGTERM) to WSL process {pid}.")
                try:
                    self.wslServerProcess.wait(timeout=3.0)
                    logInfo(
                        f"WSL server process (PID: {pid}) terminated gracefully (exit code: {self.wslServerProcess.returncode}).")
                    self._logWslProcessOutputOnError()
                except subprocess.TimeoutExpired:
                    logWarning(
                        f"WSL server process (PID: {pid}) did not terminate gracefully within timeout. Forcing kill (SIGKILL)...")
                    self.wslServerProcess.kill()
                    logDebug(f"Sent kill signal (SIGKILL) to WSL process {pid}.")
                    time.sleep(0.5)
                    finalExitCode = self.wslServerProcess.poll()
                    if finalExitCode is not None:
                        logInfo(
                            f"WSL server process (PID: {pid}) confirmed killed (exit code: {finalExitCode}).")
                    else:
                        logWarning(
                            f"WSL server process (PID: {pid}) did not exit immediately after kill signal.")
                    self._logWslProcessOutputOnError()
            except ProcessLookupError:
                logInfo(
                    f"WSL server process (PID: {pid}) already finished before termination signal could be sent.")
                self._logWslProcessOutputOnError()
            except Exception as e:
                logError(f"Error during termination of WSL server process (PID: {pid}): {e}",
                         exc_info=True)
                self._logWslProcessOutputOnError()
        else:
            logInfo(
                f"Launched WSL server process (PID: {pid}) was already finished (exit code {processExitCode}). Logging output.")
            self._logWslProcessOutputOnError()
        self.wslServerProcess = None
        logDebug("Cleared WSL server process handle.")

    def _transcriptionWorkerLoop(self):
        """
        Worker loop running in a separate thread.
        Waits for audio data on the queue, calls the ASR handler's transcribe method,
        and processes the result via the output handler.
        """
        if not self.asrModelHandler or not self.stateManager or not self.outputHandler:
            logError(
                "Transcription worker cannot start: Missing required components (ASR handler, StateManager, OutputHandler).")
            return
        handlerType = type(self.asrModelHandler).__name__
        logInfo(f"Starting Transcription Worker thread (Handler: {handlerType}).")
        queueTimeoutSeconds = 1.0
        while self.stateManager.shouldProgramContinue():
            try:
                queueItem = self.transcriptionRequestQueue.get(timeout=queueTimeoutSeconds)
                if queueItem is None:
                    logDebug("Transcription worker received None sentinel, stopping.")
                    break
                if not isinstance(queueItem, tuple) or len(queueItem) != 2:
                    logWarning(f"Invalid item in transcription queue: {type(queueItem)}.")
                    self.transcriptionRequestQueue.task_done()
                    continue
                audioDataToTranscribe, sampleRate = queueItem
                if audioDataToTranscribe is None or sampleRate <= 0:
                    logWarning(f"Skipping invalid transcription data (None or invalid rate).")
                    self.transcriptionRequestQueue.task_done()
                    continue
                if self.asrModelHandler.isModelLoaded():
                    segmentDuration = len(audioDataToTranscribe) / sampleRate
                    logDebug(
                        f"Transcription worker processing {segmentDuration:.2f}s audio segment...")
                    startTime = time.time()
                    transcriptionResult = self.asrModelHandler.transcribeAudioSegment(
                        audioDataToTranscribe, sampleRate)
                    inferenceTime = time.time() - startTime
                    logDebug(f"ASR inference took {inferenceTime:.3f}s.")
                    self.outputHandler.processTranscriptionResult(transcriptionResult,
                                                                  audioDataToTranscribe)
                    logDebug("Transcription worker finished processing segment.")
                else:
                    logWarning("Transcription worker skipped segment: ASR model not loaded.")
                    # Optionally, if dictation mode might get stuck due to this, reset relevant state
                    if self.config.get('transcriptionMode') == 'dictationMode':
                        logDebug(
                            "Considering dictation state reset if RealTimeAudioProcessor relies on output from this.")
                self.transcriptionRequestQueue.task_done()
            except queue.Empty:
                continue  # Expected when queue is empty
            except Exception as e:
                logError(f"!!! ERROR in Transcription Worker: {e}", exc_info=True)
                time.sleep(1)  # Brief pause before retrying loop
        logInfo("Transcription Worker thread stopping.")

    def _startBackgroundThreads(self):
        """Starts threads for hotkeys, model management, and transcription with error handling."""
        logDebug("Starting background threads...")
        self.threads = []
        failedThreads = []

        def threadWrapper(targetFunc, threadName, *args, **kwargs):
            logDebug(f"Thread '{threadName}' starting...")
            try:
                targetFunc(*args, **kwargs)
                logDebug(f"Thread '{threadName}' finished normally.")
            except Exception as e:
                logCritical(f"!!! EXCEPTION in thread '{threadName}': {e}", exc_info=True)
                if threadName in ["KeyboardMonitorThread", "TranscriptionWorkerThread"]:
                    logCritical(f"Critical thread '{threadName}' failed, stopping program.")
                    self.stateManager.stopProgram()
            finally:
                logDebug(f"Thread '{threadName}' has exited.")

        threadTargets = {
            "KeyboardMonitorThread": (
                self.systemInteractionHandler.monitorKeyboardShortcuts, (self,)),
            "ModelManagerThread": (self.modelLifecycleManager.manageModelLifecycle, ()),
            "TranscriptionWorkerThread": (self._transcriptionWorkerLoop, ()),
        }
        for threadName, (target, targetArgs) in threadTargets.items():
            if target is None:
                failedThreads.append(threadName)
                continue
            try:
                thread = threading.Thread(target=threadWrapper,
                                          args=(target, threadName) + targetArgs, name=threadName,
                                          daemon=True)
                self.threads.append(thread)
                thread.start()
                logDebug(f"Thread '{threadName}' initiated.")
            except Exception as e:
                logError(f"Failed start thread '{threadName}': {e}", exc_info=True)
                failedThreads.append(threadName)
        time.sleep(0.1)
        activeThreads = [t for t in self.threads if t.is_alive()]
        activeThreadCount = len(activeThreads)
        expectedThreadCount = len(threadTargets) - len(failedThreads)
        logDebug(f"Attempted {len(threadTargets)} threads. {activeThreadCount} active.")
        if failedThreads:
            logError(f"Failed threads: {', '.join(failedThreads)}")
        if activeThreadCount < expectedThreadCount:
            logWarning("One or more threads exited immediately.")
        if "KeyboardMonitorThread" not in [t.name for t in activeThreads]:
            logCritical("Keyboard monitor failed. Hotkeys disabled.")
        if "TranscriptionWorkerThread" not in [t.name for t in activeThreads]:
            logCritical("Transcription worker failed.")

    def toggleRecording(self):
        """Toggles the recording state. Called by systemInteractionHandler via hotkey."""
        if not all([self.stateManager, self.systemInteractionHandler, self.realTimeProcessor,
                    self.audioHandler]):
            logError("Cannot toggle recording: components missing.")
            return
        if self.stateManager.isRecording():
            if self.stateManager.stopRecording():
                self.systemInteractionHandler.playNotification("recordingOff")
                logInfo("Recording stopped.")
                self.realTimeProcessor.clearBuffer()
                self.audioHandler.clearQueue()
        else:
            if self.stateManager.startRecording():
                playEnable = self.config.get('playEnableSounds', False)
                self.systemInteractionHandler.playNotification("recordingOn",
                                                               forcePlay=not playEnable)
                logInfo("Recording started.")

    def toggleOutput(self):
        """Toggles the text output state. Called by systemInteractionHandler via hotkey."""
        if not all(
                [self.stateManager, self.systemInteractionHandler,
                 self.realTimeProcessor]):
            logError("Cannot toggle output: components missing.")
            return
        newState = self.stateManager.toggleOutput()
        playEnable = self.config.get('playEnableSounds', False)
        playSoundDisable = self.config.get('playSoundOnDisable', True)
        if newState and playEnable:
            self.systemInteractionHandler.playNotification("outputEnabled")
        elif not newState and playSoundDisable:
            self.systemInteractionHandler.playNotification("outputDisabled")
        if not newState:
            self.realTimeProcessor.clearBufferIfOutputDisabled()

    def forceTranscribeCurrentBuffer(self):
        """
        Forces the current audio buffer to be sent for transcription.
        Called by systemInteractionHandler via hotkey.
        """
        if not all([self.stateManager, self.realTimeProcessor, self.config,
                    self.transcriptionRequestQueue]):
            logError("Cannot force transcription: components missing.")
            return

        logInfo("Force transcription action triggered by hotkey.")

        if not self.stateManager.isOutputEnabled():
            logInfo("Force transcription: Output is currently disabled by user. Skipping action.")
            # Optionally play a 'disabled' sound or give feedback
            return

        # Get the audio buffer and clear it
        audioData = self.realTimeProcessor.getAudioBufferCopyAndClear()

        if audioData is not None and audioData.size > 0:
            sampleRate = self.config.get('actualSampleRate')
            if not sampleRate or sampleRate <= 0:
                logError("Invalid sampleRate in config. Cannot queue forced transcription.")
                return

            try:
                self.transcriptionRequestQueue.put((audioData, sampleRate), block=True, timeout=0.5)
                duration = len(audioData) / sampleRate
                logInfo(f"Forced transcription: Queued {duration:.2f}s of audio from buffer.")
                # Mark activity as this is a user-initiated ASR interaction
                self.stateManager.updateLastActivityTime()
            except queue.Full:
                logWarning(
                    "Forced transcription: Transcription request queue is full. Audio dropped.")
            except Exception as e:
                logError(f"Forced transcription: Error queuing audio data: {e}", exc_info=True)
        else:
            logInfo("Force transcription: Audio buffer was empty. Nothing to transcribe.")

    def _cleanup(self):
        """Cleans up all resources: stops threads, audio, model, WSL server."""
        logInfo("Initiating orchestrator cleanup...")
        if self.stateManager:
            self.stateManager.stopProgram()
            logDebug("Stop program signaled to StateManager.")
        try:
            self.transcriptionRequestQueue.put(None, block=False)  # Non-blocking put for sentinel
            logDebug("Sent sentinel to transcription queue.")
        except queue.Full:
            logWarning(
                "Transcription queue full, could not send sentinel during cleanup. Worker may take longer to stop.")
        except Exception as e:
            logWarning(f"Error putting sentinel to transcription queue during cleanup: {e}")

        if self.audioHandler:
            self.audioHandler.stopStream()
            logDebug("Audio stream stopped.")
        self._terminateWslServer()  # Handles its own logging
        if self.asrModelHandler:
            self.asrModelHandler.cleanup()  # Handles its own logging
        if self.realTimeProcessor:
            self.realTimeProcessor.clearBuffer()
            logDebug("RealTimeAudioProcessor buffer cleared.")
        if self.audioHandler:
            self.audioHandler.clearQueue()
            logDebug("AudioHandler queue cleared.")
        if self.systemInteractionHandler:
            self.systemInteractionHandler.cleanup()  # Handles its own logging

        logInfo("Joining background threads...")
        joinTimeout = 2.0
        threadsToJoin = list(self.threads)  # Make a copy
        self.threads = []  # Clear original list
        for t in threadsToJoin:
            threadName = t.name if hasattr(t, 'name') else "UnknownThread"
            if t is not None and t.is_alive():
                logDebug(f"Joining thread '{threadName}'...")
                t.join(timeout=joinTimeout)
                if t.is_alive():
                    logWarning(f"Thread '{threadName}' did not join within {joinTimeout}s timeout.")
                else:
                    logDebug(f"Thread '{threadName}' joined successfully.")
            else:
                logDebug(f"Thread '{threadName}' was not alive or None, skipping join.")
        logInfo("Cleanup complete.")

    def _runInitialSetup(self):
        """Handle initial setup: Launch WSL server if needed, check reachability, load model, start threads & audio stream."""
        logInfo("Running initial setup...")
        serverReachable = True
        if isinstance(self.asrModelHandler, RemoteNemoClientHandler):
            logInfo(
                "Remote NeMo handler detected, attempting WSL server launch and reachability check...")
            serverReachable = self._launchWslServer()
            if not serverReachable:
                logError(
                    "Automatic WSL NeMo server launch/reachability check failed. Remote transcription requires manual server start.")
            else:
                logInfo("WSL Server is reachable. Proceeding with model load check/trigger.")

        if serverReachable:
            logInfo("Attempting initial ASR model load/check...")
            loadSuccess = False
            try:
                loadSuccess = self.asrModelHandler.loadModel()
                if not loadSuccess:
                    logError("Initial ASR model load/check failed.")
                    if isinstance(self.asrModelHandler, WhisperModelHandler):
                        logCritical(
                            "Local Whisper model failed to load. Check model name/path and dependencies. Aborting application start.")
                        self.stateManager.stopProgram()
                        return
                    elif isinstance(self.asrModelHandler, RemoteNemoClientHandler):
                        logWarning(
                            "Failed to trigger initial load or confirm 'loaded' status on remote server (check server logs). Will rely on ModelLifecycleManager retries.")
                else:
                    logInfo("Initial ASR model load/check successful.")
            except Exception as e:
                logCritical(
                    f"Critical error during initial model load/check: {e}. Aborting application start.",
                    exc_info=True)
                logError(traceback.format_exc())
                self.stateManager.stopProgram()
                return
        else:
            logWarning("Skipping initial model load attempt as server was not reachable.")

        if self.stateManager.shouldProgramContinue():
            self._startBackgroundThreads()
        else:
            logWarning("Skipping background thread start due to earlier critical error.")

        initialStreamStarted = False
        if self.stateManager.shouldProgramContinue() and self.stateManager.isRecording():
            logInfo("Initial state is recording: attempting to start audio stream...")
            initialStreamStarted = self.audioHandler.startStream()
            if not initialStreamStarted:
                logCritical("Failed to start audio stream initially. Recording disabled.")
                self.stateManager.stopRecording()
        elif self.stateManager.shouldProgramContinue():
            logInfo("Initial state is not recording, audio stream will start when toggled on.")
        logInfo("Initial setup phase complete.")

    def _runCheckTimeoutsAndGlobalState(self):
        """Check program/recording timeouts, manage global state."""
        if not all([self.stateManager, self.realTimeProcessor]):
            return True  # Indicate continue if components missing, error logged elsewhere
        if self.stateManager.checkProgramTimeout():
            logInfo("Program timeout reached. Stopping program.")
            return False  # Indicate stop
        self.realTimeProcessor.clearBufferIfOutputDisabled()  # Clears buffer if output is off
        if self.stateManager.isRecording():
            if self.stateManager.checkRecordingTimeout():
                logInfo("Maximum recording duration reached. Stopping recording.")
                self.toggleRecording()
            elif self.stateManager.checkIdleTimeout():
                logInfo("Idle timeout reached. Stopping recording.")
                self.toggleRecording()
        return True  # Indicate continue

    def _runManageAudioStreamLifecycle(self):
        """Start/stop audio stream based on the desired recording state."""
        if not all([self.stateManager, self.audioHandler]):
            return True  # Continue, error logged if components missing
        shouldRecord = self.stateManager.isRecording()
        isCurrentlyRecording = self.audioHandler.stream is not None and self.audioHandler.stream.active
        try:
            if shouldRecord and not isCurrentlyRecording:
                logDebug("Desired state is Recording ON, but stream is OFF. Starting stream...")
                if not self.audioHandler.startStream():
                    logError("Failed to start audio stream when toggling ON. Disabling recording.")
                    self.stateManager.stopRecording()  # Ensure state reflects reality
            elif not shouldRecord and isCurrentlyRecording:
                logDebug("Desired state is Recording OFF, but stream is ON. Stopping stream...")
                self.audioHandler.stopStream()
                # Clearing buffers after stream stop is good practice
                if self.realTimeProcessor:
                    self.realTimeProcessor.clearBuffer()
                if self.audioHandler:  # audioHandler itself exists
                    self.audioHandler.clearQueue()
        except Exception as e:
            logError(f"Error managing audio stream lifecycle: {e}", exc_info=True)
            self.stateManager.stopRecording()  # Stop recording on error
        return True  # Continue

    def _runProcessAudioChunks(self):
        """Dequeue and process audio chunks from AudioHandler into RealTimeAudioProcessor."""
        if not all([self.stateManager, self.audioHandler, self.realTimeProcessor]):
            return False  # Indicate no chunks processed
        processedAnyChunks = False
        if self.stateManager.isRecording():
            # Check if audio stream is actually active
            isStreamActive = self.audioHandler.stream is not None and self.audioHandler.stream.active
            if not isStreamActive:
                # This can happen if stream failed to start or was stopped externally
                # logDebug("Audio stream not active, cannot process chunks.") # Can be noisy
                return False

            # Process a limited number of chunks per loop iteration to keep main loop responsive
            maxChunksPerIteration = 50  # Configurable if needed
            chunksProcessedThisIteration = 0
            while chunksProcessedThisIteration < maxChunksPerIteration:
                audioChunk = self.audioHandler.getAudioChunk()
                if audioChunk is None:  # No more chunks in queue currently
                    break
                if self.realTimeProcessor.processIncomingChunk(audioChunk):
                    processedAnyChunks = True
                chunksProcessedThisIteration += 1
            # if chunksProcessedThisIteration > 0 :
            #     logDebug(f"Processed {chunksProcessedThisIteration} audio chunks this iteration.")
        return processedAnyChunks

    def _runQueueTranscriptionRequest(self, audioWasProcessedThisLoop):
        """
        Checks if transcription should be triggered based on mode and state,
        then queues the audio data if conditions are met.
        """
        if not all([self.stateManager, self.realTimeProcessor, self.config,
                    self.transcriptionRequestQueue]):
            return  # Cannot proceed if components are missing

        # Only check for transcription trigger if output is enabled.
        # Audio is still buffered by RealTimeAudioProcessor even if output is off.
        # The new forceTranscribeCurrentBuffer bypasses this direct check for outputEnabled,
        # but it has its own check.
        if not self.stateManager.isOutputEnabled():
            # logDebug("Skipping transcription trigger check: Output is disabled.") # Can be noisy
            return

        # Determine if a trigger check is warranted.
        # - For dictationMode, check even if no new audio was processed this loop,
        #   as silence timing might trigger it.
        # - For constantIntervalMode, only check if new audio was processed OR if the interval is met.
        #   (checkTranscriptionTrigger handles interval logic internally).
        # The `checkTranscriptionTrigger` method itself is responsible for mode-specific logic.
        shouldCheckTrigger = True  # Default to checking
        # if self.config.get('transcriptionMode') == 'constantIntervalMode' and not audioWasProcessedThisLoop:
        #    # For constant interval, if no new audio, only the timer matters.
        #    # Let checkTranscriptionTrigger handle the timer.
        #    pass # No specific condition to skip here for now

        if shouldCheckTrigger:
            audioDataToTranscribe = self.realTimeProcessor.checkTranscriptionTrigger()
            if audioDataToTranscribe is not None and audioDataToTranscribe.size > 0:
                sampleRate = self.config.get('actualSampleRate')
                if not sampleRate or sampleRate <= 0:
                    logError("Invalid sampleRate in config. Cannot queue transcription.")
                    return
                try:
                    self.transcriptionRequestQueue.put((audioDataToTranscribe, sampleRate),
                                                       block=True, timeout=0.5)
                    duration = len(audioDataToTranscribe) / sampleRate
                    logDebug(
                        f"Queued {duration:.2f}s of audio for transcription (regular trigger).")
                    # Update activity time since we are sending data for ASR
                    self.stateManager.updateLastActivityTime()
                except queue.Full:
                    logWarning("Transcription request queue is full. Audio segment dropped.")
                except Exception as e:
                    logError(f"Error queuing audio data for transcription: {e}", exc_info=True)
            # else:
            # logDebug("No transcription trigger met or buffer was empty.") # Can be noisy

    def _runLoopSleep(self):
        """Sleep briefly to yield CPU and control loop speed."""
        # A short sleep is crucial to prevent the main loop from consuming 100% CPU.
        # 0.01 seconds (10ms) is a common starting point.
        time.sleep(0.01)

    def run(self):
        """Main execution loop."""
        logInfo("Starting main orchestrator loop...")
        initialSetupOk = False
        try:
            self._runInitialSetup()  # Handles WSL, model load, threads, initial audio stream
            if not self.stateManager.shouldProgramContinue():
                logError(
                    "Initial setup indicated program should not continue (e.g., critical model load failure).")
            else:
                initialSetupOk = True

            if initialSetupOk:
                logInfo("Entering main processing loop...")
                while self.stateManager.shouldProgramContinue():
                    # 1. Check timeouts and global state
                    if not self._runCheckTimeoutsAndGlobalState():
                        break  # Program timeout or other stop signal

                    # 2. Manage audio stream lifecycle (start/stop based on state)
                    self._runManageAudioStreamLifecycle()

                    # 3. Process incoming audio chunks if recording
                    audioProcessedThisLoop = self._runProcessAudioChunks()

                    # 4. Check for transcription trigger and queue data
                    # Pass audioProcessedThisLoop to potentially optimize constantIntervalMode checks
                    self._runQueueTranscriptionRequest(audioProcessedThisLoop)

                    # 5. Brief sleep
                    self._runLoopSleep()
        except KeyboardInterrupt:
            logInfo("\nKeyboardInterrupt received. Stopping application...")
        except Exception as e:
            logCritical(f"\n!!! CRITICAL UNHANDLED ERROR IN MAIN LOOP: {e}", exc_info=True)
            logError(
                traceback.format_exc())  # Ensure full traceback is logged for unexpected errors
        finally:
            if self.stateManager:  # Ensure stateManager exists before trying to use it
                self.stateManager.stopProgram()  # Signal all loops and threads to stop
            logInfo("Exiting main loop." if initialSetupOk else "Exiting after setup failure.")
            self._cleanup()  # Perform cleanup of all resources
            logInfo("Orchestrator run finished.")
