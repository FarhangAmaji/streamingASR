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
import logging
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
# Import logging helpers from utils
from utils import logWarning, logDebug, convertWindowsPathToWsl, logInfo, logError


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
            raise RuntimeError(
                "ASR Handler could not be initialized. Check configuration and logs. Cannot continue.")
        # --- Instantiate Model Lifecycle Manager (depends on ASR Handler) ---
        if not self.systemInteractionHandler:
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
            logError(
                "CRITICAL: No 'modelName' specified in configuration. Cannot initialize ASR handler.")
            self.asrModelHandler = None
        if self.asrModelHandler:
            logDebug(f"ASR Handler initialized: {type(self.asrModelHandler).__name__}")
        else:
            logError("ASR Handler initialization failed.")

    def _prepareWslLaunchCommand(self, modelName):
        """Prepares the command list needed to launch the WSL NeMo server script."""
        logDebug("Preparing WSL server launch command...")
        self.wslLaunchCommand = []  # Reset command list
        wslServerUrl = self.config.get('wslServerUrl')
        wslDistro = self.config.get('wslDistributionName')
        useSudo = self.config.get('wslUseSudo', False)  # Get setting from config
        # Check prerequisites
        if not wslServerUrl or not wslDistro:
            logError(
                "Config error: 'wslServerUrl' & 'wslDistributionName' are required for automatic WSL server launch.")
            logError("WSL server will NOT be launched automatically.")
            return
        if platform.system() != "Windows":
            logWarning("WSL launch command preparation skipped: Not running on Windows.")
            return
        try:
            # --- Get Port ---
            parsedUrl = urlparse(wslServerUrl)
            wslServerPort = parsedUrl.port
            if not wslServerPort:
                raise ValueError(f"Could not extract port from wslServerUrl: {wslServerUrl}")
            # --- Find Script Path ---
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
            # --- Convert Path ---
            wslServerScriptPathWsl = convertWindowsPathToWsl(wslServerScriptPathWindows)
            if not wslServerScriptPathWsl:
                raise ValueError(
                    f"Failed to convert Windows path to WSL path: {wslServerScriptPathWindows}")
            logDebug(f"Converted WSL server script path (WSL path): {wslServerScriptPathWsl}")
            # --- Construct Command List ---
            pythonExecutable = "/usr/bin/python3"  # Use full path for robustness
            # Base command to execute commands within the specified WSL distro
            commandBase = [
                "wsl.exe",
                "-d", wslDistro,
                "--"  # Separates wsl.exe options from the command to run inside WSL
            ]
            # Command sequence to run inside WSL
            commandInsideWsl = []
            # Optionally prepend sudo
            if useSudo:
                logWarning("Config 'wslUseSudo' is True. Preparing command with 'sudo'.")
                logWarning(
                    "--> CRITICAL: This requires passwordless sudo configured in WSL for the *exact* following command, otherwise launch WILL fail.")
                commandInsideWsl.append("sudo")
            # Add Python executable and script path
            commandInsideWsl.append(pythonExecutable)
            # The script path is a single argument, even if it contains spaces (list item handles this)
            commandInsideWsl.append(wslServerScriptPathWsl)
            # Add script arguments
            commandInsideWsl.extend([
                "--model_name", modelName,
                "--port", str(wslServerPort),
                "--load_on_start"  # Let the server handle this with background loading
            ])
            # Combine wsl.exe command with the command to run inside WSL
            preparedCommand = commandBase + commandInsideWsl
            # --- Assign and Log ---
            self.wslLaunchCommand = preparedCommand
            logInfo("Prepared WSL server launch command successfully.")
            # Log the command list clearly for debugging
            logDebug(f"WSL Command List (for Popen): {self.wslLaunchCommand}")
            # Log the command as a string for easier manual testing in CMD/PowerShell
            try:
                # Use list2cmdline for basic quoting, but manual verification is best
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
        # Safely get config values with defaults
        modelName = self.config.get('modelName', 'N/A')
        mode = self.config.get('transcriptionMode', 'N/A')
        recKey = self.config.get('recordingToggleKey', 'N/A')
        outKey = self.config.get('outputToggleKey', 'N/A')
        devId = self.config.get('deviceId', 'Default')
        rate = self.config.get('actualSampleRate', 'N/A')
        ch = self.config.get('actualChannels', 'N/A')
        maxRec = self.config.get('maxDurationRecording', 0)
        idleTime = self.config.get('consecutiveIdleTime', 0)
        unloadTimeout = self.config.get('model_unloadTimeout', 0)
        maxProgram = self.config.get('maxDurationProgramActive', 0)
        # Get handler specific info
        handlerType = type(self.asrModelHandler).__name__
        deviceStr = self.asrModelHandler.getDevice()  # Should return 'remote_wsl' or local device
        # Format timeouts nicely
        maxRecStr = f"{maxRec} s" if maxRec > 0 else "Unlimited"
        idleTimeStr = f"{idleTime} s" if idleTime > 0 else "Disabled"
        unloadTimeoutStr = f"{unloadTimeout} s" if unloadTimeout > 0 else "Disabled"
        maxProgramStr = f"{maxProgram} s" if maxProgram > 0 else "Unlimited"
        # Build log message string
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
        logMessage += f"--- Timeouts ---\n"
        logMessage += f"Max Recording:        {maxRecStr}\n"
        logMessage += f"Stop Rec After Idle:  {idleTimeStr}\n"
        logMessage += f"Unload Model Inactive:{unloadTimeoutStr}\n"
        logMessage += f"Program Auto-Exit:    {maxProgramStr}\n"
        logMessage += f"-------------------------"
        # Log the entire block as one INFO message
        logInfo(logMessage)

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
            # Command preparation failed or skipped, error already logged.
            logWarning("WSL server launch command not available. Skipping automatic launch.")
            return False
        # Check if we already have a process handle and if it's still running
        if self.wslServerProcess and self.wslServerProcess.poll() is None:
            # Process exists from a previous attempt/run
            logInfo(
                f"WSL server process (PID: {self.wslServerProcess.pid}) appears to be already running. Checking reachability...")
            serverReadyTimeout = self.config.get('wslServerReadyTimeout', 90.0)
            # Check network reachability first for existing process
            isReachable = self._waitForServerReachable(serverReadyTimeout,
                                                       checkProcessFirst=False)  # Use reachability check
            if isReachable:
                logInfo("Existing WSL server is reachable.")
                return True
            else:
                # Server process exists but isn't reachable
                logWarning(
                    "Existing WSL server process did not become reachable within timeout or exited.")
                # Terminate the potentially defunct existing process before trying to launch a new one
                self._terminateWslServer()
                # Proceed to launch a new instance below
        # Ensure we are on Windows before trying to execute wsl.exe
        if platform.system() != "Windows":
            logError("WSL server launch skipped: Cannot execute wsl.exe on non-Windows platform.")
            return False
        logInfo(f"Attempting to launch WSL server...")
        # Log the exact command list being passed to Popen again for clarity during launch
        logDebug(f"Executing Popen with command list: {self.wslLaunchCommand}")
        try:
            # Options to make the subprocess less intrusive on Windows
            creationFlags = 0
            startupinfo = None
            if platform.system() == "Windows":
                creationFlags = subprocess.CREATE_NO_WINDOW
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            # Launch the WSL command using Popen
            # CRITICAL: Redirect stderr to stdout to capture errors (including sudo prompts or Python tracebacks)
            # Use text mode and UTF-8 encoding for reliable output reading
            # bufsize=1 enables line buffering, might help see startup errors faster
            self.wslServerProcess = subprocess.Popen(
                self.wslLaunchCommand,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # <<< Redirect stderr to stdout
                text=True,
                encoding='utf-8',
                errors='replace',  # Handle potential encoding errors in output
                creationflags=creationFlags,
                startupinfo=startupinfo,
                bufsize=1  # Line buffering
            )
            logInfo(
                f"WSL server process launched (PID: {self.wslServerProcess.pid}). Waiting for server reachability...")
            # Wait for the server to become *reachable* (Flask running)
            serverReadyTimeout = self.config.get('wslServerReadyTimeout', 90.0)
            if serverReadyTimeout <= 0:
                # Polling disabled - highly unreliable, use only for debugging specific issues
                logWarning(
                    "Config 'wslServerReadyTimeout' <= 0. Skipping server reachability polling.")
                logWarning("Waiting fixed 5 seconds (unreliable)...")
                time.sleep(5.0)
                # Check if process exited quickly even without polling
                processExitCode = self.wslServerProcess.poll()
                if processExitCode is None:
                    logWarning(
                        "WSL server process still running after fixed wait (reachability unknown). Assuming OK.")
                    return True  # Assume reachable (risky)
                else:
                    logError(
                        f"WSL server process exited quickly (code: {processExitCode}) during fixed wait (polling disabled).")
                    # Log output from the failed process
                    self._logWslProcessOutputOnError()
                    self.wslServerProcess = None  # Clear handle
                    return False  # Failed
            # Wait for reachability, checking process status before network poll
            isReachable = self._waitForServerReachable(serverReadyTimeout,
                                                       checkProcessFirst=True)  # Use reachability check
            if isReachable:
                logInfo(f"WSL server became reachable within {serverReadyTimeout}s timeout.")
                # Model is NOT necessarily loaded yet, just the server is running Flask
                return True  # Success!
            else:
                # Failure case: Reachability check timed out or process exited prematurely
                logError(
                    f"WSL server did not become reachable within {serverReadyTimeout}s timeout or exited.")
                # _waitForServerReachable or _logWslProcessOutputOnError should have logged details.
                # Ensure process is terminated if it's somehow still running but not reachable
                if self.wslServerProcess and self.wslServerProcess.poll() is None:
                    logWarning(
                        "Terminating WSL server process as it failed reachability check but is still running.")
                    self._terminateWslServer()
                elif self.wslServerProcess:  # Process handle exists but process already exited
                    logDebug(
                        "WSL server process already exited (confirmed after reachability check).")
                    # Logging should have happened in _waitForServerReachable, but call again just in case
                    self._logWslProcessOutputOnError()
                    self.wslServerProcess = None  # Clear handle
                # If self.wslServerProcess is already None, _waitForServerReachable handled it.
                return False  # Indicate launch/reachability failure
        except FileNotFoundError:
            # This error usually means 'wsl.exe' wasn't found in the system PATH
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
            # Catch any other unexpected errors during Popen or initial setup
            logError(f"Unexpected error launching or monitoring WSL server process: {e}")
            logError(traceback.format_exc())
            # If process was created but an error occurred after, try to terminate it
            if self.wslServerProcess and self.wslServerProcess.poll() is None:
                logWarning("Terminating WSL process due to unexpected launch error.")
                self._terminateWslServer()
            self.wslServerProcess = None  # Ensure handle is cleared
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
        pollingInterval = 2.0  # How often to poll
        if not isinstance(self.asrModelHandler, RemoteNemoClientHandler):
            logError("Cannot wait for server reachable: Incorrect ASR handler type.")
            return False
        if not self.wslServerProcess:
            # This check is important as the process might fail to launch in _launchWslServer
            logError(
                "Cannot wait for server reachable: WSL process handle is None (launch may have failed).")
            return False
        # Get PID early for consistent logging, handle potential immediate exit race condition
        try:
            pid = self.wslServerProcess.pid
        except Exception:
            pid = "N/A (process exited)"
            logWarning("Could not get PID, WSL process may have exited very quickly.")
        logDebug(
            f"Waiting up to {timeoutSeconds:.1f}s for WSL server (PID: {pid}) to become reachable...")
        while True:
            elapsedTime = time.time() - startTime
            # Check 1: Timeout
            if elapsedTime >= timeoutSeconds:
                logWarning(
                    f"Timeout ({timeoutSeconds}s) waiting for WSL server reachability (PID: {pid}).")
                return False  # Timed out
            # Check 2: Process Exit (Conditional based on checkProcessFirst)
            if checkProcessFirst:
                processExitCode = self.wslServerProcess.poll()
                if processExitCode is not None:
                    logError(
                        f"WSL server process (PID: {pid}) exited prematurely during reachability wait (exit code: {processExitCode}).")
                    # Log output from the failed process
                    self._logWslProcessOutputOnError()
                    self.wslServerProcess = None  # Clear handle since it exited
                    return False  # Server definitely not reachable if process died
            # Check 3: Network Status Poll via Handler
            logDebug(
                f"Polling WSL server /status for reachability (PID: {pid})... (Elapsed: {elapsedTime:.1f}s)")
            # Use the handler's status check method, forcing a network check.
            # We only care if the request succeeded (server responded at all).
            # The method internally updates self.asrModelHandler.serverReachable.
            # Ignore the boolean return value (isLoaded) for this reachability check.
            _ = self.asrModelHandler.checkServerStatus(forceCheck=True)
            # Check the internal state updated by checkServerStatus
            if self.asrModelHandler.serverReachable is True:
                # If the request succeeded (no connection error, no timeout, no HTTP error)
                # then the Flask server is up and running.
                logDebug(f"Server /status check successful (server is reachable) (PID: {pid}).")
                return True  # Server is reachable!
            elif self.asrModelHandler.serverReachable is False:
                # checkServerStatus failed with a connection/network error.
                logWarning(f"Server confirmed unreachable during reachability check (PID: {pid}).")
                # Check if the process died *after* the failed network attempt
                processExitCode = self.wslServerProcess.poll()
                if processExitCode is not None:
                    logError(
                        f"WSL server process (PID: {pid}) confirmed exited (code: {processExitCode}) after becoming unreachable.")
                    self._logWslProcessOutputOnError()
                    self.wslServerProcess = None
                # Return False as it's not reachable
                return False
            # If checkServerStatus failed but didn't set serverReachable to False
            # (e.g., Timeout, HTTP 5xx error from server), continue polling.
            # Check 4: Process Exit (After Network Poll or if checkProcessFirst was False)
            # Catch cases where process exits between polls or after network attempt fails without ConnectionError
            if not checkProcessFirst:
                processExitCode = self.wslServerProcess.poll()
                if processExitCode is not None:
                    logError(
                        f"WSL server process (PID: {pid}) exited after network poll (exit code: {processExitCode}).")
                    self._logWslProcessOutputOnError()
                    self.wslServerProcess = None
                    return False  # Server definitely not reachable
            # --- Wait before the next poll ---
            logDebug(f"Server not reachable yet (PID: {pid}), waiting {pollingInterval}s...")
            time.sleep(pollingInterval)

    def _logWslProcessOutputOnError(self):
        """Reads and logs stdout/stderr from the WSL process, typically after an error or exit."""
        if not self.wslServerProcess:
            logDebug("Skipping reading WSL output: process handle is None.")
            return
        # Store PID for logging, handle potential early exit
        try:
            pid = self.wslServerProcess.pid
        except Exception:
            pid = "N/A (process exited)"
        logInfo(f"Attempting to read stdout/stderr from failed/exited WSL process (PID: {pid})...")
        outputLogged = False
        try:
            # communicate() reads *all* remaining buffered output until EOF and waits (with timeout).
            # This is generally the most reliable way to get output after a process
            # has exited or failed, as direct reads on pipes might miss data or block.
            stdoutData, _ = self.wslServerProcess.communicate(
                timeout=2.0)  # Use a slightly longer timeout
            if stdoutData and stdoutData.strip():
                logError(f"--- Captured WSL Server stdout/stderr (PID: {pid}) ---")
                # Log line by line for better readability in logs
                lines = stdoutData.strip().splitlines()
                for i, line in enumerate(lines):
                    # Limit number of lines logged directly if output is huge?
                    # if i < 100:
                    logError(f"WSL_OUT: {line}")
                    # else:
                    #    if i == 100: logError("WSL_OUT: ... (output truncated)")
                    #    break
                logError(f"--- End WSL Server output (PID: {pid}) ---")
                # Check for common error indicators
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
            # Process might still be running if timeout occurred, termination handled elsewhere.
        except ValueError:  # Streams might be closed already (e.g., by communicate call)
            logDebug(f"WSL process {pid} streams closed before/during communicate().")
        except Exception as readError:
            logWarning(
                f"Exception reading WSL process stdout/stderr for PID {pid} via communicate(): {readError}")
        # Fallback: If communicate failed/timed out, maybe try a non-blocking read? (Less reliable)
        if not outputLogged and self.wslServerProcess and self.wslServerProcess.stdout and not self.wslServerProcess.stdout.closed:
            logDebug(f"Attempting fallback non-blocking read for WSL process {pid}...")
            try:
                remainingOutput = self.wslServerProcess.stdout.read()
                if remainingOutput and remainingOutput.strip():
                    logError(f"--- Fallback Read WSL Server stdout/stderr (PID: {pid}) ---")
                    logError(remainingOutput.strip())
                    logError(f"--- End Fallback Read WSL Server output (PID: {pid}) ---")
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
        # Store PID for logging, handle potential early exit
        try:
            pid = self.wslServerProcess.pid
        except Exception:
            pid = "N/A (process exited before term)"
        processExitCode = self.wslServerProcess.poll()  # Check status first
        if processExitCode is None:  # Process is still running
            logInfo(f"Attempting to terminate running WSL server process (PID: {pid})...")
            try:
                # 1. Attempt graceful termination using SIGTERM
                self.wslServerProcess.terminate()
                logDebug(f"Sent terminate signal (SIGTERM) to WSL process {pid}.")
                try:
                    # Wait a short time for graceful exit
                    self.wslServerProcess.wait(timeout=3.0)
                    logInfo(
                        f"WSL server process (PID: {pid}) terminated gracefully (exit code: {self.wslServerProcess.returncode}).")
                    # Log any final output AFTER graceful termination attempt
                    self._logWslProcessOutputOnError()
                except subprocess.TimeoutExpired:
                    # 2. Force kill using SIGKILL if terminate didn't work
                    logWarning(
                        f"WSL server process (PID: {pid}) did not terminate gracefully within timeout. Forcing kill (SIGKILL)...")
                    self.wslServerProcess.kill()
                    logDebug(f"Sent kill signal (SIGKILL) to WSL process {pid}.")
                    # Short wait to allow kill signal processing
                    time.sleep(0.5)
                    finalExitCode = self.wslServerProcess.poll()
                    if finalExitCode is not None:
                        logInfo(
                            f"WSL server process (PID: {pid}) confirmed killed (exit code: {finalExitCode}).")
                    else:
                        logWarning(
                            f"WSL server process (PID: {pid}) did not exit immediately after kill signal.")
                    # Log output AFTER kill attempt (even if kill seemed ineffective)
                    self._logWslProcessOutputOnError()
            except ProcessLookupError:  # Process finished between poll() and terminate()/kill()
                logInfo(
                    f"WSL server process (PID: {pid}) already finished before termination signal could be sent.")
                # Log output if process handle is still valid
                self._logWslProcessOutputOnError()
            except Exception as e:
                logError(f"Error during termination of WSL server process (PID: {pid}): {e}")
                # Attempt to log output even on termination error
                self._logWslProcessOutputOnError()
        else:  # Process already finished before terminate was called
            logInfo(
                f"Launched WSL server process (PID: {pid}) was already finished (exit code {processExitCode}). Logging output.")
            self._logWslProcessOutputOnError()  # Log output
        # Always clear the handle after attempting termination/logging
        self.wslServerProcess = None
        logDebug("Cleared WSL server process handle.")

    # --- Background Thread Worker for Transcription ---
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
                if queueItem is None: logDebug("Worker received None sentinel."); break
                if not isinstance(queueItem, tuple) or len(queueItem) != 2:
                    logWarning(f"Invalid item in queue: {type(queueItem)}.");
                    self.transcriptionRequestQueue.task_done();
                    continue
                audioDataToTranscribe, sampleRate = queueItem
                if audioDataToTranscribe is None or sampleRate <= 0:
                    logWarning(f"Skipping invalid transcription data.");
                    self.transcriptionRequestQueue.task_done();
                    continue
                if self.asrModelHandler.isModelLoaded():
                    segmentDuration = len(audioDataToTranscribe) / sampleRate;
                    logDebug(f"Worker processing {segmentDuration:.2f}s...")
                    startTime = time.time();
                    transcriptionResult = self.asrModelHandler.transcribeAudioSegment(
                        audioDataToTranscribe, sampleRate);
                    inferenceTime = time.time() - startTime
                    logDebug(f"ASR inference took {inferenceTime:.3f}s.")
                    self.outputHandler.processTranscriptionResult(transcriptionResult,
                                                                  audioDataToTranscribe)
                    logDebug("Worker finished segment.")
                else:
                    logWarning("Worker skipped segment: ASR model not loaded.")
                    if self.config.get('transcriptionMode') == 'dictationMode': logDebug(
                        "Dictation state might need reset.")
                self.transcriptionRequestQueue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logError(f"!!! ERROR in Transcription Worker: {e}", exc_info=True);
                time.sleep(1)
        logInfo("Transcription Worker thread stopping.")

    def _startBackgroundThreads(self):
        """Starts threads for hotkeys, model management, and transcription with error handling."""
        logDebug("Starting background threads...")
        self.threads = [];
        failedThreads = []

        def threadWrapper(targetFunc, threadName, *args, **kwargs):
            logDebug(f"Thread '{threadName}' starting...")
            try:
                targetFunc(*args, **kwargs);
                logDebug(f"Thread '{threadName}' finished normally.")
            except Exception as e:
                logError(f"!!! EXCEPTION in thread '{threadName}': {e}", exc_info=True)
                if threadName in ["KeyboardMonitorThread", "TranscriptionWorkerThread"]:
                    logCritical = getattr(logging, 'critical', logError)
                    logCritical(f"Critical thread '{threadName}' failed, stopping program.");
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
            if target is None: failedThreads.append(threadName); continue
            try:
                thread = threading.Thread(target=threadWrapper,
                                          args=(target, threadName) + targetArgs, name=threadName,
                                          daemon=True);
                self.threads.append(
                    thread);
                thread.start();
                logDebug(f"Thread '{threadName}' initiated.")
            except Exception as e:
                logError(f"Failed start thread '{threadName}': {e}");
                failedThreads.append(
                    threadName)
        time.sleep(0.1);
        activeThreads = [t for t in self.threads if t.is_alive()];
        activeThreadCount = len(activeThreads);
        expectedThreadCount = len(threadTargets) - len(failedThreads)
        logDebug(f"Attempted {len(threadTargets)} threads. {activeThreadCount} active.")
        if failedThreads: logError(f"Failed threads: {', '.join(failedThreads)}")
        if activeThreadCount < expectedThreadCount: logWarning(
            "One or more threads exited immediately.")
        if "KeyboardMonitorThread" not in [t.name for t in activeThreads]: logError(
            "Keyboard monitor failed. Hotkeys disabled.")
        if "TranscriptionWorkerThread" not in [t.name for t in activeThreads]: logError(
            "Transcription worker failed.")

    # --- Public Methods for Hotkey Actions ---
    def toggleRecording(self):
        """Toggles the recording state. Called by systemInteractionHandler via hotkey."""
        if not all([self.stateManager, self.systemInteractionHandler, self.realTimeProcessor,
                    self.audioHandler]): logError(
            "Cannot toggle recording: components missing."); return
        if self.stateManager.isRecording():
            if self.stateManager.stopRecording(): self.systemInteractionHandler.playNotification(
                "recordingOff"); logInfo(
                "Recording stopped."); self.realTimeProcessor.clearBuffer(); self.audioHandler.clearQueue()
        else:
            if self.stateManager.startRecording():
                playEnable = self.config.get('playEnableSounds', False)
                self.systemInteractionHandler.playNotification("recordingOn",
                                                               forcePlay=not playEnable)  # Play if enable sounds off
                logInfo("Recording started.")

    def toggleOutput(self):
        """Toggles the text output state. Called by systemInteractionHandler via hotkey."""
        if not all(
                [self.stateManager, self.systemInteractionHandler,
                 self.realTimeProcessor]): logError(
            "Cannot toggle output: components missing."); return
        newState = self.stateManager.toggleOutput()
        playEnable = self.config.get('playEnableSounds', False)
        playSoundDisable = self.config.get('playSoundOnDisable', True)
        if newState and playEnable:
            self.systemInteractionHandler.playNotification("outputEnabled")
        elif not newState and playSoundDisable:
            self.systemInteractionHandler.playNotification("outputDisabled")
        if not newState: self.realTimeProcessor.clearBufferIfOutputDisabled()

    # ---- Cleanup ---
    def _cleanup(self):
        """Cleans up all resources: stops threads, audio, model, WSL server."""
        logInfo("Initiating orchestrator cleanup...")
        if self.stateManager: self.stateManager.stopProgram(); logDebug("Stop signaled.")
        try:
            self.transcriptionRequestQueue.put(None, block=True, timeout=0.5);
            logDebug(
                "Sent sentinel.")
        except Exception as e:
            logWarning(f"Error putting sentinel: {e}")
        if self.audioHandler: self.audioHandler.stopStream()
        self._terminateWslServer()
        if self.asrModelHandler: self.asrModelHandler.cleanup()
        if self.realTimeProcessor: self.realTimeProcessor.clearBuffer()
        if self.audioHandler: self.audioHandler.clearQueue()
        if self.systemInteractionHandler: self.systemInteractionHandler.cleanup()
        logInfo("Joining background threads...")
        joinTimeout = 2.0;
        threadsToJoin = list(self.threads);
        self.threads = []
        for t in threadsToJoin:
            threadName = t.name if hasattr(t, 'name') else "Unknown"
            if t is not None and t.is_alive():
                logDebug(f"Joining '{threadName}'...");
                t.join(timeout=joinTimeout)
                if t.is_alive():
                    logWarning(f"Thread '{threadName}' did not join.")
                else:
                    logDebug(f"Thread '{threadName}' joined.")
        logInfo("Cleanup complete.")

    # ---- Main Loop Sub-methods for Clarity ----
    def _runInitialSetup(self):
        """Handle initial setup: Launch WSL server if needed, check reachability, load model, start threads & audio stream."""
        logInfo("Running initial setup...")
        # --- Launch WSL Server If Required ---
        serverReachable = True  # Assume okay if not needed or if launch succeeds
        if isinstance(self.asrModelHandler, RemoteNemoClientHandler):
            logInfo(
                "Remote NeMo handler detected, attempting WSL server launch and reachability check...")
            # Wait for the server to become *reachable* (Flask running)
            serverReachable = self._launchWslServer()  # This calls _waitForServerReachable
            if not serverReachable:
                logError(
                    "Automatic WSL NeMo server launch/reachability check failed. Remote transcription requires manual server start.")
            else:
                logInfo("WSL Server is reachable. Proceeding with model load check/trigger.")
        # --- Initial Model Load Check/Attempt ---
        # This happens AFTER confirming server reachability (for remote) or immediately (for local)
        if serverReachable:  # Only attempt load if server is reachable (for remote) or if local
            logInfo("Attempting initial ASR model load/check...")
            loadSuccess = False
            try:
                # For RemoteNemo, this sends POST /load (if needed) and polls /status for 'loaded'
                # For LocalWhisper, this performs the actual local model loading.
                loadSuccess = self.asrModelHandler.loadModel()
                if not loadSuccess:
                    logError("Initial ASR model load/check failed.")
                    # Handle specific cases if needed
                    if isinstance(self.asrModelHandler, WhisperModelHandler):
                        logError(
                            "Critical: Local Whisper model failed to load. Check model name/path and dependencies.")
                        logError("Aborting application start due to local model load failure.")
                        self.stateManager.stopProgram()
                        return  # Abort setup
                    elif isinstance(self.asrModelHandler, RemoteNemoClientHandler):
                        logWarning(
                            "Failed to trigger initial load or confirm 'loaded' status on remote server (check server logs). Will rely on ModelLifecycleManager retries.")
                else:
                    logInfo("Initial ASR model load/check successful.")
            except Exception as e:
                logError(f"Critical error during initial model load/check: {e}", exc_info=True)
                logError("Aborting application start due to model load error.")
                self.stateManager.stopProgram()
                return  # Abort setup
        else:
            # Server wasn't reachable, cannot proceed with load attempt
            logWarning("Skipping initial model load attempt as server was not reachable.")
        # --- Start Background Threads ---
        # Only start threads if the program hasn't been stopped by a critical failure above
        if self.stateManager.shouldProgramContinue():
            self._startBackgroundThreads()
        else:
            logWarning("Skipping background thread start due to earlier critical error.")
        # --- Initial Audio Stream Start ---
        initialStreamStarted = False
        # Only start if program is still running and recording is initially enabled
        if self.stateManager.shouldProgramContinue() and self.stateManager.isRecording():
            logInfo("Initial state is recording: attempting to start audio stream...")
            initialStreamStarted = self.audioHandler.startStream()
            if not initialStreamStarted:
                logError("CRITICAL: Failed to start audio stream initially. Recording disabled.")
                self.stateManager.stopRecording()
        elif self.stateManager.shouldProgramContinue():
            logInfo("Initial state is not recording, audio stream will start when toggled on.")
        logInfo("Initial setup phase complete.")

    def _runCheckTimeoutsAndGlobalState(self):
        """Check program/recording timeouts, manage global state."""
        if not all([self.stateManager, self.realTimeProcessor]): return True
        if self.stateManager.checkProgramTimeout(): return False
        self.realTimeProcessor.clearBufferIfOutputDisabled()
        if self.stateManager.isRecording():
            if self.stateManager.checkRecordingTimeout() or self.stateManager.checkIdleTimeout():
                self.toggleRecording()
        return True

    def _runManageAudioStreamLifecycle(self):
        """Start/stop audio stream based on the desired recording state."""
        if not all([self.stateManager, self.audioHandler]): return True
        shouldRecord = self.stateManager.isRecording()
        isRecording = self.audioHandler.stream is not None and self.audioHandler.stream.active
        try:
            if shouldRecord and not isRecording:
                if not self.audioHandler.startStream(): self.stateManager.stopRecording()
            elif not shouldRecord and isRecording:
                self.audioHandler.stopStream()
                if self.realTimeProcessor: self.realTimeProcessor.clearBuffer()
                if self.audioHandler: self.audioHandler.clearQueue()
        except Exception as e:
            logError(f"Audio stream lifecycle error: {e}",
                     exc_info=True);
            self.stateManager.stopRecording()
        return True

    def _runProcessAudioChunks(self):
        """Dequeue and process audio chunks."""
        if not all([self.stateManager, self.audioHandler, self.realTimeProcessor]): return False
        processed = False
        if self.stateManager.isRecording():
            isActive = self.audioHandler.stream is not None and self.audioHandler.stream.active
            if not isActive: return False
            count = 0;
            maxCount = 50
            while count < maxCount:
                chunk = self.audioHandler.getAudioChunk()
                if chunk is None: break
                if self.realTimeProcessor.processIncomingChunk(chunk): processed = True
                count += 1
        return processed

    def _runQueueTranscriptionRequest(self, audioProcessed):
        """Check trigger and queue transcription."""
        if not all([self.stateManager, self.realTimeProcessor, self.config]): return
        check = self.stateManager.isOutputEnabled() and \
                (audioProcessed or self.config.get('transcriptionMode') == 'dictationMode')
        if check:
            data = self.realTimeProcessor.checkTranscriptionTrigger()
            if data is not None:
                rate = self.config.get('actualSampleRate')
                if not rate or rate <= 0: logError("Invalid sampleRate."); return
                try:
                    self.transcriptionRequestQueue.put((data, rate), block=True, timeout=0.5)
                    logDebug(f"Queued {len(data) / rate:.2f}s.")
                except queue.Full:
                    logWarning("Queue full.")
                except Exception as e:
                    logError(f"Queue put error: {e}", exc_info=True)

    def _runLoopSleep(self):
        """Sleep briefly."""
        time.sleep(0.01)

    # ========================================
    # ==         MAIN EXECUTION LOOP        ==
    # ========================================
    def run(self):
        """Main execution loop."""
        logInfo("Starting main orchestrator loop...")
        initialSetupOk = False
        try:
            self._runInitialSetup()
            if not self.stateManager.shouldProgramContinue():
                logError("Setup failed.")
            else:
                initialSetupOk = True
            if initialSetupOk:
                logInfo("Entering main processing loop...")
                while self.stateManager.shouldProgramContinue():
                    if not self._runCheckTimeoutsAndGlobalState(): break
                    self._runManageAudioStreamLifecycle()
                    processed = self._runProcessAudioChunks()
                    self._runQueueTranscriptionRequest(processed)
                    self._runLoopSleep()
        except KeyboardInterrupt:
            logInfo("\nKeyboardInterrupt. Stopping...")
        except Exception as e:
            logError(f"\n!!! CRITICAL MAIN LOOP ERROR: {e}", exc_info=True)
        finally:
            if self.stateManager: self.stateManager.stopProgram()  # Ensure stop signal sent on exit
            logInfo("Exiting main loop." if initialSetupOk else "Exiting after setup failure.")
            self._cleanup()
            logInfo("Orchestrator run finished.")
