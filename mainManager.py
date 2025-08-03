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
import logging  # Needed for checking log level

# Import application components
from audioProcesses import AudioHandler, RealTimeAudioProcessor
from managers import ConfigurationManager, StateManager, ModelLifecycleManager
from modelHandlers import WhisperModelHandler, RemoteNemoClientHandler
# Assuming these exist based on previous context - ensure they are implemented
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
        # Use imported logDebug directly. Logging level is controlled by the setup in useRealtimeTranscription.py
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
        # These methods will populate the attributes initialized above
        self._initializeCoreComponents()
        self._initializeAsrHandler()  # Sets self.asrModelHandler and prepares self.wslLaunchCommand

        # Ensure ASR handler was successfully initialized before proceeding
        if not self.asrModelHandler:
            # Error already logged in _initializeAsrHandler
            raise RuntimeError(
                "ASR Handler could not be initialized. Check configuration and logs. Cannot continue.")

        # --- Instantiate Model Lifecycle Manager (depends on ASR Handler) ---
        # Make sure systemInteractionHandler is initialized before passing it
        if not self.systemInteractionHandler:
            # Should have been initialized in _initializeCoreComponents
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
        # Assuming SystemInteractionHandler and TranscriptionOutputHandler exist and are imported
        self.systemInteractionHandler = SystemInteractionHandler(self.config)
        # Output handler needs systemInteractionHandler
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

        # Reset WSL attributes before attempting initialization
        self.wslServerProcess = None
        self.wslLaunchCommand = []

        if modelNameLower.startswith("nvidia/"):
            logInfo(f"Configured Nvidia model: '{modelName}'. Preparing RemoteNemoClientHandler.")
            # Check if running on Windows, as WSL launch logic assumes it
            if platform.system() != "Windows":
                logError("Remote Nvidia model selected, but application is not running on Windows.")
                logError("Automatic WSL server launching is only supported from Windows.")
                # Application can still run if server is started manually, but auto-launch won't work.
                # Proceed with handler init, but command prep will likely fail or be useless.
            self._prepareWslLaunchCommand(modelName)  # Attempt to prepare command
            self.asrModelHandler = RemoteNemoClientHandler(
                self.config)  # Instantiate client handler

        elif modelName:  # If modelName is non-empty and not starting with 'nvidia/'
            logInfo(f"Configured non-Nvidia model: '{modelName}'. Using local WhisperModelHandler.")
            self.asrModelHandler = WhisperModelHandler(self.config)  # Instantiate local handler
            self.wslServerProcess = None  # Ensure WSL attributes are cleared
            self.wslLaunchCommand = []
        else:
            # No model name configured is a critical error
            logError(
                "CRITICAL: No 'modelName' specified in configuration. Cannot initialize ASR handler.")
            self.asrModelHandler = None  # Explicitly set to None
            # Let the caller (__init__) handle the error state (e.g., raise RuntimeError)

        if self.asrModelHandler:
            logDebug(f"ASR Handler initialized: {type(self.asrModelHandler).__name__}")
        else:
            logError("ASR Handler initialization failed.")  # Should be caught by __init__

    def _prepareWslLaunchCommand(self, modelName):
        """Prepares the command list needed to launch the WSL NeMo server."""
        logDebug("Preparing WSL server launch command...")
        self.wslLaunchCommand = []  # Ensure it's reset

        # Required config settings
        wslServerUrl = self.config.get('wslServerUrl')
        wslDistro = self.config.get('wslDistributionName')

        if not wslServerUrl or not wslDistro:
            logError(
                "Config error: 'wslServerUrl' & 'wslDistributionName' are required for automatic WSL server launch.")
            logError("WSL server will NOT be launched automatically.")
            return  # Cannot proceed with command preparation

        # Check if on Windows before proceeding with WSL command construction
        if platform.system() != "Windows":
            logWarning("WSL launch command preparation skipped: Not running on Windows.")
            return

        try:
            # 1. Get Port from URL
            parsedUrl = urlparse(wslServerUrl)
            wslServerPort = parsedUrl.port
            if not wslServerPort:
                raise ValueError(f"Could not extract port from wslServerUrl: {wslServerUrl}")

            # 2. Find WSL Server Script Path ('wslNemoServer.py')
            # Assume it's in the same directory as the main running script
            import __main__  # Get the entry point script's context
            # Check if __main__ has __file__ attribute (might not in some run contexts)
            if hasattr(__main__, '__file__') and __main__.__file__:
                main_file_path = Path(os.path.abspath(__main__.__file__))
                scriptDir = main_file_path.parent
            else:
                # Fallback: Use the directory of *this* file (mainManager.py)
                # This might be less reliable if the structure changes.
                logWarning(
                    "Could not reliably determine main script path (__main__.__file__ missing). Falling back to mainManager.py directory.")
                scriptDir = Path(os.path.dirname(os.path.abspath(__file__)))

            wslServerScriptFilename = "wslNemoServer.py"
            wslServerScriptPathWindows = scriptDir / wslServerScriptFilename
            logDebug(f"Looking for WSL server script at: {wslServerScriptPathWindows}")

            if not wslServerScriptPathWindows.is_file():
                # Fallback: Check current working directory
                cwd = Path.cwd()
                fallbackPath = cwd / wslServerScriptFilename
                logDebug(
                    f"Script not found in script dir, checking fallback CWD path: {fallbackPath}")
                if not fallbackPath.is_file():
                    raise FileNotFoundError(
                        f"WSL script '{wslServerScriptFilename}' not found in script dir ({scriptDir}) or CWD ({cwd}).")
                wslServerScriptPathWindows = fallbackPath  # Use fallback path
                logWarning(f"Using WSL server script from CWD: {wslServerScriptPathWindows}")

            logDebug(f"Found WSL server script (Windows path): {wslServerScriptPathWindows}")

            # 3. Convert Windows Path to WSL Path using the utility function
            wslServerScriptPathWsl = convertWindowsPathToWsl(wslServerScriptPathWindows)
            if not wslServerScriptPathWsl:
                # Error logged within convertWindowsPathToWsl
                raise ValueError(
                    f"Failed to convert Windows path to WSL path: {wslServerScriptPathWindows}")
            logDebug(f"Converted WSL server script path (WSL path): {wslServerScriptPathWsl}")

            # 4. Construct Command List
            # Assume python3 is in the WSL distribution's PATH
            # Using /usr/bin/python3 might be slightly more robust if PATH issues arise
            pythonExecutable = "python3"  # Or "/usr/bin/python3"
            preparedCommand = [
                "wsl.exe",  # WSL command itself
                "-d", wslDistro,  # Specify the distribution
                "--",  # Separator: passes subsequent args directly to the command in WSL
                pythonExecutable,
                wslServerScriptPathWsl,
                "--model_name", modelName,
                "--port", str(wslServerPort),
                "--load_on_start"  # Tell server to load model immediately
            ]

            # 5. Assign the prepared command list
            self.wslLaunchCommand = preparedCommand
            logInfo("Prepared WSL server launch command successfully.")
            logDebug(f"WSL Command: {' '.join(self.wslLaunchCommand)}")

        except FileNotFoundError as e:
            logError(f"Error preparing WSL launch (FileNotFound): {e}")
            logError("WSL server will NOT be launched automatically.")
            self.wslLaunchCommand = []  # Ensure command is empty on error
        except ValueError as e:
            logError(f"Error preparing WSL launch (ValueError): {e}")
            logError("WSL server will NOT be launched automatically.")
            self.wslLaunchCommand = []
        except Exception as e:
            logError(f"Unexpected error preparing WSL launch command: {type(e).__name__} - {e}")
            logError(traceback.format_exc())  # Log full traceback
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
        log_message = "\n--- Application Setup ---\n"
        log_message += f"Mode:                 {mode}\n"
        log_message += f"ASR Model:            {modelName}\n"
        log_message += f"  Handler:            {handlerType}\n"
        log_message += f"  Target Device:      {deviceStr}\n"
        if handlerType == 'RemoteNemoClientHandler':
            log_message += f"  WSL Server URL:     {self.config.get('wslServerUrl', 'Not Set!')}\n"
        log_message += f"Audio Device:         ID={devId}, Rate={rate}Hz, Channels={ch}\n"
        log_message += f"--- Hotkeys ---\n"
        log_message += f"Toggle Recording:     '{recKey}'\n"
        log_message += f"Toggle Text Output:   '{outKey}' (Method: {self.systemInteractionHandler.textOutputMethod})\n"
        log_message += f"--- Timeouts ---\n"
        log_message += f"Max Recording:        {maxRecStr}\n"
        log_message += f"Stop Rec After Idle:  {idleTimeStr}\n"
        log_message += f"Unload Model Inactive:{unloadTimeoutStr}\n"
        log_message += f"Program Auto-Exit:    {maxProgramStr}\n"
        log_message += f"-------------------------"

        # Log the entire block as one INFO message
        logInfo(log_message)

    def _launchWslServer(self):
        """Launches the wslNemoServer.py script in WSL using subprocess if configured."""
        if not self.wslLaunchCommand:
            logWarning("WSL server launch command not prepared. Skipping automatic launch.")
            return False  # Indicate launch was skipped

        # Check if we already have a process handle and if it's still running
        if self.wslServerProcess and self.wslServerProcess.poll() is None:
            logInfo(
                f"WSL server process (PID: {self.wslServerProcess.pid}) appears to be already running. Skipping launch.")
            return True  # Assume it's the one we need

        # Check platform again before executing wsl.exe
        if platform.system() != "Windows":
            logError("WSL server launch skipped: Cannot execute wsl.exe on non-Windows platform.")
            return False

        logInfo(f"Attempting to launch WSL server...")
        logDebug(f"Executing command: {' '.join(self.wslLaunchCommand)}")
        try:
            # Use Popen for non-blocking execution
            # Hide the WSL console window using CREATE_NO_WINDOW flag on Windows
            # Redirect stdout/stderr to PIPE to capture initial output for diagnostics,
            # but be careful about potential blocking if output is large (server should log to file mainly).
            self.wslServerProcess = subprocess.Popen(
                self.wslLaunchCommand,
                stdout=subprocess.PIPE,  # Capture stdout
                stderr=subprocess.PIPE,  # Capture stderr
                text=True,  # Decode stdout/stderr as text
                encoding='utf-8',  # Specify encoding
                errors='replace',  # Handle potential decoding errors
                creationflags=subprocess.CREATE_NO_WINDOW  # Hide console window (Windows specific)
            )
            logInfo(
                f"WSL server process launched (PID: {self.wslServerProcess.pid}). Allowing time to start...")

            # Wait briefly for the server to initialize
            # A fixed wait is simple but not fully reliable. A better approach would involve
            # checking the server's /status endpoint, but that adds complexity.
            time.sleep(5.0)  # Adjust as needed (e.g., based on config setting)

            # === Check for immediate exit ===
            if self.wslServerProcess.poll() is not None:
                # Process exited quickly, likely an error
                logError(
                    f"WSL server process (PID: {self.wslServerProcess.pid}) exited immediately with code {self.wslServerProcess.returncode}.")
                # Try reading stderr for clues
                try:
                    stderrOutput = self.wslServerProcess.stderr.read() if self.wslServerProcess.stderr else "stderr not captured"
                    if stderrOutput:
                        logError(f"WSL Server stderr hints:\n{stderrOutput.strip()}")
                    else:
                        logWarning("No stderr output captured from WSL server process.")
                    # Also check stdout
                    stdoutOutput = self.wslServerProcess.stdout.read() if self.wslServerProcess.stdout else "stdout not captured"
                    if stdoutOutput:
                        logError(f"WSL Server stdout hints:\n{stdoutOutput.strip()}")

                except Exception as read_e:
                    logWarning(f"Exception reading WSL process stdout/stderr after exit: {read_e}")
                self.wslServerProcess = None  # Clear the handle
                return False  # Indicate launch failure

            # === Process seems to be running, maybe check initial output lines ===
            # Be cautious with blocking reads here. Read non-blockingly if possible, or just assume running.
            # Example: Read first line of stdout/stderr without blocking indefinitely
            # (Requires more complex stream handling, omitted for simplicity here)
            logInfo(
                f"WSL server process (PID: {self.wslServerProcess.pid}) appears to be running after initial wait.")
            return True  # Indicate successful launch (or at least, process started)

        except FileNotFoundError:
            logError(
                f"Error launching WSL server: 'wsl.exe' not found in system PATH. Is WSL installed and configured correctly?")
            self.wslServerProcess = None
            return False
        except Exception as e:
            logError(f"Unexpected error launching WSL server process: {e}")
            logError(traceback.format_exc())
            self.wslServerProcess = None
            return False

    def _terminateWslServer(self):
        """Terminates the launched WSL server process if it exists and is running."""
        if not self.wslServerProcess:
            logDebug("No WSL server process handle found to terminate.")
            return

        if self.wslServerProcess.poll() is None:  # Check if the process is still running
            pid = self.wslServerProcess.pid
            logInfo(f"Attempting to terminate launched WSL server process (PID: {pid})...")
            try:
                # Start with terminate (SIGTERM equivalent)
                self.wslServerProcess.terminate()
                logDebug(f"Sent terminate signal to WSL process {pid}.")
                # Wait for a short period for graceful shutdown
                try:
                    self.wslServerProcess.wait(timeout=3.0)  # Wait up to 3 seconds
                    logInfo(f"WSL server process (PID: {pid}) terminated gracefully.")
                except subprocess.TimeoutExpired:
                    logWarning(
                        f"WSL server process (PID: {pid}) did not terminate gracefully within timeout. Forcing kill...")
                    # If terminate times out, force kill (SIGKILL equivalent)
                    self.wslServerProcess.kill()
                    logDebug(f"Sent kill signal to WSL process {pid}.")
                    # Short wait after kill
                    time.sleep(0.5)
                    # Check final status after kill
                    if self.wslServerProcess.poll() is not None:
                        logInfo(f"WSL server process (PID: {pid}) killed successfully.")
                    else:
                        logWarning(
                            f"WSL server process (PID: {pid}) did not exit after kill signal.")

                # Optionally capture any final output after termination/kill attempt
                try:
                    stdout, stderr = self.wslServerProcess.communicate(
                        timeout=1.0)  # Short timeout for final reads
                    if stdout: logDebug(f"WSL Server final stdout: {stdout.strip()}")
                    if stderr: logWarning(f"WSL Server final stderr: {stderr.strip()}")
                except subprocess.TimeoutExpired:
                    logDebug("Timeout reading final output after termination.")
                except ValueError:  # Can happen if streams are already closed
                    logDebug("WSL process streams closed before final communicate.")
                except Exception as comm_e:
                    logWarning(f"Error during final WSL process communicate: {comm_e}")

            except ProcessLookupError:
                # Process might have finished between poll() check and terminate() call
                logInfo(
                    f"WSL server process (PID: {pid}) already finished before termination attempt.")
            except Exception as e:
                logError(f"Error terminating WSL server process (PID: {pid}): {e}")
        else:
            logInfo(
                f"Launched WSL server process (PID: {self.wslServerProcess.pid}) was already finished.")

        # Clear the process handle regardless of outcome
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
        queueTimeoutSeconds = 1.0  # How long to wait for an item before checking program status

        while self.stateManager.shouldProgramContinue():
            try:
                # Wait for audio data with a timeout
                queueItem = self.transcriptionRequestQueue.get(timeout=queueTimeoutSeconds)

                if queueItem is None:  # Sentinel value to exit loop cleanly
                    logDebug("Transcription worker received None sentinel, exiting.")
                    break

                # Check item validity (should be tuple: audio, rate)
                if not isinstance(queueItem, tuple) or len(queueItem) != 2:
                    logWarning(
                        f"Invalid item received in transcription queue: {type(queueItem)}. Skipping.")
                    self.transcriptionRequestQueue.task_done()  # Mark as done even if invalid
                    continue

                audioDataToTranscribe, sampleRate = queueItem
                if audioDataToTranscribe is None or sampleRate <= 0:
                    logWarning(
                        f"Skipping invalid transcription request data (Audio: {audioDataToTranscribe is not None}, Rate: {sampleRate}).")
                    self.transcriptionRequestQueue.task_done()  # Mark as done
                    continue

                # Check if model is considered ready by the handler before attempting transcription
                if self.asrModelHandler.isModelLoaded():
                    segmentDuration = len(audioDataToTranscribe) / sampleRate
                    logDebug(
                        f"Worker processing {len(audioDataToTranscribe)} samples ({segmentDuration:.2f}s)...")

                    # --- ASR Inference (Delegated to Handler: Local or Remote) ---
                    startTime = time.time()
                    transcriptionResult = self.asrModelHandler.transcribeAudioSegment(
                        audioDataToTranscribe,
                        sampleRate
                    )
                    inferenceTime = time.time() - startTime
                    logDebug(f"ASR inference took {inferenceTime:.3f} seconds.")

                    # --- Result Handling (via Output Handler) ---
                    # Pass audio data along for loudness checks etc.
                    self.outputHandler.processTranscriptionResult(
                        transcriptionResult,
                        audioDataToTranscribe  # Pass the original audio data
                    )
                    logDebug("Transcription worker finished processing segment.")
                else:
                    # Model not loaded, log warning and potentially clear dictation buffer
                    logWarning(
                        "Transcription worker skipped segment: ASR model is not loaded/ready.")
                    if self.config.get('transcriptionMode') == 'dictationMode' and hasattr(self,
                                                                                           'realTimeProcessor'):
                        # If dictation mode triggered but model wasn't ready, maybe clear buffer
                        # self.realTimeProcessor.clearBuffer() # Be careful with unintended buffer clears
                        logDebug(
                            "Dictation mode state might need reset due to skipped transcription.")

                # Signal that the queue item has been processed
                self.transcriptionRequestQueue.task_done()

            except queue.Empty:
                # Queue was empty during timeout, loop continues to check shouldProgramContinue
                continue  # Normal operation when idle
            except Exception as e:
                logError(f"!!! ERROR in Transcription Worker thread: {e}")
                logError(traceback.format_exc())
                # Avoid busy-looping on repeated errors
                time.sleep(1)

        logInfo("Transcription Worker thread stopping.")

    def _startBackgroundThreads(self):
        """Starts threads for hotkeys, model management, and transcription with error handling."""
        logDebug("Starting background threads...")
        self.threads = []  # Reset list of threads
        failedThreads = []

        # --- Wrapper Function for Robust Thread Execution ---
        def threadWrapper(targetFunc, threadName, *args, **kwargs):
            logDebug(f"Thread '{threadName}' starting...")
            try:
                targetFunc(*args, **kwargs)
                logDebug(f"Thread '{threadName}' finished execution normally.")
            except Exception as e:
                logError(f"!!! UNCAUGHT EXCEPTION in thread '{threadName}': {e}")
                logError(traceback.format_exc())
                # Signal program stop if a critical thread dies unexpectedly
                if threadName in ["KeyboardMonitorThread", "TranscriptionWorkerThread"]:
                    logCritical = getattr(logging, 'critical',
                                          logError)  # Use critical if available
                    logCritical(f"Critical thread '{threadName}' failed, signaling program stop.")
                    if hasattr(self, 'stateManager') and self.stateManager:
                        self.stateManager.stopProgram()  # Signal main loop
            finally:
                # Log regardless of exit reason
                logDebug(f"Thread '{threadName}' has exited.")

        # --- Thread Definitions ---
        threadTargets = {
            "KeyboardMonitorThread": (
            self.systemInteractionHandler.monitorKeyboardShortcuts, (self,)),
            # Pass self (orchestrator)
            "ModelManagerThread": (self.modelLifecycleManager.manageModelLifecycle, ()),
            "TranscriptionWorkerThread": (self._transcriptionWorkerLoop, ()),
        }

        # --- Create and Start Threads ---
        for threadName, (target, targetArgs) in threadTargets.items():
            if target is None:
                logWarning(f"Target function for thread '{threadName}' is None. Skipping start.")
                failedThreads.append(threadName)
                continue

            try:
                thread = threading.Thread(
                    target=threadWrapper,  # Use the robust wrapper
                    args=(target, threadName) + targetArgs,
                    # Pass target, name, and its args to wrapper
                    name=threadName,
                    daemon=True  # Daemon threads exit automatically if main thread exits
                )
                self.threads.append(thread)
                thread.start()
                logDebug(f"Thread '{threadName}' initiated.")
            except Exception as e:
                logError(f"Failed to create/start thread '{threadName}': {e}")
                failedThreads.append(threadName)

        # --- Post-Start Checks ---
        time.sleep(0.1)  # Give threads a moment to potentially fail immediately
        activeThreads = [t for t in self.threads if t.is_alive()]
        activeThreadCount = len(activeThreads)
        expectedThreadCount = len(threadTargets) - len(
            failedThreads)  # Expected count after launch attempts

        logDebug(
            f"Attempted to start {len(threadTargets)} background threads. {activeThreadCount} appear active.")

        if failedThreads:
            logError(f"Failed to initiate threads: {', '.join(failedThreads)}")
        if activeThreadCount < expectedThreadCount:
            logWarning(
                "One or more threads may have exited immediately after starting. Check logs above.")

        # Specific checks for critical threads
        if "KeyboardMonitorThread" not in [t.name for t in activeThreads]:
            logError("Keyboard monitor thread failed to start or died. Hotkeys will NOT work.")
            logError("--> If on Windows, try running the script as Administrator.")
            logError(
                "--> If on Linux, ensure your user has permissions (e.g., 'input' group) or run with 'sudo'.")
        if "TranscriptionWorkerThread" not in [t.name for t in activeThreads]:
            logError(
                "Transcription worker thread failed to start or died. Transcription processing will NOT occur.")

    # --- Public Methods for Hotkey Actions ---
    def toggleRecording(self):
        """Toggles the recording state. Called by systemInteractionHandler via hotkey."""
        if not self.stateManager or not self.systemInteractionHandler or not self.realTimeProcessor or not self.audioHandler:
            logError("Cannot toggle recording: Core components not initialized.")
            return

        if self.stateManager.isRecording():
            # --- Stop Recording ---
            if self.stateManager.stopRecording():  # Returns True if state actually changed
                self.systemInteractionHandler.playNotification("recordingOff")
                logInfo("Recording stopped via hotkey.")
                # Clear buffers immediately on manual stop to prevent processing stale audio
                self.realTimeProcessor.clearBuffer()
                self.audioHandler.clearQueue()
                # Audio stream stopping is handled by the main loop's lifecycle management
        else:
            # --- Start Recording ---
            # Check if model is ready before allowing recording state change? Optional.
            # modelReady = self.asrModelHandler.isModelLoaded() if self.asrModelHandler else False
            # if not modelReady:
            #      logWarning("Cannot start recording: Model is not loaded/ready.")
            #      # Play a different sound? E.g., self.systemInteractionHandler.playNotification("error")
            #      return

            if self.stateManager.startRecording():  # Returns True if state actually changed
                # Play sound only if state changed to ON
                if self.config.get('playEnableSounds', False):
                    self.systemInteractionHandler.playNotification("recordingOn")
                else:  # Play standard 'on' sound if enable sounds are off
                    self.systemInteractionHandler.playNotification(
                        "recordingOn")  # Assuming default behavior is desired sound on toggle
                logInfo("Recording started via hotkey.")
                # Model loading and stream starting are handled by their respective managers/loops

    def toggleOutput(self):
        """Toggles the text output state. Called by systemInteractionHandler via hotkey."""
        if not self.stateManager or not self.systemInteractionHandler or not self.realTimeProcessor:
            logError("Cannot toggle output: Core components not initialized.")
            return

        newState = self.stateManager.toggleOutput()  # This logs the state change
        notification = "outputEnabled" if newState else "outputDisabled"

        # Play sound only if state changed and appropriate setting is enabled
        if newState and self.config.get('playEnableSounds', False):
            self.systemInteractionHandler.playNotification("outputEnabled")
        elif not newState:  # Always play disable sound if notifications enabled
            self.systemInteractionHandler.playNotification("outputDisabled")

        # Optional: If output just got disabled, clear the processor buffer to prevent backlog
        if not newState:
            self.realTimeProcessor.clearBufferIfOutputDisabled()

    # ---- Cleanup ---
    def _cleanup(self):
        """Cleans up all resources: stops threads, audio, model, WSL server."""
        logInfo("Initiating orchestrator cleanup...")
        # 1. Signal all loops and threads to stop
        if self.stateManager:
            self.stateManager.stopProgram()
            logDebug("Program stop signaled to state manager.")
        else:
            logWarning("StateManager not available during cleanup.")

        # 2. Signal transcription worker to finish by putting None sentinel
        # Use timeout to avoid blocking if queue is full or worker died
        try:
            self.transcriptionRequestQueue.put(None, block=True, timeout=0.5)
            logDebug("Sent None sentinel to transcription worker queue.")
        except queue.Full:
            logWarning(
                "Transcription queue full during cleanup, worker might not receive sentinel.")
        except Exception as e:
            logWarning(f"Error putting sentinel on transcription queue: {e}")

        # 3. Stop audio stream explicitly (if running)
        if self.audioHandler:
            logInfo("Stopping audio stream...")
            self.audioHandler.stopStream()
        else:
            logWarning("AudioHandler not available during cleanup.")

        # 4. Terminate WSL Server Process (if launched and still exists)
        # Do this *before* attempting to unload the model via the client handler
        self._terminateWslServer()

        # 5. Cleanup ASR Handler (local unload or signal remote server if configured)
        if self.asrModelHandler:
            logInfo("Cleaning up ASR model handler...")
            try:
                self.asrModelHandler.cleanup()  # This might call unloadModel internally
            except Exception as e:
                logError(f"Error during ASR handler cleanup: {e}", exc_info=True)
        else:
            logWarning("AsrModelHandler not available during cleanup.")

        # 6. Clear processing buffers/queues (redundant if stream stopped, but safe)
        if self.realTimeProcessor:
            logInfo("Clearing audio processing buffer...")
            self.realTimeProcessor.clearBuffer()
        if self.audioHandler:
            logInfo("Clearing audio input queue...")
            self.audioHandler.clearQueue()

        # 7. Cleanup System Interaction Handler (e.g., pygame mixer)
        if self.systemInteractionHandler:
            logInfo("Cleaning up system interaction handler...")
            try:
                self.systemInteractionHandler.cleanup()
            except Exception as e:
                logError(f"Error during system interaction handler cleanup: {e}", exc_info=True)
        else:
            logWarning("SystemInteractionHandler not available during cleanup.")

        # 8. Attempt to join background threads
        logInfo("Attempting to join background threads...")
        joinTimeout = 2.0  # Seconds to wait per thread
        # Make a copy in case a thread modifies the list (unlikely with daemons)
        threads_to_join = list(self.threads)
        self.threads = []  # Clear original list

        for t in threads_to_join:
            threadName = t.name if hasattr(t, 'name') else "Unknown Thread"
            if t is not None and t.is_alive():
                logDebug(f"Joining thread '{threadName}'...")
                try:
                    t.join(timeout=joinTimeout)
                    if t.is_alive():
                        logWarning(
                            f"Thread '{threadName}' did not terminate within {joinTimeout}s.")
                    else:
                        logDebug(f"Thread '{threadName}' joined successfully.")
                except Exception as e:
                    logWarning(f"Error joining thread '{threadName}': {e}")
            # else:
            #    logDebug(f"Thread '{threadName}' was None or not alive before join attempt.")

        logInfo("Orchestrator cleanup complete.")

    # ---- Main Loop Sub-methods for Clarity ----
    def _run_initialSetup(self):
        """Handle initial setup: Launch WSL server if needed, load model, start threads & audio stream."""
        logInfo("Running initial setup...")

        # --- Launch WSL Server If Required ---
        serverLaunchedOk = True  # Assume okay if not needed
        if isinstance(self.asrModelHandler, RemoteNemoClientHandler):
            logInfo("Remote NeMo handler detected, attempting WSL server launch...")
            serverLaunchedOk = self._launchWslServer()  # This logs details
            if not serverLaunchedOk:
                # Log critical failure but allow app to continue, relying on manual server start
                logError(
                    "Automatic WSL NeMo server launch failed. Remote transcription will require manual server start.")
                # Don't return early, let loadModel attempt connection later

        # --- Initial Model Load Check/Attempt ---
        # Try to load/prepare the model regardless of recording state initially? Or only if recording?
        # Let's attempt it always, as the lifecycle manager might unload it later if inactive.
        logInfo("Ensuring ASR model is loaded/ready...")
        try:
            load_success = self.asrModelHandler.loadModel()  # Handles local or remote load attempt
            if not load_success:
                # loadModel logs the specific error (local failure or remote connection issue)
                logError("Initial model load/preparation failed.")
                # If remote and server launch also failed, this is expected.
                if isinstance(self.asrModelHandler,
                              RemoteNemoClientHandler) and not serverLaunchedOk:
                    logWarning(
                        "Model load failure likely due to automatic WSL server launch failure.")
                # Consider if failure is critical - maybe stop if local load fails?
                # if isinstance(self.asrModelHandler, WhisperModelHandler):
                #     logCritical("Local model failed to load. Cannot continue.")
                #     self.stateManager.stopProgram()
                #     return # Abort setup
            else:
                logInfo("Initial model load/check successful.")
        except Exception as e:
            logError(f"Critical error during initial model load/check: {e}", exc_info=True)
            # Maybe stop program depending on severity/model type
            # self.stateManager.stopProgram()
            # return # Abort setup

        # --- Start Background Threads ---
        # Do this after attempting model load so threads have a handler instance
        self._startBackgroundThreads()

        # --- Initial Audio Stream Start ---
        # Only start if configured to start recording immediately
        initialStreamStarted = False
        if self.stateManager.isRecording():
            logInfo("Initial state is recording: attempting to start audio stream...")
            initialStreamStarted = self.audioHandler.startStream()
            if not initialStreamStarted:
                logError("CRITICAL: Failed to start audio stream initially. Recording disabled.")
                # If stream fails, disable recording state
                self.stateManager.stopRecording()
        else:
            logInfo("Initial state is not recording, audio stream will start when toggled on.")

        logInfo("Initial setup phase complete.")

    def _run_checkTimeoutsNGlobalState(self):
        """Check program/recording timeouts, manage global state like clearing buffer if output disabled."""
        if not self.stateManager or not self.realTimeProcessor:
            logWarning(
                "Skipping timeout/global state check: StateManager or RealTimeProcessor not available.")
            return True  # Allow loop to continue, but log issue

        # --- Program Timeout ---
        if self.stateManager.checkProgramTimeout():
            logInfo("Maximum program duration reached. Signaling stop.")
            # stateManager.stopProgram() is called inside checkProgramTimeout if condition met
            return False  # Signal loop to exit

        # --- Clear Buffer if Output Disabled ---
        # Prevents audio buildup when user doesn't want output typed/copied
        self.realTimeProcessor.clearBufferIfOutputDisabled()

        # --- Recording-Specific Timeouts (Only check if recording active) ---
        if self.stateManager.isRecording():
            # Max Recording Duration
            if self.stateManager.checkRecordingTimeout():
                logInfo("Maximum recording session duration reached, stopping recording...")
                self.toggleRecording()  # Use toggle method to handle state change and notifications

            # Consecutive Idle Time
            elif self.stateManager.checkIdleTimeout():
                logInfo(
                    "Consecutive idle time reached (no valid transcription output), stopping recording...")
                self.toggleRecording()

        return True  # Signal loop to continue

    def _run_manageAudioStreamLifecycle(self):
        """Start/stop audio stream based on the desired recording state."""
        if not self.stateManager or not self.audioHandler:
            logWarning(
                "Skipping audio stream management: StateManager or AudioHandler not available.")
            return True  # Allow loop to continue

        shouldStreamBeActive = self.stateManager.isRecording()
        isStreamActuallyActive = self.audioHandler.stream is not None and self.audioHandler.stream.active

        try:
            if shouldStreamBeActive and not isStreamActuallyActive:
                # Need to start the stream
                logDebug("Attempting to start audio stream (recording is active)...")
                if not self.audioHandler.startStream():
                    # Failed to start stream, log error and disable recording state
                    logError("Failed to start/restart audio stream. Disabling recording.")
                    self.stateManager.stopRecording()
                    # No need to return False, state change handles it in next loop iter
                # else: Stream started successfully (logged within startStream)

            elif not shouldStreamBeActive and isStreamActuallyActive:
                # Need to stop the stream
                logDebug("Stopping audio stream (recording is inactive)...")
                self.audioHandler.stopStream()
                # Also clear buffers when stream stops manually or due to state change
                if self.realTimeProcessor: self.realTimeProcessor.clearBuffer()
                if self.audioHandler: self.audioHandler.clearQueue()

        except Exception as e:
            logError(f"Error during audio stream lifecycle management: {e}", exc_info=True)
            # Attempt to recover by stopping stream and disabling recording if error occurs
            if self.audioHandler: self.audioHandler.stopStream()
            self.stateManager.stopRecording()

        return True  # Always continue the main loop

    def _run_processAudioChunks(self):
        """Dequeue and process audio chunks from the audio handler's queue into the processor's buffer."""
        if not self.stateManager or not self.audioHandler or not self.realTimeProcessor:
            logWarning("Skipping audio chunk processing: Required components missing.")
            return False

        audioProcessedThisLoop = False
        # Only process if recording is active
        if self.stateManager.isRecording():
            # Check if stream is active before getting chunks? Optional, callback handles adding.
            # isStreamActive = self.audioHandler.stream is not None and self.audioHandler.stream.active
            # if not isStreamActive:
            #      logDebug("Audio stream not active, skipping chunk processing.")
            #      return False

            # Process all available chunks in the queue to minimize latency
            processedChunkCount = 0
            maxChunksPerLoop = 50  # Limit to prevent blocking main loop too long if queue grows large
            while processedChunkCount < maxChunksPerLoop:
                chunk = self.audioHandler.getAudioChunk()  # Non-blocking get
                if chunk is None:
                    break  # No more chunks currently in queue

                # Pass chunk to the RealTimeAudioProcessor
                if self.realTimeProcessor.processIncomingChunk(chunk):
                    audioProcessedThisLoop = True  # Mark if any chunk was successfully added
                processedChunkCount += 1

            # if processedChunkCount > 0: logDebug(f"Processed {processedChunkCount} audio chunks from queue.")

        return audioProcessedThisLoop

    def _run_queueTranscriptionRequest(self, audioProcessedThisLoop):
        """Checks if transcription should be triggered and queues the request for the worker thread."""
        if not self.stateManager or not self.realTimeProcessor or not self.config:
            logWarning("Skipping transcription trigger check: Required components missing.")
            return

        # Only check trigger conditions if output is potentially enabled
        # and if audio was potentially added to the buffer this loop (optimization)
        if self.stateManager.isOutputEnabled() and audioProcessedThisLoop:

            # Ask the processor if conditions are met based on its internal state/mode
            # This method returns the audio data to transcribe if ready, or None
            audioDataToTranscribe = self.realTimeProcessor.checkTranscriptionTrigger()

            if audioDataToTranscribe is not None:
                # We have audio data ready for transcription
                actualSampleRate = self.config.get('actualSampleRate')
                if not actualSampleRate or actualSampleRate <= 0:
                    logError(
                        "Cannot queue transcription request: Invalid actualSampleRate in config.")
                    return

                # Send data to the background worker thread via the queue
                try:
                    queueItem = (audioDataToTranscribe, actualSampleRate)
                    # Use non-blocking put with timeout to avoid deadlocks if worker thread died
                    self.transcriptionRequestQueue.put(queueItem, block=True, timeout=0.5)
                    segDuration = len(audioDataToTranscribe) / actualSampleRate
                    logDebug(
                        f"Queued transcription request for {len(audioDataToTranscribe)} samples ({segDuration:.2f}s).")
                except queue.Full:
                    logWarning(
                        "Transcription request queue is full. Skipping current segment to avoid backlog.")
                    # If dictation mode triggered but queue full, need to reset processor state
                    if self.config.get('transcriptionMode') == 'dictationMode':
                        self.realTimeProcessor.clearBuffer()
                        self.realTimeProcessor.isCurrentlySpeaking = False
                        self.realTimeProcessor.silenceStartTime = None
                        logDebug("Reset dictation state after failing to queue.")
                except Exception as e:
                    logError(f"Failed to queue transcription request: {e}", exc_info=True)

    def _run_loopSleep(self):
        """Sleep briefly to prevent high CPU usage, especially when idle."""
        # A very short sleep is usually sufficient
        time.sleep(0.01)  # 10ms sleep

    # ========================================
    # ==         MAIN EXECUTION LOOP        ==
    # ========================================
    def run(self):
        """Main execution loop orchestrating the real-time transcription process."""
        logInfo("Starting main orchestrator loop...")
        try:
            # --- Initial Setup Phase ---
            # Handles WSL launch, initial model load, thread starts, initial stream start
            self._run_initialSetup()

            # Check if initial setup signaled a critical failure (e.g., local model load failed)
            if not self.stateManager.shouldProgramContinue():
                logError("Main loop cannot start due to critical error during initial setup.")
                self._cleanup()  # Attempt cleanup even if setup failed
                return

            logInfo("Entering main processing loop...")
            # --- Main Loop ---
            while self.stateManager.shouldProgramContinue():

                # 1. Check Timeouts & Global State Changes
                #    (e.g., max program duration, max recording, idle timeout)
                if not self._run_checkTimeoutsNGlobalState():
                    logInfo("Exiting main loop due to timeout or state change signal.")
                    break  # Exit loop if program timeout reached or stop signaled

                # 2. Manage Audio Stream Lifecycle
                #    (Starts/stops sounddevice stream based on recording state)
                self._run_manageAudioStreamLifecycle()

                # 3. Process Incoming Audio Chunks
                #    (Dequeues from AudioHandler, passes to RealTimeAudioProcessor buffer)
                #    Returns true if any audio was added to the buffer this iteration
                audioProcessed = self._run_processAudioChunks()

                # 4. Check Transcription Trigger & Queue Request
                #    (RealTimeProcessor checks its buffer/mode; if ready, queues data for worker)
                self._run_queueTranscriptionRequest(audioProcessed)

                # 5. Short Sleep
                #    (Yield CPU to prevent busy-waiting, especially when idle)
                self._run_loopSleep()

        except KeyboardInterrupt:
            logInfo("\nKeyboardInterrupt detected. Stopping application...")
            if self.stateManager: self.stateManager.stopProgram()  # Signal cleanup
        except Exception as e:
            logError(f"\n!!! UNEXPECTED CRITICAL ERROR in main orchestrator loop: {e}")
            logError(traceback.format_exc())
            if self.stateManager: self.stateManager.stopProgram()  # Signal cleanup
        finally:
            logInfo("Exiting main orchestrator loop.")
            # --- Cleanup Phase ---
            # Ensures threads are joined, resources released, WSL server terminated
            self._cleanup()
            logInfo("Orchestrator run method finished.")
