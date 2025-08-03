# mainManager.py

# ==============================================================================
# Real-Time Speech-to-Text - Main Execution Script
# ==============================================================================
#
# Purpose:
# - Entry point for the real-time transcription application.
# - Sets up user configuration.
# - Instantiates the main `SpeechToTextOrchestrator`.
# - The orchestrator determines whether to use a local ASR model (like Whisper)
#   or connect to the remote NeMo server based on the configured `modelName`.
# - Runs the main processing loop.
#
# Usage:
# - Configure settings in the `userSettings` dictionary below.
# - If using an 'nvidia/' model, ensure the `wslNemoServer.py` script is
#   running in your WSL environment first.
# - Run this script from your primary OS (e.g., Windows): `python useRealtimeTranscription.py`
#
# Dependencies (Ensure installed on the system running this script):
# - All dependencies listed in `mainTranscriberLogic.py`
# ==============================================================================
# ==============================================================================
# Real-Time Speech-to-Text Transcription Tool - Core Logic
# ==============================================================================
#
# Purpose:
# - Contains the main application classes responsible for configuration, state,
#   audio input/processing, output handling, system interaction (hotkeys, sounds),
#   and model lifecycle management (when running locally).
# - Defines the abstract base class for ASR models.
# - Includes the concrete implementation for local models (e.g., Whisper via Transformers).
# - Includes a client handler class (`RemoteNemoClientHandler`) responsible for
#   communicating with a separate server process (running in WSL) for models
#   that require it (e.g., NeMo models).
#
# Architecture Notes:
# - This file forms the core of the application run on the primary OS (e.g., Windows).
# - It uses composition: the main Orchestrator holds instances of components.
# - ASR models are accessed through the AbstractAsrModelHandler interface,
#   allowing either local processing or remote calls via the client handler.
#
# Dependencies (Ensure installed on the system running this code):
# - Python standard libraries (abc, gc, os, queue, threading, time, pathlib, platform, subprocess, shutil, string)
# - sounddevice: For audio input.
# - soundfile: For audio file operations (used by FileTranscriber).
# - numpy: For numerical audio data manipulation.
# - torch: Required by Transformers and potentially other local models.
# - transformers: For Hugging Face models (like Whisper).
# - huggingface_hub: For listing models.
# - keyboard: For global hotkey monitoring.
# - pygame: For audio notifications.
# - requests: For communicating with the WSL ASR server (used by RemoteNemoClientHandler).
# - pyautogui: (Optional, for Windows native typing) - install if needed.
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

from audioProcesses import AudioHandler, RealTimeAudioProcessor
from managers import ConfigurationManager, StateManager, ModelLifecycleManager
from modelHandlers import WhisperModelHandler, RemoteNemoClientHandler
from systemInteractions import SystemInteractionHandler
from tasks import TranscriptionOutputHandler
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
    """

    def __init__(self, **userConfig):
        """Initializes the orchestrator by setting up config and calling helper methods."""
        # --- Basic Setup ---
        self.config = ConfigurationManager(**userConfig)
        self._logDebug = lambda msg: logDebug(msg, self.config.get('debugPrint'))
        self._logDebug("Initializing SpeechToText Orchestrator...")

        # Initialize attributes that might be set later
        self.asrModelHandler = None
        self.wslServerProcess = None  # Handle for the launched WSL process
        self.wslLaunchCommand = []  # Store the command to launch WSL server

        # --- Instantiate Components & Handlers ---
        self._initializeCoreComponents()
        self._initializeAsrHandler()  # Sets self.asrModelHandler and prepares self.wslLaunchCommand

        # Ensure ASR handler was successfully initialized before proceeding
        if not self.asrModelHandler:
            raise RuntimeError(
                "ASR Handler could not be initialized. Check configuration and logs.")

        # --- Instantiate Model Lifecycle Manager ---
        self.modelLifecycleManager = ModelLifecycleManager(
            self.config, self.stateManager, self.asrModelHandler, self.systemInteractionHandler
        )

        # --- Threading Setup ---
        self.transcriptionRequestQueue = queue.Queue()
        self.threads = []

        # --- Final Steps ---
        self._printInitialInstructions()

    def _initializeCoreComponents(self):
        """Initializes the core state and interaction components."""
        self._logDebug("Initializing core components...")
        self.stateManager = StateManager(self.config)
        self.systemInteractionHandler = SystemInteractionHandler(self.config)
        self.audioHandler = AudioHandler(self.config, self.stateManager)
        self.realTimeProcessor = RealTimeAudioProcessor(self.config, self.stateManager)
        self.outputHandler = TranscriptionOutputHandler(self.config, self.stateManager,
                                                        self.systemInteractionHandler)
        self._logDebug("Core components initialized.")

    def _initializeAsrHandler(self):
        """Determines and initializes the appropriate ASR handler (local or remote client)."""
        self._logDebug("Initializing ASR handler...")
        modelName = self.config.get('modelName', '')
        modelNameLower = modelName.lower()

        # Reset WSL attributes before attempting initialization
        self.wslServerProcess = None
        self.wslLaunchCommand = []

        if modelNameLower.startswith("nvidia/"):
            logInfo(f"Configured Nvidia model: '{modelName}'. Preparing RemoteNemoClientHandler.")
            # Attempt to prepare the launch command; result stored in self.wslLaunchCommand
            self._prepareWslLaunchCommand(modelName)
            # Instantiate the client handler regardless of command prep success
            self.asrModelHandler = RemoteNemoClientHandler(self.config)

        elif modelName:
            logInfo(f"Configured non-Nvidia model: '{modelName}'. Using local WhisperModelHandler.")
            # Instantiate the local handler
            self.asrModelHandler = WhisperModelHandler(self.config)
            # Ensure WSL attributes are default for local models
            self.wslServerProcess = None
            self.wslLaunchCommand = []
        else:
            # No model name configured
            logError("No 'modelName' specified in configuration. Cannot initialize ASR handler.")
            # Set handler to None to indicate failure
            self.asrModelHandler = None
            # Optionally raise an error if this is considered fatal
            # raise ValueError("ASR model name configuration is missing.")

        if self.asrModelHandler:
            self._logDebug(f"ASR Handler initialized: {type(self.asrModelHandler).__name__}")
        else:
            self._logDebug("ASR Handler initialization failed.")

    def _prepareWslLaunchCommand(self, modelName):
        """Prepares the command list needed to launch the WSL NeMo server."""
        self._logDebug("Preparing WSL server launch command...")
        self.wslLaunchCommand = []  # Ensure it's reset before trying

        wslServerUrl = self.config.get('wslServerUrl')
        wslDistro = self.config.get('wslDistributionName')

        if not wslServerUrl or not wslDistro:
            logError("Config error: 'wslServerUrl' & 'wslDistributionName' needed for WSL launch.")
            return

        try:
            # 1. Get Port
            parsedUrl = urlparse(wslServerUrl)
            wslServerPort = parsedUrl.port
            if not wslServerPort:
                raise ValueError(f"Could not extract port from wslServerUrl: {wslServerUrl}")

            # 2. Find WSL Server Script Path
            import __main__
            main_file_path = os.path.abspath(__main__.__file__)
            scriptDir = Path(os.path.dirname(main_file_path))
            wslServerScriptPathWindows = scriptDir / "wslNemoServer.py"
            logDebug(f"Checking primary path: {wslServerScriptPathWindows}")  # Debug path check

            if not wslServerScriptPathWindows.is_file():
                cwd = Path.cwd()
                fallbackPath = cwd / "wslNemoServer.py"
                logDebug(f"Checking fallback CWD path: {fallbackPath}")  # Debug path check
                if not fallbackPath.is_file():
                    raise FileNotFoundError(
                        f"WSL script 'wslNemoServer.py' not found in script dir ({scriptDir}) or CWD ({cwd}).")
                wslServerScriptPathWindows = fallbackPath
                logWarning(f"Using wslNemoServer.py from CWD: {wslServerScriptPathWindows}")
            else:
                logDebug(
                    f"Using wslNemoServer.py from script directory: {wslServerScriptPathWindows}")

            # 3. Convert Windows Path to WSL Path
            logDebug(
                f"Converting path for WSL: {wslServerScriptPathWindows}")  # Debug conversion input
            wslServerScriptPathWsl = convertWindowsPathToWsl(wslServerScriptPathWindows)
            if not wslServerScriptPathWsl:
                raise ValueError(
                    f"Failed to convert Windows path to WSL path: {wslServerScriptPathWindows}")
            logDebug(f"Converted path: {wslServerScriptPathWsl}")  # Debug conversion output

            # 4. Construct Command List
            pythonWslPath = "/usr/bin/python3"
            preparedCommand = [
                "wsl.exe", "-d", wslDistro, "--",
                pythonWslPath, wslServerScriptPathWsl,
                "--model_name", modelName, "--port", str(wslServerPort), "--load_on_start"
            ]

            # 5. Assign if successful
            self.wslLaunchCommand = preparedCommand
            logInfo("Prepared WSL server launch command successfully.")
            self._logDebug(f"WSL Command: {' '.join(self.wslLaunchCommand)}")

        # === MODIFIED EXCEPTION LOGGING ===
        except FileNotFoundError as e:
            logError(f"Error preparing WSL launch (FileNotFound): {e}")  # Log specific error
            logError("WSL server will NOT be launched automatically.")
        except ValueError as e:
            logError(f"Error preparing WSL launch (ValueError): {e}")  # Log specific error
            logError("WSL server will NOT be launched automatically.")
        except Exception as e:
            # Catch any other unexpected error during preparation
            logError(
                f"Unexpected error preparing WSL launch command: {type(e).__name__} - {e}")  # Log type and message
            logError(traceback.format_exc())  # Log full traceback
            logError("WSL server will NOT be launched automatically.")

    def _printInitialInstructions(self):
        """Prints setup info and user instructions based on configuration."""
        # Ensure handler is initialized before printing info
        if not self.asrModelHandler:
            logWarning("Cannot print initial instructions: ASR Handler not initialized.")
            return

        deviceStr = self.asrModelHandler.getDevice()  # Get device info from handler
        handlerType = type(self.asrModelHandler).__name__

        print(f"\n--- Configuration ---")
        print(f"Mode: {self.config.get('transcriptionMode')}")
        print(
            f"ASR Model: {self.config.get('modelName')} (Using Handler: {handlerType}, Target Device: {deviceStr})")
        if handlerType == 'RemoteNemoClientHandler':
            print(f"  WSL Server URL: {self.config.get('wslServerUrl', 'Not Set!')}")
        print(
            f"Audio Device: ID={self.config.get('deviceId') or 'Default'}, Rate={self.config.get('actualSampleRate')}Hz, Channels={self.config.get('actualChannels')}")
        print(f"--- Hotkeys ---")
        print(f"Toggle Recording: '{self.config.get('recordingToggleKey')}'")
        print(
            f"Toggle Text Output: '{self.config.get('outputToggleKey')}' (Method: {self.systemInteractionHandler.textOutputMethod})")
        print(f"--- Timeouts ---")
        print(f"Max Recording Duration: {self.config.get('maxDurationRecording')} s")
        print(f"Stop Recording After Silence: {self.config.get('consecutiveIdleTime')} s")
        print(f"Unload Model After Inactivity: {self.config.get('modelUnloadTimeout')} s")
        print(f"Program Auto-Exit After: {self.config.get('maxDurationProgramActive')} s")
        print(f"------------------\n")

    # --- Method to launch WSL Server ---
    def _launchWslServer(self):
        """Launches the wslNemoServer.py script in WSL using subprocess."""
        if not self.wslLaunchCommand:
            logWarning("WSL server launch command not prepared. Skipping automatic launch.")
            return False

        if self.wslServerProcess and self.wslServerProcess.poll() is None:
            logInfo("WSL server process appears to be already running.")
            return True  # Assume it's okay

        logInfo(f"Attempting to launch WSL server...")
        self._logDebug(f"Executing: {' '.join(self.wslLaunchCommand)}")
        try:
            # Use Popen to run in the background
            # Capture stdout/stderr to prevent blocking and optionally log later
            # Use CREATE_NO_WINDOW on Windows to hide the WSL console window
            creationflags = 0
            # Use the imported platform module here
            if platform.system() == "Windows":
                creationflags = subprocess.CREATE_NO_WINDOW

            self.wslServerProcess = subprocess.Popen(
                self.wslLaunchCommand,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,  # Decode stdout/stderr as text
                encoding='utf-8',  # Specify encoding
                errors='replace',  # Handle potential decoding errors
                creationflags=creationflags
            )
            logInfo(
                f"WSL server process launched (PID: {self.wslServerProcess.pid if hasattr(self.wslServerProcess, 'pid') else 'N/A'}). Allowing time to start...")
            # Give the server a few seconds to initialize
            time.sleep(5.0)  # Adjust as needed

            # Check if the process exited immediately (sign of an error)
            # Ensure process handle is valid before polling
            if self.wslServerProcess and self.wslServerProcess.poll() is not None:
                stderrOutput = "Could not read stderr."
                try:
                    # Read stderr non-blockingly or with timeout if possible
                    # Reading might block if the process generated a lot of output quickly
                    # For robustness, maybe read in a separate thread or use non-blocking reads if essential
                    stderrOutput = self.wslServerProcess.stderr.read() if self.wslServerProcess.stderr else "stderr not available"
                except Exception as stderr_e:
                    logWarning(f"Exception reading WSL stderr: {stderr_e}")
                    pass
                logError(
                    f"WSL server process exited immediately after launch (return code: {self.wslServerProcess.returncode}). Check WSL environment/script.")
                if stderrOutput:
                    logError(f"WSL Server stderr hints: {stderrOutput.strip()}")
                self.wslServerProcess = None
                return False
            elif self.wslServerProcess:
                # Check initial output from server for confirmation
                try:
                    stdout_line = self.wslServerProcess.stdout.readline() if self.wslServerProcess.stdout else ""
                    if stdout_line:
                        logInfo(f"Initial WSL Server stdout: {stdout_line.strip()}")
                        # You could check here if the line indicates successful startup
                    # else check stderr similarly
                    stderr_line = self.wslServerProcess.stderr.readline() if self.wslServerProcess.stderr else ""
                    if stderr_line:
                        logWarning(f"Initial WSL Server stderr: {stderr_line.strip()}")
                        # If stderr has output immediately, it might indicate a problem
                except Exception as std_e:
                    logWarning(f"Exception checking initial WSL stdout/stderr: {std_e}")

                logInfo("WSL server appears to be running (process exists).")
                return True  # Successfully launched (we assume)
            else:
                # Should not happen if Popen succeeded but good to handle
                logError("WSL server process object is unexpectedly None after launch attempt.")
                return False


        except FileNotFoundError:
            logError(f"Error launching WSL server: 'wsl.exe' not found in system PATH.")
            self.wslServerProcess = None
            return False
        except Exception as e:
            logError(f"Error launching WSL server process: {e}")
            logError(traceback.format_exc())
            self.wslServerProcess = None
            return False

    # --- Background Thread Worker for Transcription ---
    def _transcriptionWorkerLoop(self):
        """
        Worker loop running in a separate thread.
        Waits for audio data on the queue, calls the ASR handler's transcribe method
        (which handles local/remote), and processes the result via the output handler.
        """
        handlerType = type(self.asrModelHandler).__name__
        logInfo(f"Starting Transcription Worker thread (Handler: {handlerType}).")
        queueTimeoutSeconds = 1.0  # How long to wait for an item before checking program status

        while self.stateManager.shouldProgramContinue():
            try:
                # Wait for audio data with a timeout
                queueItem = self.transcriptionRequestQueue.get(timeout=queueTimeoutSeconds)

                if queueItem is None:  # Sentinel value to exit loop
                    self._logDebug("Transcription worker received None sentinel, exiting.")
                    break

                # Unpack the data
                audioDataToTranscribe, sampleRate = queueItem

                # Check if model is considered ready by the handler before attempting
                # For remote, this checks the last known server status
                if self.asrModelHandler.isModelLoaded():
                    self._logDebug(
                        f"Worker processing {len(audioDataToTranscribe)} samples ({len(audioDataToTranscribe) / sampleRate:.2f}s)...")
                    # --- ASR Inference (Delegated to Handler: Local or Remote) ---
                    transcriptionResult = self.asrModelHandler.transcribeAudioSegment(
                        audioDataToTranscribe,
                        sampleRate
                    )
                    # --- Result Handling (via Output Handler) ---
                    # Pass audio data along for loudness checks if available
                    self.outputHandler.processTranscriptionResult(
                        transcriptionResult,
                        audioDataToTranscribe
                    )
                    # self._logDebug("Transcription worker finished processing segment.")
                else:
                    # This might happen if the model failed to load initially,
                    # or if the remote server became unreachable.
                    logWarning(
                        "Transcription worker skipped segment: ASR model is not loaded/ready.")
                    # Reset processor state if needed, especially for dictation mode buffer clearing
                    if self.config.get('transcriptionMode') == 'dictationMode':
                        # If trigger occurred but transcription skipped, ensure buffer is cleared
                        # Note: RealTimeAudioProcessor might need refined logic for this edge case
                        # self.realTimeProcessor.clearBuffer() # Be careful not to clear valid ongoing audio
                        self._logDebug(
                            "Dictation mode state might need reset by processor due to skipped transcription.")

                self.transcriptionRequestQueue.task_done()  # Signal queue item processed

            except queue.Empty:
                # Queue was empty during timeout, loop continues to check shouldProgramContinue
                continue
            except Exception as e:
                logError(f"!!! ERROR in Transcription Worker thread: {e}")
                logError(traceback.format_exc())
                # Avoid busy-looping on errors
                time.sleep(1)

        logInfo("Transcription Worker thread stopping.")

    def _startBackgroundThreads(self):
        """Starts threads for hotkeys, model management, and transcription with better error reporting."""
        self._logDebug("Starting background threads...")
        self.threads = []  # Clear list
        failedThreads = []

        # Function to wrap thread target for exception logging
        def threadWrapper(targetFunc, *args, **kwargs):
            threadName = threading.current_thread().name
            try:
                targetFunc(*args, **kwargs)
            except Exception as e:
                logError(f"!!! UNCAUGHT EXCEPTION in thread '{threadName}': {e}")
                logError(traceback.format_exc())
                # Optionally signal program stop if a critical thread dies
                if threadName in ["KeyboardMonitorThread", "TranscriptionWorkerThread"]:
                    logError(f"Critical thread {threadName} failed, signaling program stop.")
                    self.stateManager.stopProgram()

        # 1. Keyboard Monitor Thread
        target = self.systemInteractionHandler.monitorKeyboardShortcuts
        threadName = "KeyboardMonitorThread"
        try:
            keyboardThread = threading.Thread(
                target=threadWrapper,  # Use the wrapper
                args=(target, self),  # Pass original target and args to wrapper
                name=threadName,
                daemon=True
            )
            self.threads.append(keyboardThread)
            keyboardThread.start()
        except Exception as e:
            logError(f"Failed to create/start {threadName}: {e}")
            failedThreads.append(threadName)

        # 2. Model Lifecycle Manager Thread
        target = self.modelLifecycleManager.manageModelLifecycle
        threadName = "ModelManagerThread"
        try:
            modelThread = threading.Thread(
                target=threadWrapper,  # Use the wrapper
                args=(target,),  # Pass original target and args to wrapper
                name=threadName,
                daemon=True
            )
            self.threads.append(modelThread)
            modelThread.start()
        except Exception as e:
            logError(f"Failed to create/start {threadName}: {e}")
            failedThreads.append(threadName)

        # 3. Transcription Worker Thread
        target = self._transcriptionWorkerLoop
        threadName = "TranscriptionWorkerThread"
        try:
            transcriptionThread = threading.Thread(
                target=threadWrapper,  # Use the wrapper
                args=(target,),  # Pass original target and args to wrapper
                name=threadName,
                daemon=True
            )
            self.threads.append(transcriptionThread)
            transcriptionThread.start()
        except Exception as e:
            logError(f"Failed to create/start {threadName}: {e}")
            failedThreads.append(threadName)

        # Check status after attempting starts
        time.sleep(0.1)  # Give threads a moment to potentially fail immediately
        activeThreadCount = sum(1 for t in self.threads if t.is_alive())
        expectedThreadCount = len(self.threads)

        self._logDebug(f"Attempted to start {expectedThreadCount} background threads.")
        if failedThreads:
            logError(f"Failed to initiate threads: {', '.join(failedThreads)}")
        if activeThreadCount < (expectedThreadCount - len(failedThreads)):
            logWarning(
                "One or more threads started but exited immediately. Check logs above for errors (especially permissions for keyboard).")
        elif activeThreadCount == expectedThreadCount:
            self._logDebug("All background threads appear to be running.")

        if "KeyboardMonitorThread" in failedThreads or not any(
                t.name == "KeyboardMonitorThread" and t.is_alive() for t in self.threads):
            logWarning("Keyboard monitor thread failed to start or died. Hotkeys will NOT work.")
            logError("--> If on Windows, try running the script as Administrator.")
            logError(
                "--> If on Linux, ensure your user is in the 'input' group or run with 'sudo'.")
        if "TranscriptionWorkerThread" in failedThreads or not any(
                t.name == "TranscriptionWorkerThread" and t.is_alive() for t in self.threads):
            logWarning(
                "Transcription worker thread failed to start or died. Transcription will NOT occur.")

    # --- Public Methods for Hotkey Actions ---
    def toggleRecording(self):
        """Toggles the recording state. Called by systemInteractionHandler."""
        if self.stateManager.isRecording():
            # Stop recording
            if self.stateManager.stopRecording():  # Returns True if state changed
                self.systemInteractionHandler.playNotification("recordingOff")
                logInfo("Recording stopped.")
                # Clear buffers immediately on manual stop
                self.realTimeProcessor.clearBuffer()
                self.audioHandler.clearQueue()
                # Ensure stream stops (handled by main loop's lifecycle management)
        else:
            # Start recording
            if self.stateManager.startRecording():  # Returns True if state changed
                self.systemInteractionHandler.playNotification("recordingOn")
                logInfo("Recording started.")
                # Ensure model is loaded and stream starts (handled by lifecycle managers)

    def toggleOutput(self):
        """Toggles the text output state. Called by systemInteractionHandler."""
        newState = self.stateManager.toggleOutput()
        notification = "outputEnabled" if newState else "outputDisabled"
        self.systemInteractionHandler.playNotification(notification)
        # Optional: If output just got disabled, clear the processor buffer
        if not newState:
            self.realTimeProcessor.clearBufferIfOutputDisabled()

    # ---- Cleanup ---
    def _cleanup(self):
        """Cleans up all resources and attempts to join threads and terminate WSL server."""
        logInfo("Initiating cleanup...")
        # 1. Signal all loops to stop
        self.stateManager.stopProgram()

        # --- Terminate WSL Server Process (if launched and still exists) ---
        if self.wslServerProcess:
            # Check if it's still running before trying to terminate
            if self.wslServerProcess.poll() is None:
                logInfo(
                    f"Attempting to terminate launched WSL server process (PID: {self.wslServerProcess.pid if hasattr(self.wslServerProcess, 'pid') else 'N/A'})...")
                try:
                    # Try terminate first (allows graceful shutdown)
                    self.wslServerProcess.terminate()
                    # Wait briefly for termination and capture output
                    try:
                        stdout, stderr = self.wslServerProcess.communicate(timeout=2.0)
                        logInfo("WSL server process terminated.")
                        # Log output only if it contains something significant
                        if stdout and stdout.strip(): self._logDebug(
                            f"WSL Server stdout on exit: {stdout.strip()}")
                        if stderr and stderr.strip(): logWarning(
                            f"WSL Server stderr on exit: {stderr.strip()}")
                    except subprocess.TimeoutExpired:
                        logWarning(
                            "WSL server did not terminate gracefully within timeout, attempting to kill...")
                        self.wslServerProcess.kill()  # Force kill if terminate timed out
                        # Give kill a moment
                        time.sleep(0.5)
                        logInfo("WSL server process killed.")
                    except ValueError:
                        logWarning("WSL process streams closed before communicate could finish.")
                    except Exception as comm_e:
                        logWarning(f"Error during WSL process communicate: {comm_e}")

                except ProcessLookupError:
                    logInfo(
                        "WSL server process already finished before termination attempt.")  # Process might have already exited
                except Exception as e:
                    logError(f"Error terminating WSL server process: {e}")
                    # Ensure handle is cleared even if termination fails
                    self.wslServerProcess = None
            else:
                logInfo("Launched WSL server process was already finished.")
            # Clear the handle after attempting termination/kill
            self.wslServerProcess = None
        # --- End WSL Server Termination ---

        # 2. Stop audio stream explicitly (if running)
        logInfo("Stopping audio stream...")
        self.audioHandler.stopStream()

        # 3. Clear processing buffers/queues
        logInfo("Clearing audio buffers and queues...")
        self.realTimeProcessor.clearBuffer()
        self.audioHandler.clearQueue()

        # 4. Signal transcription worker to finish
        try:
            self.transcriptionRequestQueue.put(None, timeout=0.5)
        except queue.Full:
            logWarning(
                "Transcription queue full during cleanup, worker might not receive sentinel.")
        except Exception as e:
            logWarning(f"Error putting sentinel on transcription queue: {e}")

        # 5. Give threads a moment to react
        logInfo("Waiting briefly for background threads to react...")
        time.sleep(1.0)

        # 6. Cleanup ASR Handler (local unload or remote signal)
        if self.asrModelHandler:
            logInfo("Cleaning up ASR model handler...")
            try:
                # Cleanup might try to send /unload to the server we just terminated
                # This is okay, _makeServerRequest handles connection errors gracefully
                self.asrModelHandler.cleanup()
            except Exception as e:
                logError(f"Error during ASR handler cleanup: {e}")

        # 7. Cleanup System Interaction Handler
        logInfo("Cleaning up system interaction handler...")
        try:
            self.systemInteractionHandler.cleanup()
        except Exception as e:
            logError(f"Error during system interaction handler cleanup: {e}")

        # 8. Attempt to join background threads
        logInfo("Attempting to join background threads...")
        joinTimeout = 2.0
        # Make a copy of the list in case a thread removes itself (unlikely with daemons)
        threads_to_join = list(self.threads)
        self.threads = []  # Clear original list
        for t in threads_to_join:
            if t is not None and t.is_alive():
                threadName = t.name or "Unknown Thread"
                self._logDebug(f"Joining thread {threadName}...")
                try:
                    t.join(timeout=joinTimeout)
                    if t.is_alive():
                        logWarning(
                            f"Thread '{threadName}' did not terminate within {joinTimeout}s.")
                except Exception as e:
                    logWarning(f"Error joining thread {threadName}: {e}")
            # else:
            #    self._logDebug(f"Thread {t.name if t else 'N/A'} was None or not alive.")

        logInfo("Program cleanup complete.")

    # ---- Main Loop Sub-methods for Clarity ----
    def _run_initialSetup(self):
        """Handle initial setup: Launch WSL server if needed, load model, start threads & audio stream."""

        # --- Launch WSL Server If Required ---
        serverLaunchedOk = True  # Assume okay if not needed
        # Check the type of the *instantiated* handler
        if isinstance(self.asrModelHandler, RemoteNemoClientHandler):
            serverLaunchedOk = self._launchWslServer()
            if not serverLaunchedOk:
                logError(
                    "CRITICAL: Failed to launch required WSL NeMo server. Remote transcription will fail.")
                # Decide if you want to stop the whole app if server fails to launch
                # Option 1: Continue (remote handler will fail on requests)
                # Option 2: Stop program immediately
                # self.stateManager.stopProgram()
                # return # Early exit from setup if critical error
        # --- End WSL Server Launch ---

        # Initial Model Load Check/Attempt (if configured to start recording)
        # Proceed even if server launch failed; loadModel() in client handler will handle connection error
        if self.stateManager.isRecording():
            logInfo("Initial state is recording: ensuring model is loaded/ready...")
            try:
                self.asrModelHandler.loadModel()  # Handles local or remote load attempt
                if not self.asrModelHandler.isModelLoaded():
                    # Log details based on handler type
                    handlerType = type(self.asrModelHandler).__name__
                    if handlerType == 'RemoteNemoClientHandler':
                        logError(
                            f"CRITICAL: Failed to connect to or load model on remote WSL server ({self.config.get('wslServerUrl')}). Recording disabled.")
                    else:  # Local handler
                        logError("CRITICAL: Initial local model load failed. Recording disabled.")
                    self.stateManager.stopRecording()
            except Exception as e:
                logError(
                    f"CRITICAL: Error during initial model load/check: {e}. Recording disabled.")
                logError(traceback.format_exc())
                self.stateManager.stopRecording()

        # Start Background Threads
        self._startBackgroundThreads()

        # Initial Audio Stream Start (only if still set to record after potential load failures)
        initialStreamStarted = False
        if self.stateManager.isRecording():
            logInfo("Initial state is recording: attempting to start audio stream...")
            initialStreamStarted = self.audioHandler.startStream()
            if not initialStreamStarted:
                logError("CRITICAL: Failed to start audio stream initially. Recording disabled.")
                self.stateManager.stopRecording()

    def _run_checkTimeoutsNGlobalState(self):
        """Check program/recording timeouts, manage global state like clearing buffer if output disabled."""
        # --- Call to the method that failed ---
        if self.stateManager.checkProgramTimeout():
            logInfo("Maximum program duration reached.")
            return False  # Signal loop to exit

        # Clear audio buffer if output is disabled to prevent backlog
        self.realTimeProcessor.clearBufferIfOutputDisabled()

        # Check recording-specific timeouts only if recording is active
        if self.stateManager.isRecording():
            # --- Call to the other timeout checks ---
            if self.stateManager.checkRecordingTimeout():
                logInfo("Recording session timeout reached, stopping recording...")
                self.toggleRecording()  # Use toggle method to handle state and sound
            elif self.stateManager.checkIdleTimeout():
                logInfo("Idle timeout reached, stopping recording...")
                self.toggleRecording()

        return True  # Signal loop to continue

    def _run_manageAudioStreamLifecycle(self):
        """Start/stop audio stream based on recording state."""
        shouldStreamBeActive = self.stateManager.isRecording()
        isStreamActuallyActive = self.audioHandler.stream is not None and self.audioHandler.stream.active

        if shouldStreamBeActive and not isStreamActuallyActive:
            self._logDebug("Attempting to start audio stream (recording active)...")
            if not self.audioHandler.startStream():
                logError("Failed to start/restart audio stream. Disabling recording.")
                self.stateManager.stopRecording()
                time.sleep(1)  # Pause before next attempt
                return False  # Indicate failure that might need handling

        elif not shouldStreamBeActive and isStreamActuallyActive:
            self._logDebug("Stopping audio stream (recording inactive)...")
            self.audioHandler.stopStream()
            # Also clear buffers when stream stops
            self.realTimeProcessor.clearBuffer()
            self.audioHandler.clearQueue()

        return True  # Indicate success or no action needed

    def _run_processAudioChunks(self):
        """Dequeue and process audio chunks from the audio handler queue."""
        audioProcessedThisLoop = False
        # Only process if recording is active AND stream seems okay
        if self.stateManager.isRecording() and (
                self.audioHandler.stream is None or self.audioHandler.stream.active):
            # Process all available chunks in the queue
            while True:
                chunk = self.audioHandler.getAudioChunk()
                if chunk is None:
                    break  # No more chunks in queue
                if self.realTimeProcessor.processIncomingChunk(chunk):
                    audioProcessedThisLoop = True  # Mark if any chunk was added to buffer

        return audioProcessedThisLoop

    def _run_queueTranscriptionRequest(self, audioProcessedThisLoop):
        """Checks if transcription should be triggered and queues the request for the worker thread."""
        # Only queue if:
        # 1. Output is enabled
        # 2. Audio was potentially added to the buffer this loop (optimization)
        # 3. Trigger conditions are met
        if self.stateManager.isOutputEnabled() and audioProcessedThisLoop:
            # Ask the processor if conditions are met to transcribe
            audioDataToTranscribe = self.realTimeProcessor.checkTranscriptionTrigger()

            if audioDataToTranscribe is not None:
                # Send data to the background thread via the queue
                try:
                    sampleRate = self.config.get('actualSampleRate')
                    queueItem = (audioDataToTranscribe, sampleRate)
                    # Use non-blocking put with timeout to avoid deadlocks if worker dies
                    self.transcriptionRequestQueue.put(queueItem, timeout=0.5)
                    self._logDebug(
                        f"Queued transcription request for {len(audioDataToTranscribe)} samples.")
                except queue.Full:
                    logWarning("Transcription request queue is full. Skipping current segment.")
                    # Clear buffer for dictation mode if trigger happened but queue is full
                    if self.config.get('transcriptionMode') == 'dictationMode':
                        self.realTimeProcessor.clearBuffer()
                        self.realTimeProcessor.isCurrentlySpeaking = False  # Reset dictation state
                        self.realTimeProcessor.silenceStartTime = None
                except Exception as e:
                    logError(f"Failed to queue transcription request: {e}")

    def _run_loopSleep(self):
        """Sleep briefly to prevent high CPU usage, especially if idle."""
        time.sleep(0.01)  # Small sleep in main loop

    # ========================================
    # ==         MAIN EXECUTION LOOP        ==
    # ========================================
    def run(self):
        """Main execution loop orchestrating the real-time transcription process."""
        logInfo("Starting main orchestrator loop...")
        try:
            self._run_initialSetup()

            # Check if initial setup failed critically (e.g., no handler)
            if not self.asrModelHandler:
                logError("Orchestrator cannot run: ASR Handler initialization failed.")
                return  # Exit early

            while self.stateManager.shouldProgramContinue():
                # 1. Check Timeouts & Global State (like clearing buffer if output off)
                if not self._run_checkTimeoutsNGlobalState():
                    break  # Exit loop if program timeout reached

                # 2. Manage Audio Stream (Start/Stop based on recording state)
                if not self._run_manageAudioStreamLifecycle():
                    # If stream management failed critically, maybe pause or stop
                    time.sleep(1)  # Pause before retrying state checks
                    continue

                # 3. Process Incoming Audio Chunks (Fill buffer)
                audioProcessedThisLoop = self._run_processAudioChunks()

                # 4. Check Trigger & Queue Transcription Request (Send buffer to worker)
                self._run_queueTranscriptionRequest(audioProcessedThisLoop)

                # 5. Short sleep to yield CPU
                self._run_loopSleep()

        except KeyboardInterrupt:
            logInfo("\nKeyboardInterrupt received. Stopping...")
            self.stateManager.stopProgram()  # Ensure cleanup runs
        except Exception as e:
            logError(f"\n!!! UNEXPECTED ERROR in main orchestrator loop: {e}")
            logError(traceback.format_exc())
            self.stateManager.stopProgram()  # Ensure cleanup runs
        finally:
            self._cleanup()  # Essential cleanup routine
