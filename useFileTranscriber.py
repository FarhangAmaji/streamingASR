# useFileTranscriber.py
# ==============================================================================
# Example: Transcribing an Audio File
# ==============================================================================
#
# Purpose:
# - Demonstrates how to use the `FileTranscriber` class from
#   `tasks.py` to transcribe a pre-recorded audio file.
# - Shows how to configure the appropriate ASR model handler (local Whisper
#   or the remote NeMo client) based on the desired model.
#
# Usage:
# - Modify the `userSettingsExample` dictionary with your desired model,
#   WSL server URL (if using NeMo), etc.
# - Set the `audioFilePathToTranscribe` and optionally `outputTranscriptionPath`.
# - If using an 'nvidia/' model, ensure the `wslNemoServer.py` script is
#   running in your WSL environment first.
# - Run this script: `python useFileTranscriber.py`
#
# Dependencies:
# - Requires access to `managers.py`, `modelHandlers.py`, `tasks.py`, `utils.py`.
# - Dependencies listed in those files are also required here.
# ==============================================================================
import os
import time
import traceback
from pathlib import Path

from managers import ConfigurationManager
from modelHandlers import WhisperModelHandler, RemoteNemoClientHandler
from tasks import FileTranscriber
from utils import logInfo, logError, configureLogging  # Added configureLogging

# --- Configure Logging FIRST ---
# Determine log file path (e.g., in the script's directory)
logDir = Path(os.path.dirname(os.path.abspath(__file__))) / "logs"
logDir.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists
logFilePath = logDir / "file_transcription.log"
# Set debugMode=True for verbose logs during troubleshooting
configureLogging(logFileName=logFilePath, debugMode=True)  # Set debug=True for detailed logs

# ==================================
# Example Configuration & Execution
# ==================================
if __name__ == "__main__":
    # --- Configuration for File Transcription ---
    # Adapt these settings as needed
    userSettingsExample = {
        # --- Core Model Settings ---
        "modelName": "openai/whisper-large-v3",  # Example: Local Whisper
        # "modelName": "nvidia/parakeet-rnnt-1.1b", # Example: Remote NeMo
        "language": "en",
        "CPU": False,
        # --- Remote Server Settings (Only needed for 'nvidia/' models) ---
        "wslServerUrl": "http://localhost:5001",
        "serverRequestTimeout": 60.0,  # Allow more time for potentially long files
        "unloadRemoteModelOnExit": False,  # Keep server model loaded after script finishes?
        # --- File Transcription Specific ---
        "removeTrailingDots": True,
        # --- Debugging ---
        "debugPrint": False,  # Set to True for verbose logging from handlers
        # --- Other settings (less relevant for file transcription but needed by ConfigManager) ---
        "scriptDir": Path(os.path.dirname(os.path.abspath(__file__)))  # Use this script's dir
    }
    # --- File Paths ---
    # !!! CHANGE THIS to your audio file !!!
    audioFilePathToTranscribe = "path/to/your/audiofile.wav"  # e.g., "test_audio.wav"
    # Optional: Set a path to save the text output
    outputTranscriptionPath = "output/transcription_result.txt"  # e.g., None to print to console
    logInfo("--- File Transcription Example ---")
    logInfo(f"Input Audio File: {audioFilePathToTranscribe}")
    logInfo(f"Output File: {outputTranscriptionPath or 'Console'}")
    # --- Setup ---
    asrHandler = None
    fileTranscriberInstance = None
    try:
        config = ConfigurationManager(**userSettingsExample)
        modelName = config.get('modelName', '')
        # --- Choose and Instantiate ASR Handler ---
        if modelName.lower().startswith("nvidia/"):
            logInfo(f"Using RemoteNemoClientHandler for model: {modelName}")
            logInfo(f"Ensure WSL server is running at: {config.get('wslServerUrl')}")
            asrHandler = RemoteNemoClientHandler(config)
        elif modelName:
            logInfo(f"Using local WhisperModelHandler for model: {modelName}")
            asrHandler = WhisperModelHandler(config)
        else:
            raise ValueError("No 'modelName' specified in configuration.")
        # --- Instantiate File Transcriber ---
        fileTranscriberInstance = FileTranscriber(config, asrHandler)
        # --- Run Transcription ---
        logInfo("Starting file transcription process...")
        startTime = time.time()
        resultText = fileTranscriberInstance.transcribeFile(
            audioFilePathToTranscribe,
            outputTranscriptionPath
        )
        endTime = time.time()
        if resultText is not None:
            logInfo(
                f"File transcription completed successfully in {endTime - startTime:.2f} seconds.")
            # Result already printed or saved by _handleOutput
        else:
            logError(f"File transcription failed after {endTime - startTime:.2f} seconds.")
    except FileNotFoundError:
        logError(
            f"Input audio file not found at '{audioFilePathToTranscribe}'. Please check the path.")
    except ValueError as e:
        logError(f"Configuration error: {e}")
    except Exception as e:
        logError(f"An unexpected error occurred: {e}")
        logError(traceback.format_exc())
    finally:
        # --- Cleanup ---
        # Clean up the ASR handler (this might unload the local model or signal the server)
        if asrHandler:
            logInfo("Cleaning up ASR handler...")
            try:
                asrHandler.cleanup()
            except Exception as e:
                logError(f"Error during ASR handler cleanup: {e}")
        logInfo("File transcription example finished.")
