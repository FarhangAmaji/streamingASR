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
# Import new logging configuration function and helpers
from utils import logInfo, logError, logCritical, configure_dynamic_logging
from define_logConfigSets import defaultLogConfigSets  # Import the default log sets

# --- Configure Logging FIRST ---
# For useFileTranscriber, we can use a simpler logging setup or a specific
# configSet from defaultLogConfigSets.
# Here, we'll use the defaultLogConfigSets.
# The loggerName can be customized if needed.
configure_dynamic_logging(
    loggerName="FileTranscriberApp",
    logConfigSets=defaultLogConfigSets,  # Use the imported default sets
    highOrderOptions={}  # No high-order options for this example script by default
)

# ==================================
# Example Configuration & Execution
# ==================================
if __name__ == "__main__":
    # --- Configuration for File Transcription ---
    # Adapt these settings as needed
    userSettingsExample = {
        # --- Core Model Settings ---
        "modelName": "openai/whisper-tiny.en",  # Example: Local Whisper (smaller for quick test)
        # "modelName": "nvidia/parakeet-rnnt-1.1b", # Example: Remote NeMo
        "language": "en",
        "CPU": False,  # Set to True to force CPU for local Whisper
        # --- Remote Server Settings (Only needed for 'nvidia/' models) ---
        "wslServerUrl": "http://localhost:5001",
        "serverRequestTimeout": 60.0,  # Allow more time for potentially long files
        "unloadRemoteModelOnExit": False,  # Keep server model loaded after script finishes?
        # --- File Transcription Specific ---
        "removeTrailingDots": True,
        # --- Other settings (less relevant for file transcription but needed by ConfigManager) ---
        # scriptDir is now set within ConfigurationManager using DynamicLogger's basePath implicitly if DynamicLogger sets it
        # Or, if DynamicLogger is instantiated in utils.py, its basePath is derived from utils.py's location.
        # For simplicity, if ConfigurationManager needs it explicitly, it can be set.
        # "scriptDir": Path(os.path.dirname(os.path.abspath(__file__))) # This script's dir
    }

    # --- File Paths ---
    # !!! CHANGE THIS to your audio file !!!
    # Create a dummy WAV file for testing if one doesn't exist.
    dummyAudioFilePath = Path(os.path.dirname(os.path.abspath(__file__))) / "test_audio_sample.wav"
    if not dummyAudioFilePath.exists():
        try:
            import soundfile as sf
            import numpy as np

            sampleRate = 16000
            duration = 2  # seconds
            frequency = 440  # Hz
            t = np.linspace(0, duration, int(sampleRate * duration), False)
            note = np.sin(frequency * t * 2 * np.pi)
            # Ensure data is in float32, as expected by handlers
            audio_data_normalized = note.astype(np.float32)
            sf.write(dummyAudioFilePath, audio_data_normalized, sampleRate)
            logInfo(f"Created dummy audio file for testing: {dummyAudioFilePath}")
            audioFilePathToTranscribe = str(dummyAudioFilePath)
        except Exception as e_dummy:
            logError(
                f"Could not create dummy audio file: {e_dummy}. Please set audioFilePathToTranscribe manually.")
            audioFilePathToTranscribe = "path/to/your/audiofile.wav"  # Placeholder if dummy creation fails
    else:
        audioFilePathToTranscribe = str(dummyAudioFilePath)
        logInfo(f"Using existing test audio file: {audioFilePathToTranscribe}")

    # Optional: Set a path to save the text output
    outputDir = Path(os.path.dirname(os.path.abspath(__file__))) / "output"
    outputDir.mkdir(parents=True, exist_ok=True)
    outputTranscriptionPath = str(
        outputDir / "file_transcription_result.txt")  # e.g., None to print to console

    logInfo("--- File Transcription Example ---")
    logInfo(f"Input Audio File: {audioFilePathToTranscribe}")
    logInfo(f"Output File: {outputTranscriptionPath or 'Console'}")

    # --- Setup ---
    asrHandler = None
    # fileTranscriberInstance = None # Defined later

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
            logCritical(
                "No 'modelName' specified in configuration. Cannot proceed.")  # Critical error
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
            f"Input audio file not found at '{audioFilePathToTranscribe}'. Please check the path.",
            exc_info=True)
    except ValueError as e:  # Configuration or other value errors
        logCritical(f"Configuration or Value error: {e}", exc_info=True)  # Critical for setup
    except Exception as e:
        logCritical(f"An unexpected error occurred: {e}", exc_info=True)
        logError(traceback.format_exc())  # Keep explicit traceback for unexpected errors
    finally:
        # --- Cleanup ---
        # Clean up the ASR handler (this might unload the local model or signal the server)
        if asrHandler:
            logInfo("Cleaning up ASR handler...")
            try:
                asrHandler.cleanup()
            except Exception as e:
                logError(f"Error during ASR handler cleanup: {e}", exc_info=True)
        logInfo("File transcription example finished.")
