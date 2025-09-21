# useFileTranscriber.py
# ==============================================================================
# Example: Transcribing an Audio File
# ==============================================================================
#
# Purpose:
# - Demonstrates how to use the `FileTranscriber` class to transcribe a pre-recorded audio file.
# - Shows how to configure the appropriate ASR model handler (local Whisper or remote NeMo).
#
# Usage:
# - Modify the `userSettingsExample` dictionary with your desired model and settings.
# - Set the `audioFilePathToTranscribe`.
# - If using an 'nvidia/' model, ensure the `wslNemoServer.py` script is running.
# - Run this script: `python useFileTranscriber.py`
# ==============================================================================
import os
import time
import traceback
from pathlib import Path

from managers import ConfigurationManager
from modelHandlers import WhisperModelHandler
from wslServer.clientHandler import RemoteNemoClientHandler
from tasks import FileTranscriber
# Import the new universal logger instances
from utils.loggerInstance import uniLogger, uniDebugLogger

# ==================================
# Example Configuration & Execution
# ==================================
if __name__ == "__main__":
    # --- Configuration for File Transcription ---
    # Adapt these settings as needed for your model and environment.
    userSettingsExample = {
        # --- Core Model Settings ---
        "modelName": "openai/whisper-tiny.en",
        # "modelName": "nvidia/parakeet-rnnt-1.1b", # Example: Remote NeMo
        "language": "en",  # Language for transcription
        "CPU": False,  # Set to True to force CPU for local Whisper models
        # --- Remote Server Settings (Only for 'nvidia/' models) ---
        "wslServerUrl": "http://localhost:5001",
        "serverRequestTimeout": 60.0,  # Allow more time for long files
        "unloadRemoteModelOnExit": False,  # Keep server model loaded after script finishes?
        # --- File Transcription Specific ---
        "removeTrailingDots": True,
    }

    # --- File Paths ---
    # Define the path for a dummy audio file for testing purposes.
    dummyAudioFilePath = Path(os.path.dirname(os.path.abspath(__file__))) / "test_audio_sample.wav"
    # If the dummy file doesn't exist, create it.
    if not dummyAudioFilePath.exists():
        try:
            import soundfile as sf
            import numpy as np

            # Create a simple 2-second sine wave at 440Hz
            sampleRate, duration, frequency = 16000, 2, 440
            t = np.linspace(0, duration, int(sampleRate * duration), False)
            note = np.sin(frequency * t * 2 * np.pi).astype(np.float32)
            sf.write(dummyAudioFilePath, note, sampleRate)
            uniLogger.info(f"Created dummy audio file for testing: {dummyAudioFilePath}")
            audioFilePathToTranscribe = str(dummyAudioFilePath)
        except Exception as e_dummy:
            uniLogger.error(
                f"Could not create dummy audio file: {e_dummy}. Please set path manually.")
            audioFilePathToTranscribe = "path/to/your/audiofile.wav"  # Placeholder
    else:
        # Use the existing dummy file
        audioFilePathToTranscribe = str(dummyAudioFilePath)
        uniLogger.info(f"Using existing test audio file: {audioFilePathToTranscribe}")

    # Define and create the output directory for the transcription result.
    outputDir = Path(os.path.dirname(os.path.abspath(__file__))) / "output"
    outputDir.mkdir(parents=True, exist_ok=True)
    outputTranscriptionPath = str(outputDir / "file_transcription_result.txt")

    uniLogger.info("--- File Transcription Example ---")
    uniLogger.info(f"Input Audio File: {audioFilePathToTranscribe}")
    uniLogger.info(f"Output File: {outputTranscriptionPath or 'Console'}")

    # Initialize handler to None for cleanup in the finally block
    asrHandler = None
    try:
        # --- Setup ---
        config = ConfigurationManager(**userSettingsExample)
        modelName = config.get('modelName', '')

        # --- Choose and Instantiate ASR Handler based on modelName ---
        if modelName.lower().startswith("nvidia/"):
            uniLogger.info(f"Using RemoteNemoClientHandler for model: {modelName}")
            uniLogger.info(f"Ensure WSL server is running at: {config.get('wslServerUrl')}")
            asrHandler = RemoteNemoClientHandler(config)
        elif modelName:
            uniLogger.info(f"Using local WhisperModelHandler for model: {modelName}")
            asrHandler = WhisperModelHandler(config)
        else:
            # If no model name is provided, raise an error.
            raise ValueError("No 'modelName' specified in configuration.")

        # Instantiate the FileTranscriber with the config and chosen handler
        fileTranscriberInstance = FileTranscriber(config, asrHandler)

        # --- Run Transcription ---
        uniLogger.info("Starting file transcription process...")
        startTime = time.time()
        resultText = fileTranscriberInstance.transcribeFile(
            audioFilePathToTranscribe,
            outputTranscriptionPath
        )
        endTime = time.time()

        # --- Report Result ---
        if resultText is not None:
            uniLogger.info(
                f"File transcription completed successfully in {endTime - startTime:.2f} seconds.")
        else:
            uniLogger.error(f"File transcription failed after {endTime - startTime:.2f} seconds.")

    except Exception as e:
        # Catch any unexpected errors during the process
        uniLogger.critical(f"An unexpected error occurred: {e}", excInfo=True)
    finally:
        # --- Cleanup ---
        # Ensure the ASR handler is cleaned up properly (e.g., model unloaded)
        if asrHandler:
            uniLogger.info("Cleaning up ASR handler...")
            try:
                asrHandler.cleanup()
            except Exception as e:
                uniLogger.error(f"Error during ASR handler cleanup: {e}", excInfo=True)
        uniLogger.info("File transcription example finished.")
