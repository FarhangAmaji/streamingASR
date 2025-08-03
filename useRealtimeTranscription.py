# useRealtimeTranscription.py
# ccc1
#  Check or to_do:
#   - Being able to change their user settings while the program is still on That, especially can be helpful in the cases that I turn on and off music, or change the model, When I require more accuracy or require less VRAM used
#   - This one is probably much harder to implement, Have the output of Playing sounds I already used them From the sound sent to be transcribed. Should be Implemented For OS Specific
#   - check the times are correct, not sure but felt that models unloads earlier than model_unloadTimeout
#   - Add ask AI later
#   - debugging modes (only if later, I require extensive debugging again)
import os
import traceback
from pathlib import Path

from mainManager import SpeechToTextOrchestrator
# Import new logging configuration function and helpers
from utils import logInfo, logWarning, logError, logDebug, logCritical, configure_dynamic_logging
from define_logConfigSets import defaultLogConfigSets  # Import the default log sets

# --- Configure Logging FIRST ---
# For useRealtimeTranscription, we want to use a comprehensive logging setup.
# We'll pass the defaultLogConfigSets and an empty highOrderOptions dictionary.
# The loggerName "RealtimeASRApp" will be used.

# Define empty highOrderOptions as requested
# These can be populated later for dynamic log control without code changes.
highOrderOptions = {
    # Example (currently commented out):
    # "MyDemoClass.methodB": {
    #     "exclude": True,
    # },
    # "SENSITIVE_OPERATION_LOG": {
    #     "writeToFile": False,
    #     "inlineConfigSetName": "simpleConsoleOnly" # Example: force a specific simple console output
    # }
}

configure_dynamic_logging(
    loggerName="RealtimeASRApp",  # Specific name for this app
    logConfigSets=defaultLogConfigSets,
    highOrderOptions=highOrderOptions
)

# --- User Configuration ---
# Define configuration as a dictionary. Adjust these values as needed.
userSettings = {
    # --- Core Model Settings ---
    # Choose the ASR model. Examples:
    # "modelName": "openai/whisper-tiny.en",
    # Local Whisper (requires Transformers, smaller for quick test)
    #"modelName": "openai/whisper-large-v3",  # Local Whisper (requires Transformers)
    # "modelName": "openai/whisper-medium.en", # Local Whisper (English-only, smaller)
    "modelName": "nvidia/canary-180m-flash",  # Example: Remote NeMo (requires WSL server)
    # "modelName": "nvidia/parakeet-rnnt-1.1b", # Example: Remote NeMo (larger, requires WSL server)
    # Target language for transcription.
    # For Whisper, set to None to enable auto-detection (can be slower).
    # For NeMo multilingual models (like Canary), specify the expected language code (e.g., 'en', 'es').
    "language": "en",
    # Force CPU usage for local models
    # If True, overrides automatic GPU detection for Whisper. Ignored by remote NeMo handler.
    "CPU": False,

    # --- Remote Server Settings (ONLY used if modelName starts with 'nvidia/') ---
    # URL where the wslNemoServer.py script listens inside WSL.
    "wslServerUrl": "http://localhost:5001",
    # The exact name of your WSL distribution where the NeMo server will run.
    # Check available distributions by running `wsl -l` in Windows CMD or PowerShell.
    "wslDistributionName": "Ubuntu-22.04",
    # *** CHANGE THIS to your actual WSL distribution name ***
    # Option to run the wslNemoServer.py script using 'sudo' inside WSL.
    # WARNING: Read notes in mainManager.py/_prepareWslLaunchCommand carefully if enabling.
    "wslUseSudo": False,  # Keep False unless passwordless sudo is correctly configured.
    # Max seconds the client will wait for a response from the WSL server for various requests.
    "serverRequestTimeout": 15.0,
    # Ask the WSL server to unload the NeMo model when this main application exits?
    "unloadRemoteModelOnExit": True,
    # Max seconds the client will wait for the automatically launched WSL server model to be ready.
    "wslServerReadyTimeout": 90.0,

    # --- Transcription Mode & Settings ---
    # "dictationMode": Transcribes after detecting a pause (silence) following speech.
    # "constantIntervalMode": Transcribes audio in the buffer at fixed time intervals.
    "transcriptionMode": "dictationMode",
    # (dictationMode only) Seconds of silence after speech to trigger transcription.
    "dictationMode_silenceDurationToOutput": 0.6,
    # (dictationMode only) Audio loudness below which audio is considered 'silence'.
    "dictationMode_silenceLoudnessThreshold": 0.00035,  # Adjust based on mic/noise.
    # (constantIntervalMode only) Interval in seconds for transcription.
    "constantIntervalMode_transcriptionInterval": 4.0,

    # --- Silence Skipping & Filtering (Applied by TranscriptionOutputHandler) ---
    # Minimum total duration (seconds) of loud audio in a segment to be considered valid.
    "minLoudDurationForTranscription": 0.3,
    # Average segment loudness below which extra filtering applies.
    "silenceSkip_threshold": 0.0002,
    # If avg loudness is low, check start N seconds; if loud, keep segment. (0 to disable)
    "skipSilence_beforeNSecSilence": 0.3,
    # If avg loudness is low, check end N seconds; if loud, keep segment. (0 to disable)
    "skipSilence_afterNSecSilence": 0.3,
    # Words/phrases filtered if segment loudness is below 'loudnessThresholdOf_commonFalseDetectedWords'.
    "commonFalseDetectedWords": ["you", "thank you", "bye", 'amen', 'thanks', 'okay', 'uh', 'um',
                                 'hmm'],
    # Loudness threshold for filtering 'commonFalseDetectedWords'.
    "loudnessThresholdOf_commonFalseDetectedWords": 0.00065,
    # Words/phrases always removed from transcription (case-insensitive).
    "bannedWords": ["<|endoftext|>"],

    # --- General Behavior ---
    # Remove trailing ellipsis (...) or periods (.) from ASR output.
    "removeTrailingDots": True,
    # Initial state of text output (typing/clipboard). Toggled by 'outputToggleKey'.
    "outputEnabled": False,
    # Start microphone recording immediately on launch. Toggled by 'recordingToggleKey'.
    "isRecordingActive": True,
    # Enable audio feedback sounds (e.g., for recording start/stop). Requires 'pygame'.
    "enableAudioNotifications": True,
    # Play sounds for 'Recording ON' and 'Output ENABLED' events?
    "playEnableSounds": False,
    # Master switch for text output action (typing/clipboard).
    "enableTypingOutput": True,

    # --- Hotkeys ---
    # Global key combinations. See keyboard library for syntax.
    "recordingToggleKey": "windows+alt+l",  # Toggle microphone recording.
    "outputToggleKey": "ctrl+q",  # Toggle text output.
    "forceTranscriptionKey": "ctrl+.",
    # New hotkey (Ctrl + Comma). The keyboard library expects ',' for the comma key.

    # --- Text Output Style (Windows Native PyAutoGUI only) ---
    # "letter": Types char by char. "word": Types word by word. "whole": Types entire block.
    "typingMode": "whole",

    # --- Timeouts (0 to disable) ---
    # Max duration (seconds) for a single continuous recording session.
    "maxDurationRecording": 0,
    # Max duration (seconds) the entire application will run.
    "maxDurationProgramActive": 0,
    # Seconds of inactivity before automatically unloading the ASR model.
    "model_unloadTimeout": 60 * 60,  # e.g., 10 minutes
    # Seconds of consecutive silence (while recording) before auto-stopping recording.
    "consecutiveIdleTime": 30 * 60,  # e.g., 2 minutes

    # --- Audio Settings ---
    # Sample rate in Hz (e.g., 16000 for Whisper/NeMo).
    "sampleRate": 16000,
    # Number of audio channels (1 for mono, 2 for stereo). Mono is standard.
    "channels": 1,
    # Audio chunk size in samples. Affects latency and CPU.
    "blockSize": 1024,
    # Audio input device ID or name substring. None for system default.
    # Run `python -m sounddevice` to list devices.
    "deviceId": None,
    # Note: "debugPrint" is no longer directly used from userSettings;
    # DynamicLogger's behavior is controlled by logConfigSets and highOrderOptions.
    # The initial logging level is set via configure_dynamic_logging in utils.py,
    # which can be influenced by logConfigSets.
}

# --- Instantiate and Run Orchestrator ---
orchestrator = None
try:
    logInfo("Initializing application (useRealtimeTranscription)...")
    orchestrator = SpeechToTextOrchestrator(**userSettings)
    logInfo("Starting application run loop (useRealtimeTranscription)...")
    orchestrator.run()
except ValueError as e:  # Catch specific configuration errors
    logCritical(f"!!! CONFIGURATION ERROR: {e}", exc_info=True)  # Log as critical
    logError(
        "Please check the 'userSettings' dictionary in useRealtimeTranscription.py.")  # Specific hint
except ImportError as e:  # Catch errors importing necessary libraries
    logCritical(f"!!! IMPORT ERROR: {e}", exc_info=True)  # Log as critical
    logError(
        "Please ensure all required libraries (like sounddevice, torch, transformers, etc.) are installed correctly in your Python environment.")
except Exception as e:  # Catch any other unexpected errors
    logCritical(f"!!! PROGRAM CRITICAL ERROR: {e}", exc_info=True)  # Log as critical
    logError(traceback.format_exc())  # Keep explicit traceback for unexpected main errors
finally:
    logInfo("Application (useRealtimeTranscription) has stopped.")
    if orchestrator and hasattr(orchestrator, 'stateManager') and orchestrator.stateManager:
        if hasattr(orchestrator.stateManager,
                   'isProgramActive') and orchestrator.stateManager.isProgramActive:
            logWarning("Performing emergency cleanup due to unexpected exit...")
            if hasattr(orchestrator, '_cleanup'):
                try:
                    orchestrator._cleanup()
                except Exception as cleanupError:
                    logError(f"Error during emergency cleanup: {cleanupError}", exc_info=True)
        else:
            logDebug(
                "Program stopped normally or stateManager indicated no longer active, normal cleanup should have occurred.")
    else:
        logDebug(
            "Orchestrator or StateManager not fully initialized, skipping final cleanup check.")

# This print statement is outside the logging system and serves as a final script end marker.
print("Exiting useRealtimeTranscription.py script.")
