# useRealtimeTranscription.py
# ==============================================================================
# Real-Time Speech-to-Text - Main Execution Script
# ==============================================================================
#
# Purpose:
# - This is the main entry point to run the real-time speech-to-text application.
# - It defines the user configuration in the `userSettings` dictionary.
# - It initializes the main `SpeechToTextOrchestrator` with the user settings.
# - It starts the application's main loop and handles critical errors.
#
# To-Do / Checks:
# - Consider a way to change user settings while the program is running.
# - Investigate if the model unloads earlier than the specified `model_unloadTimeout`.
# ==============================================================================
import traceback
from mainManager import SpeechToTextOrchestrator
# Import the universal logger instances directly
from utils.loggerInstance import uniLogger, uniDebugLogger
import logging

# --- User Configuration ---
# Define configuration as a dictionary. Adjust these values as needed.

# Set the logging level for the 'transformers' library to ERROR to reduce verbose output.
logging.getLogger("transformers").setLevel(logging.ERROR)

userSettings = {
    # --- Core Model Settings ---
    # Choose the ASR model. Examples:
    # "modelName": "openai/whisper-large-v3",
    # "modelName": "nvidia/parakeet-rnnt-1.1b", # Remote NeMo (requires WSL server)
    "modelName": "openai/whisper-tiny",  # Local Whisper (smaller for quick test)
    # Target language for transcription. 'en' for English.
    "language": "en",
    # Force CPU usage for local models. If True, overrides automatic GPU detection for Whisper.
    "CPU": False,

    # --- Remote Server Settings (ONLY used if modelName starts with 'nvidia/') ---
    # URL where the wslNemoServer.py script listens inside WSL.
    "wslServerUrl": "http://localhost:5001",
    # The exact name of your WSL distribution. Check with `wsl -l`.
    "wslDistributionName": "Ubuntu-22.04",
    # *** CHANGE THIS to your actual WSL distribution name ***
    # Use 'sudo' to run the server script in WSL. Requires passwordless sudo configuration.
    "wslUseSudo": False,
    # Max seconds the client will wait for a response from the WSL server.
    "serverRequestTimeout": 15.0,
    # Ask the WSL server to unload the NeMo model when this application exits.
    "unloadRemoteModelOnExit": True,
    # Max seconds to wait for the automatically launched WSL server to become ready.
    "wslServerReadyTimeout": 90.0,

    # --- Transcription Mode & Settings ---
    # "dictationMode": Transcribes after a pause.
    # "constantIntervalMode": Transcribes at fixed time intervals.
    "transcriptionMode": "dictationMode",
    # (dictationMode) Seconds of silence after speech to trigger transcription.
    "dictationMode_silenceDurationToOutput": 0.6,
    # (dictationMode) Audio loudness below which is considered 'silence'.
    "dictationMode_silenceLoudnessThreshold": 0.00035,
    # (constantIntervalMode) Interval in seconds for transcription.
    "constantIntervalMode_transcriptionInterval": 4.0,

    # --- Silence Skipping & Filtering ---
    # Minimum total duration (seconds) of loud audio in a segment to be considered valid.
    "minLoudDurationForTranscription": 0.3,
    # Average segment loudness below which extra filtering applies.
    "silenceSkip_threshold": 0.0002,
    # If avg loudness is low, check start N seconds; if loud, keep segment.
    "skipSilence_beforeNSecSilence": 0.3,
    # If avg loudness is low, check end N seconds; if loud, keep segment.
    "skipSilence_afterNSecSilence": 0.3,
    # Words filtered if segment loudness is below the corresponding threshold.
    "commonFalseDetectedWords": ["you", "thank you", "bye", 'amen', 'thanks', 'okay', 'uh', 'um',
                                 'hmm'],
    # Loudness threshold for filtering 'commonFalseDetectedWords'.
    "loudnessThresholdOf_commonFalseDetectedWords": 0.00065,
    # Words/phrases always removed from transcription (case-insensitive).
    "bannedWords": ["<|endoftext|>"],

    # --- General Behavior ---
    # Remove trailing ellipsis (...) or periods (.) from ASR output.
    "removeTrailingDots": True,
    # Initial state of text output (typing/clipboard).
    "outputEnabled": False,
    # Start microphone recording immediately on launch.
    "isRecordingActive": True,
    # Enable audio feedback sounds (e.g., for recording start/stop).
    "enableAudioNotifications": True,
    # Play sounds for 'Recording ON' and 'Output ENABLED' events.
    "playEnableSounds": False,
    # Master switch for text output action (typing/clipboard).
    "enableTypingOutput": True,
    # "letter": Types char by char. "word": Types word by word. "whole": Pastes entire block.
    "typingMode": "whole",

    # --- Hotkeys ---
    # Global key combinations to control the application.
    "recordingToggleKey": "windows+alt+l",  # Toggle microphone recording.
    "outputToggleKey": "ctrl+q",  # Toggle text output.
    "forceTranscriptionKey": "ctrl+.",  # Force transcription of the current audio buffer.

    # --- Timeouts (0 to disable) ---
    # Max duration (seconds) for a single continuous recording session.
    "maxDurationRecording": 0,
    # Max duration (seconds) the entire application will run.
    "maxDurationProgramActive": 0,
    # Seconds of inactivity before automatically unloading the ASR model to save VRAM.
    "model_unloadTimeout": 60 * 10,  # 10 minutes
    # Seconds of consecutive silence (while recording) before auto-stopping recording.
    "consecutiveIdleTime": 60 * 2,  # 2 minutes

    # --- Audio Settings ---
    # Sample rate in Hz (16000 is standard for Whisper/NeMo).
    "sampleRate": 16000,
    # Number of audio channels (1 for mono).
    "channels": 1,
    # Audio chunk size in samples. Affects latency.
    "blockSize": 1024,
    # Audio input device ID. None for system default. Run `python -m sounddevice` to list devices.
    "deviceId": None,
}

# --- Instantiate and Run Orchestrator ---.                
orchestrator = None
try:
    # Initialize the main application orchestrator with the defined settings.
    uniLogger.info("Initializing application...")
    orchestrator = SpeechToTextOrchestrator(**userSettings)
    # Start the main application loop. This will block until the program exits.
    uniLogger.info("Starting application run loop...")
    orchestrator.run()
except ValueError as e:  # Catch specific configuration errors
    uniLogger.critical(f"!!! CONFIGURATION ERROR: {e}", excInfo=True)
    uniLogger.error("Please check the 'userSettings' dictionary in useRealtimeTranscription.py.")
except ImportError as e:  # Catch errors from missing libraries
    uniLogger.critical(f"!!! IMPORT ERROR: {e}", excInfo=True)
    uniLogger.error("Please ensure all required libraries are installed correctly.")
except Exception as e:  # Catch any other unexpected errors during setup or runtime
    uniLogger.critical(f"!!! PROGRAM CRITICAL ERROR: {e}", excInfo=True)
finally:
    # This block executes after the main loop finishes, either normally or due to an error.
    uniLogger.info("Application (useRealtimeTranscription) has stopped.")
    # The orchestrator's run() method has its own internal finally block for resource cleanup.

# Final confirmation print outside the logging system to indicate the script has finished.
print("Exiting useRealtimeTranscription.py script.")
