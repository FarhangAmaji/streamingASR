# useRealtimeTranscription.py
import traceback

from mainManager import SpeechToTextOrchestrator
from utils import logWarning, logInfo, logError

# --- User Configuration ---
# Define configuration as a dictionary. Adjust these values as needed.
userSettings = {
    # --- Core Model Settings ---
    # Choose the ASR model. Examples:
    # "modelName": "openai/whisper-large-v3",   # Local Whisper (requires Transformers)
    # "modelName": "openai/whisper-medium.en", # Local Whisper (English-only, smaller)
    "modelName": "nvidia/canary-180m-flash",
    # Remote NeMo (requires WSL server running wslNemoServer.py)
    # "modelName": "nvidia/parakeet-rnnt-1.1b", # Remote NeMo (larger, requires WSL server)

    "language": "en",  # Language code (e.g., 'en', 'es'). Use None for Whisper auto-detect.
    "onlyCpu": False,  # Force CPU for local models (Whisper)? Ignored by remote NeMo handler.

    # --- Remote Server Settings (ONLY used if modelName starts with 'nvidia/') ---
    "wslServerUrl": "http://localhost:5001",  # URL where wslNemoServer.py listens
    "wslDistributionName": "Ubuntu-22.04",
    # *** CHANGE THIS to your actual WSL distribution name (e.g., Ubuntu, Debian) *** Use `wsl -l` in CMD to check.
    "wslUseSudo": True,
    "serverRequestTimeout": 15.0,  # Seconds client waits for server response before timeout
    "unloadRemoteModelOnExit": True,
    # Ask WSL server to unload model when this app closes? Set to False if server should persist.

    # --- Transcription Mode & Settings ---
    "transcriptionMode": "dictationMode",  # "dictationMode" or "constantIntervalMode"
    "dictationMode_silenceDurationToOutput": 0.6,
    # Seconds of silence after speech to trigger output
    "dictationMode_silenceLoudnessThreshold": 0.0004,
    # Audio level below which is considered silence (adjust based on mic sensitivity)
    "constantIntervalMode_transcriptionInterval": 4.0,
    # Seconds between transcriptions in constant interval mode

    # --- Silence Skipping & Filtering (Applied in Output Handler) ---
    # These help filter out noise or unwanted short utterances.
    "minLoudDurationForTranscription": 0.3,
    # Min required total seconds of audio *above* silence threshold in a segment to be considered valid (0 to disable)
    "silenceSkip_threshold": 0.0002,
    # Average loudness across a segment below which *extra* checks apply (using start/end loudness)
    "skipSilence_beforeNSecSilence": 0.3,
    # If avg loudness is low, check loudness in first N sec (if loud enough, override skip) (0 to disable override)
    "skipSilence_afterNSecSilence": 0.3,
    # If avg loudness is low, check loudness in last N sec (if loud enough, override skip) (0 to disable override)
    "commonFalseDetectedWords": ["you", "thank you", "bye", 'amen', 'thanks', 'okay', 'uh',
                                 'um', 'hmm'],
    # List of words to filter if segment loudness is below the next threshold
    "loudnessThresholdOf_commonFalseDetectedWords": 0.00045,
    # Loudness below which words in 'commonFalseDetectedWords' list are filtered out

    # --- General Behavior ---
    "removeTrailingDots": True,  # Remove "..." or "." from end of transcription
    "outputEnabled": False,
    # Initial state of sending output (typing/clipboard) - toggled by hotkey `outputToggleKey`
    "isRecordingActive": True,
    # Start recording immediately on launch? Toggled by hotkey `recordingToggleKey`.
    "enableAudioNotifications": True,
    # Play sounds for events like start/stop (requires pygame)
    "playEnableSounds": False,
    # Play sounds specifically for 'recording ON' and 'output ENABLED' events?
    "enableTypingOutput": True,
    # Master switch for simulated typing / clipboard output (controlled by `outputEnabled` state)

    # --- Hotkeys ---
    # See https://github.com/boppreh/keyboard#keyboard.add_hotkey for key syntax
    "recordingToggleKey": "windows+alt+l",  # Key combination to toggle recording on/off
    "outputToggleKey": "ctrl+q",
    # Key combination to toggle text output (typing/clipboard) on/off

    # --- Timeouts ---
    # Set to 0 to disable a specific timeout
    "maxDurationRecording": 0,
    # Max seconds for a single recording session (0 = no limit). Recording stops automatically after this duration.
    "maxDurationProgramActive": 0,
    # Max seconds the entire program will run before automatically exiting (0 = no limit).
    "model_unloadTimeout": 10 * 60,
    # Seconds of inactivity (no recording) before automatically unloading model (local or remote) to save resources (0 = never unload).
    "consecutiveIdleTime": 2 * 60,
    # Seconds of no valid transcription output *while recording is active* before stopping recording automatically (0 = no limit).

    # --- Audio Settings ---
    "sampleRate": 16000,
    # Desired sample rate (Hz). Whisper/NeMo typically use 16000. Check device compatibility.
    "channels": 1,
    # Number of audio channels (1 = mono, 2 = stereo). Mono is usually preferred.
    "blockSize": 1024,
    # Audio buffer size (samples per callback chunk). Smaller values may reduce latency but increase CPU load. Powers of 2 are common.
    "deviceId": None,
    # Specific audio input device ID (integer or string name part). Set to None to use the system's default input device. Use 'python -m sounddevice' to list available devices.

    # --- Debugging ---
    "debugPrint": False
    # Enable detailed DEBUG log messages for troubleshooting? Set to True for development.
}

# --- Instantiate and Run Orchestrator ---
orchestrator = None  # Initialize to None ensures cleanup can be attempted even if __init__ fails
try:
    logInfo("Initializing application...")
    # Create the main orchestrator instance, passing all settings
    orchestrator = SpeechToTextOrchestrator(**userSettings)
    logInfo("Starting application run loop...")
    # Start the main loop (this call is blocking until the program exits)
    orchestrator.run()

except ValueError as e:
    # Catch specific configuration errors (like missing required settings)
    logError(f"\n!!! CONFIGURATION ERROR: {e}")
    logError("Please check the 'userSettings' dictionary in useRealtimeTranscription.py.")
except ImportError as e:
    # Catch errors importing necessary libraries
    logError(f"\n!!! IMPORT ERROR: {e}")
    logError(
        "Please ensure all required libraries (like sounddevice, torch, transformers, etc.) are installed correctly in your Python environment.")
except Exception as e:
    # Catch any other unexpected errors during initialization or the main loop
    logError(f"\n!!! PROGRAM CRITICAL ERROR: {e}")
    logError(traceback.format_exc())  # Print the full traceback for debugging
finally:
    # This block always runs, even if errors occur or the program exits normally
    logInfo("Application has stopped.")
    # Attempt cleanup only if the orchestrator was successfully initialized
    # and might still think the program is active (e.g., due to unclean exit)
    if orchestrator and hasattr(orchestrator,
                                'stateManager') and orchestrator.stateManager and orchestrator.stateManager.isProgramActive:
        logWarning("Performing emergency cleanup due to unexpected exit...")
        orchestrator._cleanup()

print("Exiting useRealtimeTranscription.py script.")  # Final message indicating script end
