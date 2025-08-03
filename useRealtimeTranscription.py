# useRealtimeTranscription.py

import os  # Import os for path joining if needed by logger setup
import traceback
from pathlib import Path  # Import Path

from mainManager import SpeechToTextOrchestrator
# Import setupLogging from utils
from utils import logWarning, logInfo, logError, configureLogging

# --- Configure Logging FIRST ---
# Determine log file path (e.g., in the script's directory)
logDir = Path(os.path.dirname(os.path.abspath(__file__))) / "logs"
logDir.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists
logFilePath = logDir / "realtime_transcription.log"
# Set debugMode=True for verbose logs during troubleshooting
configureLogging(logFileName=logFilePath, debugMode=True)  # Set debug=True for detailed logs

# --- User Configuration ---
# Define configuration as a dictionary. Adjust these values as needed.
userSettings = {
    # --- Core Model Settings ---
    # Choose the ASR model. Examples:
    # "modelName": "openai/whisper-large-v3",   # Local Whisper (requires Transformers)
    # "modelName": "openai/whisper-medium.en", # Local Whisper (English-only, smaller)
    "modelName": "nvidia/canary-180m-flash",  # Example: Remote NeMo (requires WSL server)
    # "modelName": "nvidia/parakeet-rnnt-1.1b", # Example: Remote NeMo (larger, requires WSL server)

    # Target language for transcription.
    # For Whisper, set to None to enable auto-detection (can be slower).
    # For NeMo multilingual models (like Canary), specify the expected language code (e.g., 'en', 'es').
    "language": "en",

    # Force CPU usage for local models (Whisper)?
    # If True, overrides automatic GPU detection for Whisper. Ignored by remote NeMo handler.
    "onlyCpu": False,

    # --- Remote Server Settings (ONLY used if modelName starts with 'nvidia/') ---
    # URL where the wslNemoServer.py script listens inside WSL.
    # 'localhost' relies on WSL's automatic port forwarding (recommended if working).
    # If 'localhost' fails, try the specific WSL IP address (find via `ip addr` in WSL), but note it might change.
    "wslServerUrl": "http://localhost:5001",

    # The exact name of your WSL distribution where the NeMo server will run.
    # Check available distributions by running `wsl -l` in Windows CMD or PowerShell.
    # *** CHANGE THIS to your actual WSL distribution name (e.g., Ubuntu, Debian, Ubuntu-22.04) ***
    "wslDistributionName": "Ubuntu-22.04",

    # Option to run the wslNemoServer.py script using 'sudo' inside WSL.
    # WARNING: Setting this to True will likely FAIL if your WSL user requires a password for sudo,
    # as this script cannot provide it interactively.
    # Set to True ONLY IF you have specifically configured passwordless sudo for executing
    # '/usr/bin/python3 /path/to/wslNemoServer.py ...' for your user within the WSL distribution
    # (e.g., by carefully editing the sudoers file via 'sudo visudo').
    # Generally, this should be kept False, as binding to ports above 1024 (like 5001)
    # usually does not require root privileges.
    "wslUseSudo": True,  # Keep False unless passwordless sudo is configured

    # Max seconds the client will wait for a response from the WSL server for requests
    # like /status, /load, /unload, /transcribe before giving up.
    "serverRequestTimeout": 15.0,

    # Ask the WSL server to unload the NeMo model when this main application exits?
    # Set to False if you want the model to remain loaded in the WSL server process
    # after this application closes (e.g., if the server runs persistently).
    "unloadRemoteModelOnExit": True,

    # Max seconds the client will wait for the automatically launched WSL server
    # to report that the model is loaded (via the /status endpoint) before timing out the startup.
    # Increase if using large models or on slower systems. Set to 0 to disable waiting (unreliable).
    "wslServerReadyTimeout": 90.0,

    # --- Transcription Mode & Settings ---
    # Determines how transcription segments are triggered.
    # "dictationMode": Transcribes after detecting a pause (silence) following speech. Good for dictation.
    # "constantIntervalMode": Transcribes whatever audio is in the buffer at fixed time intervals.
    "transcriptionMode": "dictationMode",

    # (dictationMode only) Seconds of silence detected after speech required to trigger transcription output.
    "dictationMode_silenceDurationToOutput": 0.6,
    # (dictationMode only) Audio loudness level below which audio is considered 'silence'.
    # Adjust based on microphone sensitivity and background noise. Lower values detect quieter sounds as speech.
    "dictationMode_silenceLoudnessThreshold": 0.0004,

    # (constantIntervalMode only) Interval in seconds at which to trigger transcription, regardless of speech/silence.
    "constantIntervalMode_transcriptionInterval": 4.0,

    # --- Silence Skipping & Filtering (Applied by TranscriptionOutputHandler) ---
    # These rules help filter out segments containing only background noise or very short/unwanted utterances.

    # Minimum total duration (in seconds) of audio *above* the 'dictationMode_silenceLoudnessThreshold'
    # within a segment required for the transcription to be considered valid. Helps filter purely silent segments.
    # Set to 0 to disable this check.
    "minLoudDurationForTranscription": 0.3,

    # Average loudness across an entire segment below which extra filtering checks might apply.
    # If average loudness is below this, but the start/end is loud, it might still be kept.
    "silenceSkip_threshold": 0.0002,
    # If average loudness is below 'silenceSkip_threshold', check the loudness of the first N seconds.
    # If this initial part is loud enough (>= dictationMode_silenceLoudnessThreshold), the skip is overridden.
    # Set to 0 to disable this override check.
    "skipSilence_beforeNSecSilence": 0.3,
    # If average loudness is below 'silenceSkip_threshold', also check the loudness of the last N seconds.
    # If this final part is loud enough (>= dictationMode_silenceLoudnessThreshold), the skip is overridden.
    # Set to 0 to disable this override check.
    "skipSilence_afterNSecSilence": 0.3,

    # List of common words/phrases that might be hallucinated by the ASR model during near-silence.
    # These will be filtered out ONLY if the segment's average loudness is also below 'loudnessThresholdOf_commonFalseDetectedWords'.
    "commonFalseDetectedWords": ["you", "thank you", "bye", 'amen', 'thanks', 'okay', 'uh',
                                 'um', 'hmm'],
    # The loudness threshold used for filtering 'commonFalseDetectedWords'.
    "loudnessThresholdOf_commonFalseDetectedWords": 0.00045,

    # --- General Behavior ---
    # Remove trailing ellipsis (...) or periods (.) often added by ASR models.
    "removeTrailingDots": True,

    # Initial state of text output (simulated typing or clipboard copy).
    # Can be toggled during runtime using the 'outputToggleKey'.
    "outputEnabled": False,  # Set to True for easier initial testing

    # Start listening to the microphone immediately when the application launches?
    # Can be toggled during runtime using the 'recordingToggleKey'.
    "isRecordingActive": False,

    # Enable audio feedback sounds for events like recording start/stop, output enable/disable, model unload.
    # Requires the 'pygame' library to be installed (`pip install pygame`).
    "enableAudioNotifications": True,
    # Play specific sounds for 'Recording ON' and 'Output ENABLED' events?
    # If False, these events won't play sounds even if enableAudioNotifications is True.
    "playEnableSounds": False,

    # Master switch to enable/disable the actual text output action (typing/clipboard).
    # Output only occurs if this is True AND the 'outputEnabled' state is True.
    "enableTypingOutput": True,

    # --- Hotkeys ---
    # Define global key combinations to control the application.
    # See https://github.com/boppreh/keyboard#keyboard.add_hotkey for key syntax examples.
    # Keys are case-insensitive. Use '+' to combine keys (e.g., 'ctrl+shift+a').
    # Special keys: 'alt', 'ctrl', 'shift', 'windows', 'left windows', 'right windows', etc.
    "recordingToggleKey": "windows+alt+l",  # Key combination to toggle microphone recording on/off.
    "outputToggleKey": "ctrl+q",  # Key combination to toggle text output (typing/clipboard) on/off.

    # --- Timeouts ---
    # Configure automatic stop conditions. Set to 0 to disable a specific timeout.

    # Maximum duration (in seconds) for a single continuous recording session.
    # If recording is active for this long, it will be automatically stopped. 0 = unlimited.
    "maxDurationRecording": 0,
    # Maximum duration (in seconds) the entire application will run before automatically exiting.
    # Useful for scheduled tasks or limiting resource usage. 0 = unlimited.
    "maxDurationProgramActive": 0,

    # Seconds of inactivity (defined as time since last ASR request/model interaction or recording state change)
    # before automatically unloading the ASR model (local Whisper or remote NeMo) to free up memory (CPU/GPU/RAM).
    # The model will be reloaded automatically when needed again. 0 = never unload automatically.
    "model_unloadTimeout": 10 * 60,  # e.g., 10 minutes

    # Seconds of consecutive silence *while recording is active* (i.e., no valid transcription output produced)
    # before automatically stopping the recording. Helps prevent recording long periods of silence. 0 = unlimited.
    "consecutiveIdleTime": 2 * 60,  # e.g., 2 minutes

    # --- Audio Settings ---
    # Sample rate in Hz for audio capture. Must be compatible with the selected ASR model
    # and supported by your audio hardware. Whisper and NeMo typically expect 16000 Hz.
    "sampleRate": 16000,
    # Number of audio channels. 1 = mono, 2 = stereo. Mono is standard for ASR.
    "channels": 1,
    # Size of audio chunks (in samples) processed by the audio callback.
    # Smaller values (~512, 1024) may reduce latency but increase CPU load.
    # Larger values (~2048, 4096) may increase latency but reduce CPU load.
    # Must be compatible with audio hardware/drivers.
    "blockSize": 1024,
    # Specify the input audio device.
    # Set to None to use the system's default input device.
    # To find device IDs or names: run `python -m sounddevice` in your terminal.
    # Can be an integer ID (e.g., 1) or a substring of the device name (e.g., "Microphone Array", "USB Audio").
    "deviceId": None,

    # --- Debugging ---
    # Set to True to enable detailed DEBUG level log messages in the console and log file.
    # Useful for troubleshooting, but can be very verbose. Set to False for normal operation.
    # Also ensure configureLogging in useRealtimeTranscription.py uses this value.
    "debugPrint": True  # Match this with configureLogging call
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
