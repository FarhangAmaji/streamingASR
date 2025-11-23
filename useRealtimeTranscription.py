# useRealtimeTranscription.py
# ==============================================================================
# Real-Time Speech-to-Text - Main Execution Script (GUI Enabled)
# ==============================================================================
#
# Purpose:
# - This is the main entry point to run the real-time speech-to-text application.
# - It initializes the PyQt Application (GUI main thread).
# - It defines the **initial default settings** in `userSettings`.
# - It initializes the main `SpeechToTextOrchestrator` and runs it
#   in a separate background thread.
# - It creates and shows the `ControlPanel` GUI, connecting it to the
#   Orchestrator's configuration object.
# ==============================================================================
import sys
import traceback
import logging
from threading import Thread

# --- Qt and GUI Imports ---
try:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
except ImportError:
    print("CRITICAL ERROR: PyQt5 not found. Please run 'pip install PyQt5'")
    sys.exit(1)

# --- Application Core Imports ---
from mainManager import SpeechToTextOrchestrator
from gui import ControlPanel  # Import the new GUI
from utils.loggerInstance import uniLogger, uniDebugLogger

# --- User Configuration (Initial Defaults) ---
# These are the default values. The GUI (gui.py) will load these
# and then allow the user to change them at runtime.

# Set the logging level for 'transformers' to ERROR to reduce verbose output.
logging.getLogger("transformers").setLevel(logging.ERROR)

userSettings = {
    # --- Core Model Settings ---
    # "modelName": "openai/whisper-large-v3",
    # "modelName": "openai/whisper-tiny",#
    "modelName": "nvidia/canary-180m-flash",
    # "modelName": "nvidia/stt_en_fastconformer_ctc_small",
    "language": "en",
    "CPU": True,

    # --- Remote Server Settings (ONLY used if modelName starts with 'nvidia/') ---
    "wslServerUrl": "http://172.21.73.28:5001",
    "wslDistributionName": "Ubuntu",  # *** CHANGE THIS if needed ***
    "wslUseSudo": False,
    "serverRequestTimeout": 15.0,
    "unloadRemoteModelOnExit": True,
    "wslServerReadyTimeout": 90.0,

    # --- Transcription Mode & Settings ---
    "transcriptionMode": "dictationMode",
    "dictationMode_silenceDurationToOutput": 0.6,
    "dictationMode_silenceLoudnessThreshold": 0.00035,
    "constantIntervalMode_transcriptionInterval": 4.0,

    # --- Silence Skipping & Filtering ---
    "minLoudDurationForTranscription": 0.3,
    "silenceSkip_threshold": 0.0002,
    "skipSilence_beforeNSecSilence": 0.3,
    "skipSilence_afterNSecSilence": 0.3,
    "commonFalseDetectedWords": ["you", "thank you", "bye", 'amen', 'thanks', 'okay', 'uh', 'um',
                                 'hmm'],
    "loudnessThresholdOf_commonFalseDetectedWords": 0.00065,
    "bannedWords": ["<|endoftext|>"],

    # --- General Behavior ---
    "removeTrailingDots": True,
    "outputEnabled": False,
    "isRecordingActive": True,
    "enableAudioNotifications": True,
    "playEnableSounds": False,
    "enableTypingOutput": True,
    "typingMode": "whole",

    # --- Hotkeys ---
    "recordingToggleKey": "windows+alt+l",
    "outputToggleKey": "ctrl+q",
    "forceTranscriptionKey": "ctrl+.",

    # --- Timeouts (0 to disable) ---
    "maxDurationRecording": 0,
    "maxDurationProgramActive": 0,
    "model_unloadTimeout": 60 * 10,  # 10 minutes
    "consecutiveIdleTime": 60 * 2,  # 2 minutes

    # --- Audio Settings ---
    "sampleRate": 16000,
    "channels": 1,
    "blockSize": 1024,
    "deviceId": None,  # 'None' for default, GUI will load this
}


def main():
    """
    Main execution function.
    Initializes the Qt Application, the ASR Orchestrator (in a thread),
    and the Control Panel GUI.
    """
    orchestrator = None
    orchestrator_thread = None

    try:
        # 1. Initialize the Qt Application (MUST be first)
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        app = QApplication(sys.argv)
        uniLogger.info("QApplication initialized (Main Thread).")

        # 2. Initialize the main application orchestrator
        # This creates the object but does NOT start its blocking .run() method yet.
        uniLogger.info("Initializing application orchestrator...")
        orchestrator = SpeechToTextOrchestrator(**userSettings)

        # 3. Initialize the GUI Control Panel
        # We pass the orchestrator's 'config' and 'audioHandler' objects
        # to the GUI. The GUI can now call config.set() and audioHandler.listAudioDevices().
        uniLogger.info("Initializing Control Panel GUI...")

        # Check if audioHandler was initialized (it might not be if error)
        if not orchestrator.audioHandler:
            uniLogger.critical("AudioHandler failed to initialize. GUI cannot list devices.")
            # We can still proceed, but the device list will be empty/fallback

        control_panel = ControlPanel(
            config=orchestrator.config,
            audio_handler=orchestrator.audioHandler
            # Pass the *actual* audioHandler instance
        )
        control_panel.show()

        # 4. Start the main application loop (orchestrator.run())
        # This MUST be done in a separate thread so the GUI (app.exec_())
        # does not block.
        orchestrator_thread = Thread(
            target=orchestrator.run,
            name="OrchestratorThread",
            daemon=True  # 'daemon=True' ensures thread exits when main script exits
        )
        orchestrator_thread.start()
        uniLogger.info(
            "Starting application run loop in a background thread...")

        # 5. Start the Qt event loop (THIS IS A BLOCKING CALL)
        # This runs the GUI, handles button clicks, etc.
        # When the GUI window is closed, app.exec_() will return.
        uniLogger.info("Starting Qt event loop (GUI)...")
        exit_code = app.exec_()
        uniLogger.info(f"Qt event loop finished with exit code {exit_code}.")

    except ValueError as e:
        uniLogger.critical(f"!!! CONFIGURATION ERROR: {e}", excInfo=True)
    except ImportError as e:
        uniLogger.critical(f"!!! IMPORT ERROR: {e}", excInfo=True)
    except Exception as e:
        uniLogger.critical(f"!!! PROGRAM CRITICAL ERROR: {e}", excInfo=True)
    finally:
        # This block executes after the Qt loop finishes (e.g., GUI closed).
        uniLogger.info(
            "Application (useRealtimeTranscription) is stopping...")
        if orchestrator and orchestrator.stateManager:
            # Signal the orchestrator's thread to stop
            uniLogger.info("Requesting orchestrator shutdown...")
            orchestrator.stateManager.stopProgram()

        # Wait for the orchestrator thread to finish its cleanup
        if orchestrator_thread and orchestrator_thread.is_alive():
            uniLogger.info("Waiting for orchestrator thread to join...")
            orchestrator_thread.join(timeout=5.0)
            if orchestrator_thread.is_alive():
                uniLogger.warning("Orchestrator thread did not exit cleanly.")

    # Final confirmation print
    print("Exiting useRealtimeTranscription.py script.")


# --- Script Entry Point ---
if __name__ == "__main__":
    main()