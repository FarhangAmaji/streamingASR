import gc
import os
import queue
import threading
import time
from pathlib import Path

import huggingface_hub
import keyboard
import numpy as np
import pyautogui
import pygame
import sounddevice as sd
import soundfile as sf
import torch
from transformers import pipeline


class BaseTranscriber:
    """
    Base class for speech-to-text transcription functionality.
    Handles common ASR model management (loading, unloading, basic inference)
    and resource handling (GPU memory).
    """

    def __init__(self,
                 modelName="openai/whisper-large-v3",
                 language="en",
                 removeTrailingDots=True,
                 onlyCpu=False,  # User/Developer: Option to force CPU usage
                 debugPrint=False):
        """
        Initialize the base transcriber with common parameters.

        Args:
            modelName (str): User: Name of the Hugging Face ASR model (e.g., "openai/whisper-large-v3").
            language (str): User: Language code for transcription (e.g., "en", "es"). Affects model output.
            removeTrailingDots (bool): User: If True, removes trailing '.' characters from the final transcription.
            onlyCpu (bool): User/Developer: If True, forces the model to run on the CPU, even if CUDA is available.
            debugPrint (bool): Developer: If True, enables verbose printing for debugging and monitoring.
        """
        self.modelName = modelName
        self.language = language
        self.removeTrailingDots = removeTrailingDots
        self.debugPrint = debugPrint
        self.asr = None  # Developer: Holds the loaded pipeline object
        self.modelLoaded = False  # Developer: Flag to track if the model is currently loaded

        if onlyCpu:
            self.device = torch.device('cpu')
            self._debugPrint("INFO: CPU usage forced by 'onlyCpu=True'.")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cuda':
                self._debugPrint("INFO: CUDA GPU detected and will be used.")
            else:
                self._debugPrint("INFO: CUDA GPU not found or 'onlyCpu=True', using CPU.")

    def _debugPrint(self, message):
        """Developer: Helper function for conditional debug printing."""
        if self.debugPrint:
            print(message)

    def loadModel(self):
        """
        Developer: Loads the specified ASR model onto the configured device.
        Initializes the Transformers pipeline. Includes configuration for
        long-form audio handling via `return_timestamps=True`.
        """
        if not self.modelLoaded:
            self._debugPrint(f"Loading model '{self.modelName}' to {self.device}...")

            self._monitorMemory()  # Monitor before load
            self._cudaClean()

            gen_kwargs = {"language": self.language}
            gen_kwargs["return_timestamps"] = True
            self._debugPrint(f"Pipeline generate_kwargs: {gen_kwargs}")

            try:
                self.asr = pipeline(
                    "automatic-speech-recognition",
                    model=self.modelName,
                    generate_kwargs=gen_kwargs,  # Pass configured generation arguments
                    device=self.device
                )
                self.modelLoaded = True
                self._debugPrint("Model pipeline loaded successfully.")

                self._debugPrint("Warming up the model...")
                dummyAudio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
                _ = self.asr({"raw": dummyAudio, "sampling_rate": 16000})  # Discard warm-up result
                self._debugPrint("Model warm-up complete.")

            except Exception as e:
                print(f"!!! ERROR loading ASR model: {e}")
                print("!!! Check model name, internet connection, and dependencies.")
                self.modelLoaded = False  # Ensure flag is False if loading failed

            self._monitorMemory()  # Monitor after load

    def unloadModel(self):
        """Developer: Unloads the ASR model from memory and attempts to clear GPU cache."""
        if self.modelLoaded:
            self._debugPrint(f"Unloading model '{self.modelName}' from {self.device}...")

            del self.asr
            self.asr = None

            self._cudaClean()

            self.modelLoaded = False
            self._debugPrint("Model unloaded.")
            self._monitorMemory()  # Monitor after unload

    def _cudaClean(self):
        """Developer: Performs garbage collection and attempts to clear PyTorch's CUDA cache."""
        self._debugPrint("Cleaning CUDA memory...")
        gc.collect()  # Python garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # PyTorch CUDA cache clearing
            torch.cuda.ipc_collect()
        self._debugPrint("CUDA memory cleaning attempt finished.")

    def _monitorMemory(self):
        """Developer: Monitors and prints current GPU memory usage if debugPrint is enabled."""
        if torch.cuda.is_available() and self.debugPrint:
            try:
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            except Exception as e:
                print(f"Warning: Failed to get GPU memory stats: {e}")

    def transcribeAudio(self, audioData, sampleRate):
        """
        Developer: Transcribes a given audio data segment using the loaded ASR model.
        Handles basic audio pre-processing (mono conversion) and extracts text from the model result.

        Args:
            audioData (numpy.ndarray): Audio data (as float32 NumPy array).
                                       This is the 'Transcription Window'.
            sampleRate (int): Sample rate of the audio data (e.g., 16000 Hz).

        Returns:
            str: The transcribed text. Returns an empty string if transcription fails,
                 the model is not loaded, or the result format is unexpected.
        """
        if not self.modelLoaded:
            self._debugPrint("Transcription skipped: Model not loaded.")
            return ""

        if len(audioData.shape) > 1 and audioData.shape[1] > 1:
            if self.debugPrint:
                print(f"DEBUG: Converting stereo audio (shape {audioData.shape}) to mono.")
            audioData = np.mean(audioData, axis=1)  # Average channels

        if audioData.dtype != np.float32:
            self._debugPrint(
                f"DEBUG: Converting audio data type from {audioData.dtype} to float32.")
            audioData = audioData.astype(np.float32)

        transcription = ""  # Developer: Initialize default return value
        try:
            asr_input = {"raw": audioData, "sampling_rate": sampleRate}
            segment_duration_sec = len(audioData) / sampleRate
            self._debugPrint(
                f"DEBUG: Sending {segment_duration_sec:.2f}s audio Transcription Window to ASR pipeline...")

            result = self.asr(asr_input)
            self._debugPrint("DEBUG: ASR pipeline processing finished.")

            self._debugPrint(f"ASR Raw Result: {result}")

            if isinstance(result, dict) and "text" in result:
                transcription = result["text"]
            else:
                print(
                    f"Warning: Unexpected ASR result structure: {type(result)}. Could not extract text.")
                transcription = ""

            if self.removeTrailingDots and isinstance(transcription, str):
                transcription = transcription.strip(
                    '. ')  # Remove trailing dots and surrounding whitespace

        except Exception as e:
            print(f"!!! ERROR during transcription: {e}")
            transcription = ""  # Ensure empty string is returned on error

        if self.debugPrint:
            print(f"DEBUG: Transcription result (cleaned): '{transcription[:100]}...'")

        return transcription

    def cleanup(self):
        """Developer: Cleans up resources, primarily by unloading the model."""
        self._debugPrint("BaseTranscriber cleanup initiated.")
        if self.modelLoaded:
            self.unloadModel()
        self._debugPrint("BaseTranscriber cleanup complete.")

    @staticmethod
    def listAsrModels():
        """
        Static method to retrieve all ASR models from Hugging Face Hub.

        Returns:
            List[str]: A list of model IDs that support automatic speech recognition.
        """
        asrModels = [model.id for model in
                     huggingface_hub.list_models(filter="automatic-speech-recognition")]
        return asrModels


class FileTranscriber(BaseTranscriber):
    """
    User: Handles transcription of pre-recorded audio files.
    Developer: Inherits model management and basic transcription from BaseTranscriber.
    """

    def __init__(self,
                 modelName="openai/whisper-large-v3",
                 language="en",
                 removeTrailingDots=True,
                 onlyCpu=False,  # Inherited arg
                 debugPrint=False):
        """
        Initialize the file transcriber.

        Args:
             modelName (str): User: Name of the Hugging Face ASR model.
             language (str): User: Language code for transcription.
             removeTrailingDots (bool): User: If True, remove trailing '.' from transcription.
             onlyCpu (bool): User/Developer: Force CPU usage.
             debugPrint (bool): Developer: Enable verbose debug printing.
        """
        super().__init__(modelName=modelName,
                         language=language,
                         removeTrailingDots=removeTrailingDots,
                         onlyCpu=onlyCpu,
                         debugPrint=debugPrint)

    def transcribeFile(self, audioFilePath, outputFilePath=None):
        """
        User: Transcribe an audio file and optionally save the transcription to a text file.

        Args:
            audioFilePath (str): User: Path to the input audio file (e.g., .wav, .mp3).
            outputFilePath (str, optional): User: Path to save the transcription text file.
                                            If None, prints transcription to the console. Defaults to None.

        Returns:
            str: Transcribed text, or None if transcription fails.
        """
        self._debugPrint(f"Attempting to transcribe file: {audioFilePath}")
        transcription = None  # Default return value
        try:
            if not self.modelLoaded:
                self.loadModel()
            if not self.modelLoaded:
                print("Error: Model could not be loaded for file transcription.")
                return None

            audioData, sampleRate = sf.read(audioFilePath, dtype='float32')  # Ensure float32
            self._debugPrint(
                f"File read successfully. Sample rate: {sampleRate}, Duration: {len(audioData) / sampleRate:.2f}s")

            transcription = self.transcribeAudio(audioData, sampleRate)

            if transcription is not None:  # Check if transcription succeeded
                if outputFilePath:
                    try:
                        with open(outputFilePath, 'w', encoding='utf-8') as outputFile:
                            outputFile.write(transcription)
                        self._debugPrint(f"Transcription saved to: {outputFilePath}")
                    except IOError as e:
                        print(f"Error writing transcription to file {outputFilePath}: {e}")
                        print("Transcription:", transcription)  # Print to console as fallback
                else:
                    print("\n--- Transcription ---")
                    print(transcription)
                    print("---------------------\n")
            else:
                print("Transcription failed (returned None).")

            return transcription  # Return the result

        except FileNotFoundError:
            print(f"Error: Audio file not found at {audioFilePath}")
            return None
        except Exception as e:
            print(f"Error transcribing file '{audioFilePath}': {e}")
            return None

    def cleanup(self):
        """Developer: Cleans up resources by calling the base class cleanup."""
        super().cleanup()
        self._debugPrint("File transcriber cleanup complete.")


class SpeechToTextTranscriber(BaseTranscriber):
    """
    User: Handles real-time speech-to-text transcription from a microphone input.
    Offers different modes for controlling transcription behavior.
    Developer: Manages audio stream, buffering, state tracking (for dictationMode),
    and interaction with BaseTranscriber for ASR. Uses threading for non-blocking
    hotkey monitoring and model management.
    """

    def __init__(self,
                 modelName="openai/whisper-large-v3",
                 language="en",
                 transcriptionMode="dictationMode",
                 constantIntervalMode_transcriptionInterval=3,
                 dictationMode_silenceDurationToOutput=0.6,
                 silenceSkip_threshold=.0002,
                 dictationMode_silenceLoudnessThreshold=2,
                 skipSilence_afterNSecSilence=0.3,
                 commonFalseDetectedWords=None,
                 loudnessThresholdOf_commonFalseDetectedWords=.0008,
                 maxDuration_recording=10000,
                 maxDuration_programActive=60 * 60,
                 model_unloadTimeout=20 * 60,
                 consecutiveIdleTime=100,
                 isRecordingActive=True,  # User: Start with recording ON (True) or OFF (False)
                 isProgramActive=True,  # Developer: Internal flag to control main loops
                 outputEnabled=False,
                 sampleRate=16000,
                 channels=1,  # User/Developer: Number of audio channels (1=mono recommended).
                 removeTrailingDots=True,  # User: Remove trailing '.' from transcription.
                 playEnableSounds=True,
                 onlyCpu=False,  # User/Developer: Force use of CPU.
                 debugPrint=False,  # Developer: Enable verbose debug logging.
                 recordingToggleKey="win+alt+l",  # User: Hotkey to toggle recording on/off.
                 outputToggleKey="ctrl+q"):  # User: Hotkey to toggle text output on/off.
        """
        Initialize the real-time speech-to-text transcriber.

        Args:
            modelName (str): User: Name of the Whisper model to use (e.g., "openai/whisper-large-v3").
            language (str): User: Language code for transcription (e.g., "en").
            transcriptionMode (str): User: Mode of operation. Options:
                                     - "dictationMode": Transcribes after speech pauses (natural dictation).
                                     - "constantIntervalMode": Transcribes at fixed intervals.
            constantIntervalMode_transcriptionInterval (float): User: Seconds between transcription attempts in 'constantIntervalMode'.
            dictationMode_silenceDurationToOutput (float): User: Seconds of silence required *after* speech to trigger transcription in 'dictationMode'.
            silenceSkip_threshold (float): User: Overall average loudness (mean abs amplitude) threshold for a Transcription Window.
                                           If the window's average loudness is below this, transcription *might* be skipped.
            dictationMode_silenceLoudnessThreshold (float): User: Loudness threshold (mean abs amplitude) for individual incoming audio chunks (Smallest Sound Segments).
                                                            Used in 'dictationMode' to determine if a chunk is speech or silence for state tracking.
                                                            Also used by 'skipSilence_afterNSecSilence'.
            skipSilence_afterNSecSilence (float): User: Refines the `silenceSkip_threshold`. If a segment's overall loudness is below `silenceSkip_threshold`,
                                                  it will *only* be skipped if the average loudness of the *last* N seconds (this value) of the segment
                                                  was *also* below `dictationMode_silenceLoudnessThreshold`. Helps prevent skipping segments that end with a quiet word.
                                                  Set to 0 to disable this trailing check (skip based only on overall average).
            commonFalseDetectedWords (list | None): User: A list of lowercase words (e.g., ["you", "thank you"]) that are often falsely detected during silence/noise.
            loudnessThresholdOf_commonFalseDetectedWords (float): User: If a common false word is detected, it will be ignored (treated as silence) if the overall
                                                                   average loudness of the Transcription Window was below this threshold.
            maxDuration_recording (int): User: Maximum duration (seconds) for a single continuous recording session before it's automatically stopped.
            maxDuration_programActive (int): User: Maximum duration (seconds) the entire program will run before automatically shutting down.
            model_unloadTimeout (int): User: Time (seconds) of inactivity (recording off) after which the ASR model is unloaded from memory (GPU/RAM) to save resources.
            consecutiveIdleTime (int): User: Time (seconds) of continuous effective silence (no valid transcriptions generated, considering filters)
                                       after which the recording will automatically stop.
            isRecordingActive (bool): User: Initial state of recording when the program starts (True = starts recording, False = starts paused).
            isProgramActive (bool): Developer: Internal flag controlling the main application loop. Typically starts True.
            outputEnabled (bool): User: Initial state of the text output feature (simulated typing) when the program starts.
            sampleRate (int): User/Developer: Desired sample rate in Hz (e.g., 16000). The actual rate used depends on the audio device capabilities.
            channels (int): User/Developer: Number of audio input channels (1 for mono is recommended for most ASR).
            removeTrailingDots (bool): User: If True, removes trailing '.' and whitespace from the final transcription.
            playEnableSounds (bool): User: If True, plays short audio cues for enabling actions ('recordingOn', 'outputEnabled'). Other sounds always play if available.
            onlyCpu (bool): User/Developer: If True, forces model execution on the CPU.
            debugPrint (bool): Developer: If True, enables extensive logging to the console for troubleshooting.
            recordingToggleKey (str): User: String representation of the hotkey to toggle recording (e.g., 'win+alt+l', 'ctrl+shift+r'). See 'keyboard' library format.
            outputToggleKey (str): User: String representation of the hotkey to toggle text output (e.g., 'ctrl+q').
        """
        super().__init__(modelName=modelName, language=language,
                         removeTrailingDots=removeTrailingDots,
                         onlyCpu=onlyCpu,
                         debugPrint=debugPrint)

        self.transcriptionMode = transcriptionMode
        self.dictationMode_silenceDurationToOutput = dictationMode_silenceDurationToOutput
        self.skipSilence_afterNSecSilence = skipSilence_afterNSecSilence  # For refined silence skipping
        self.constantIntervalMode_transcriptionInterval = constantIntervalMode_transcriptionInterval  # For constant mode

        self.sampleRate = sampleRate  # Target sample rate
        self.channels = channels  # Target channels
        self.blockSize = 1024  # Developer: Frames per audio callback buffer (affects latency and chunk size)
        self.maxDurationRecording = maxDuration_recording
        self.maxDurationProgramActive = maxDuration_programActive
        self.consecutiveIdleTime = consecutiveIdleTime
        self.modelUnloadTimeout = model_unloadTimeout
        self.silenceSkip_threshold = silenceSkip_threshold  # Renamed from lowLoudnessSkip_threshold
        self.dictationModeSilenceThreshold = dictationMode_silenceLoudnessThreshold  # For chunk analysis in dictationMode

        self.commonFalseDetectedWords = commonFalseDetectedWords if commonFalseDetectedWords else []
        self.loudnessThresholdOfCommonFalseDetectedWords = loudnessThresholdOf_commonFalseDetectedWords

        self.scriptDir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.audioFiles = {  # Developer: Map notification names to file paths
            "modelUnloaded": str(self.scriptDir / "modelUnloaded.mp3"),
            "outputDisabled": str(self.scriptDir / "outputDisabled.mp3"),
            "outputEnabled": str(self.scriptDir / "outputEnabled.mp3"),
            "recordingOff": str(self.scriptDir / "recordingOff.mp3"),
            "recordingOn": str(self.scriptDir / "recordingOn.mp3")
        }
        for name, path in self.audioFiles.items():
            if not Path(path).is_file():
                print(f"Warning: Notification sound file not found: {path}")
        try:
            pygame.mixer.init()
        except pygame.error as e:
            print(f"Warning: Failed to initialize pygame mixer: {e}. Notification sounds disabled.")
            self.audioFiles = {}  # Disable sounds if mixer fails

        self.recordingToggleKey = recordingToggleKey
        self.outputToggleKey = outputToggleKey
        self.playEnableSounds = playEnableSounds  # User setting for enabling sounds
        self.enablingSounds = {"outputEnabled",
                               "recordingOn"}  # Developer: Sounds affected by playEnableSounds

        self.audioQueue = queue.Queue()

        self.isRecordingActive = isRecordingActive  # Current recording state (toggled by hotkey)
        self.isProgramActive = isProgramActive  # Overall program loop control
        self.outputEnabled = outputEnabled  # Text output (typing) state (toggled by hotkey)
        self.lastActivityTime = time.time()  # Timestamp of last significant activity (used for model unload)

        self.audioBuffer = np.array([],
                                    dtype=np.float32)  # Accumulates audio chunks for a Transcription Window
        self.recordingStartTime = 0  # Timestamp when the current recording session started
        self.lastTranscriptionTime = 0  # Timestamp of the last transcription attempt/trigger
        self.actualSampleRate = self.sampleRate  # Developer: Actual rate used by device (set in setupDeviceInfo)
        self.actualChannels = self.channels  # Developer: Actual channels used by device (set in setupDeviceInfo)

        self.lastValidTranscriptionTime = time.time()

        self.isCurrentlySpeakingFlag = False  # Tracks if recent audio chunks suggest speech
        self.silence_start_time = None  # Timestamp marking when silence began *after* speech

        self._printAudioDevices()

    def _printAudioDevices(self):
        """Developer: Helper function to print available audio devices."""
        print("--- Available Audio Devices ---")
        try:
            devices = sd.query_devices()
            print(devices)
        except Exception as e:
            print(f"Could not query audio devices: {e}")
        print("-----------------------------")

    def playNotification(self, soundName):
        """Developer: Plays a notification sound if available and enabled."""
        if not self.playEnableSounds and soundName in self.enablingSounds:
            self._debugPrint(f"Skipping enabling sound: {soundName}")
            return
        if soundName in self.audioFiles:
            sound_path = self.audioFiles[soundName]
            if sound_path:  # Check if path is not empty (in case init failed)
                try:
                    sound = pygame.mixer.Sound(sound_path)
                    sound.play()
                    self._debugPrint(f"Played notification sound: {soundName}")
                except Exception as e:
                    print(f"Error playing notification sound '{sound_path}': {e}")

    def audioCallback(self, inData, frames, timeInfo, status):
        """
        Developer: Callback function executed by the sounddevice stream in a separate thread.
        Receives raw audio data chunks and puts them into a queue for the main thread.
        """
        if status:
            print(f"Audio callback status: {status}")
        if self.isRecordingActive:
            self.audioQueue.put(inData.copy())

    def toggleOutput(self):
        """User/Developer: Toggles the text output (typing) feature on/off."""
        self.outputEnabled = not self.outputEnabled
        status = 'enabled' if self.outputEnabled else 'disabled'
        notification = "outputEnabled" if self.outputEnabled else "outputDisabled"
        self.playNotification(notification)
        print(f"Output {status}")
        self._debugPrint(f"Output toggled to: {self.outputEnabled}")

    def startRecording(self):
        """User/Developer: Activates recording, loads model if needed, resets state."""
        if self.isRecordingActive:  # Avoid redundant starts
            self._debugPrint("Recording already active.")
            return

        self.isRecordingActive = True
        self.lastActivityTime = time.time()  # Update activity time for model manager
        self.recordingStartTime = time.time()  # Mark start of this recording session
        self.lastValidTranscriptionTime = time.time()  # Reset idle timer
        self._debugPrint("Attempting to start recording...")

        if not self.modelLoaded:
            self.loadModel()

        self.isCurrentlySpeakingFlag = False
        self.silence_start_time = None
        self.audioBuffer = np.array([], dtype=np.float32)  # Clear buffer for new session

        print("Recording started...")
        self.playNotification("recordingOn")

    def stopRecording(self):
        """User/Developer: Deactivates recording."""
        if not self.isRecordingActive:  # Avoid redundant stops
            self._debugPrint("Recording already stopped.")
            return

        self.isRecordingActive = False
        self._debugPrint("Attempting to stop recording...")
        self.playNotification("recordingOff")
        print("Recording stopped.")

    def monitorKeyboardShortcuts(self):
        """
        Developer: Runs in a separate thread to monitor global hotkeys
        for toggling recording and output.
        """
        self._debugPrint("Keyboard shortcut monitor thread started.")
        thread_start_time = time.time()
        while self.isProgramActive and (
                time.time() - thread_start_time) < self.maxDurationProgramActive:
            try:
                if keyboard.is_pressed(self.recordingToggleKey):
                    self._debugPrint(f"'{self.recordingToggleKey}' pressed.")
                    if self.isRecordingActive:
                        self.stopRecording()
                    else:
                        self.startRecording()
                    while keyboard.is_pressed(self.recordingToggleKey):
                        time.sleep(0.1)
                    self._debugPrint(f"'{self.recordingToggleKey}' released.")

                if keyboard.is_pressed(self.outputToggleKey):
                    self._debugPrint(f"'{self.outputToggleKey}' pressed.")
                    self.toggleOutput()
                    while keyboard.is_pressed(self.outputToggleKey):
                        time.sleep(0.1)
                    self._debugPrint(f"'{self.outputToggleKey}' released.")

            except Exception as e:
                print(f"Error in keyboard monitoring thread: {e}")
                time.sleep(1)

            time.sleep(0.05)  # Slightly less frequent check than main loop

        self.isProgramActive = False  # Ensure program stops if thread exits due to timeout
        self._debugPrint("Keyboard shortcut monitor thread stopping.")

    def modelManager(self):
        """
        Developer: Runs in a separate thread to automatically unload the ASR model
        after a period of inactivity (recording stopped) to conserve resources.
        Also ensures model is loaded if recording becomes active.
        """
        self._debugPrint("Model manager thread started.")
        while self.isProgramActive:
            currentTime = time.time()

            if not self.isRecordingActive and self.modelLoaded:
                if (currentTime - self.lastActivityTime) >= self.modelUnloadTimeout:
                    self._debugPrint(
                        f"Model inactive for {currentTime - self.lastActivityTime:.1f}s (>= {self.modelUnloadTimeout}s), unloading...")
                    self.unloadModel()
                    self.playNotification("modelUnloaded")  # Notify user
                else:
                    pass

            if self.isRecordingActive and not self.modelLoaded:
                self._debugPrint("Recording active but model not loaded. Triggering load...")
                self.loadModel()
                self.lastActivityTime = time.time()

            time.sleep(10)

        self._debugPrint("Model manager thread stopping.")

    def setupDeviceInfo(self, deviceId=None):
        """
        Developer: Queries audio device information and sets actual sample rate/channels.
        Prints setup information for the user.
        """
        self._debugPrint(f"Setting up device info. Requested device ID: {deviceId}")
        try:
            if deviceId is not None:
                deviceInfo = sd.query_devices(deviceId)
                self._debugPrint(f"Device info for ID {deviceId}: {deviceInfo}")
                self.actualSampleRate = int(deviceInfo.get("default_samplerate", self.sampleRate))
                self.actualChannels = min(self.channels,
                                          int(deviceInfo.get("max_input_channels", self.channels)))
            else:
                self._debugPrint("Using default audio device.")
                default_device_info = sd.query_devices(kind='input')
                if default_device_info and isinstance(default_device_info, dict):
                    self.actualSampleRate = int(
                        default_device_info.get("default_samplerate", self.sampleRate))
                    self.actualChannels = min(self.channels,
                                              int(default_device_info.get("max_input_channels",
                                                                          self.channels)))
                    self._debugPrint(f"Default input device info: {default_device_info}")
                else:
                    self.actualSampleRate = self.sampleRate
                    self.actualChannels = self.channels

            if self.actualChannels < 1:
                print(
                    f"Warning: Could not determine valid input channels (got {self.actualChannels}), defaulting to 1.")
                self.actualChannels = 1

        except Exception as e:
            print(
                f"Warning: Could not query audio device information: {e}. Using configured defaults.")
            self.actualSampleRate = self.sampleRate
            self.actualChannels = self.channels

        print(f"\n--- Audio Setup ---")
        print(f"Mode: {self.transcriptionMode}")
        print(f"Using sample rate: {self.actualSampleRate} Hz")
        print(f"Using channels: {self.actualChannels}")
        print(f"ASR Model: {self.modelName} on {self.device}")
        print(f"--- Instructions ---")
        print(f"Press '{self.recordingToggleKey}' to toggle recording.")
        print(f"Press '{self.outputToggleKey}' to toggle text output (typing).")
        print(f"--- Timeouts ---")
        print(f"Max recording duration per session: {self.maxDurationRecording} s")
        print(f"Stop recording after silence: {self.consecutiveIdleTime} s")
        print(f"Unload model after inactivity: {self.modelUnloadTimeout} s")
        print(f"Program auto-exit after inactivity: {self.maxDurationProgramActive} s")
        print(f"------------------\n")

    def startThreads(self):
        """Developer: Starts the background threads for monitoring hotkeys and model state."""
        self._debugPrint("Starting background threads...")

        keyboardThread = threading.Thread(target=self.monitorKeyboardShortcuts,
                                          name="KeyboardMonitorThread")
        keyboardThread.daemon = True  # Developer: Allow program to exit even if thread is running
        keyboardThread.start()

        modelThread = threading.Thread(target=self.modelManager, name="ModelManagerThread")
        modelThread.daemon = True
        modelThread.start()

        self._debugPrint("Background threads started.")

    def processAudioChunks(self):
        """
        Developer: Processes audio chunks from the queue populated by the audio callback.
        Converts audio to mono, updates the speaking state for 'dictationMode',
        and appends the chunk to the main audio buffer.
        """
        processed_chunk = False  # Developer: Flag indicates if any chunks were processed in this call
        while not self.audioQueue.empty():
            audioChunk = self.audioQueue.get()

            if self.actualChannels > 1:
                mono_chunk = np.mean(audioChunk, axis=1).flatten()
            else:
                mono_chunk = audioChunk.flatten()

            if self.isRecordingActive and self.transcriptionMode == "dictationMode":
                chunk_loudness = np.mean(np.abs(mono_chunk))

                if self.debugPrint:
                    print(
                        f"DEBUG (Chunk): Loudness={chunk_loudness:.6f}, Threshold={self.dictationModeSilenceThreshold:.6f}")

                if chunk_loudness >= self.dictationModeSilenceThreshold:
                    if self.debugPrint:
                        print(
                            f"DEBUG: Speech detected (Loudness {chunk_loudness:.6f} >= {self.dictationModeSilenceThreshold:.6f})")
                    self.isCurrentlySpeakingFlag = True
                    self.silence_start_time = None
                else:
                    if self.isCurrentlySpeakingFlag:
                        if self.silence_start_time is None:
                            if self.debugPrint:
                                print(
                                    f"DEBUG: Silence detected after speech (Loudness {chunk_loudness:.6f}), starting silence timer ({self.dictationMode_silenceDurationToOutput}s)")
                            self.silence_start_time = time.time()

            self.audioBuffer = np.concatenate((self.audioBuffer, mono_chunk))
            processed_chunk = True

        return processed_chunk  # Developer: Return flag (currently unused, but potentially useful)

    def handleRecordingTiming(self):
        """Developer: Checks and enforces the maximum recording duration per session."""
        if self.isRecordingActive:
            if self.recordingStartTime == 0:
                self.recordingStartTime = time.time()
                self.lastActivityTime = time.time()  # Ensure activity time is updated

            currentTime = time.time()
            elapsedRecordingTime = currentTime - self.recordingStartTime
            if elapsedRecordingTime >= self.maxDurationRecording:
                print(
                    f"\nRecording reached maximum duration of {self.maxDurationRecording} seconds, stopping...")
                self.stopRecording()
                self.recordingStartTime = 0

    def transcribeAudio(self):
        """
        Developer: Determines when to trigger transcription based on the selected mode,
        prepares the audio data (Transcription Window), performs loudness checks/skips,
        calls the base class for ASR, and initiates output handling.
        """
        if not self.isRecordingActive:
            return

        currentTime = time.time()
        audioData = None  # Developer: Holds the Transcription Window data
        perform_transcription = False  # Developer: Flag to trigger the ASR call and output handling

        if self.transcriptionMode == "constantIntervalMode":
            if (
                    currentTime - self.lastTranscriptionTime) >= self.constantIntervalMode_transcriptionInterval:
                if len(self.audioBuffer) > 0:
                    audioData = self.audioBuffer.copy()  # Get the Transcription Window
                    self.audioBuffer = np.array([],
                                                dtype=np.float32)  # Clear buffer for next interval
                    self.lastTranscriptionTime = currentTime  # Reset interval timer
                    perform_transcription = True
                    if self.debugPrint:
                        print(
                            f"DEBUG (constantIntervalMode): Interval reached. Processing buffer of {len(audioData)} samples.")
                else:
                    self.lastTranscriptionTime = currentTime
                    if self.debugPrint:
                        print(
                            "DEBUG (constantIntervalMode): Interval reached. Buffer empty, resetting timer.")

        elif self.transcriptionMode == "dictationMode":
            if self.isCurrentlySpeakingFlag and self.silence_start_time is not None:
                elapsed_silence = currentTime - self.silence_start_time
                if elapsed_silence >= self.dictationMode_silenceDurationToOutput:
                    if self.debugPrint:
                        print(
                            f"DEBUG (dictationMode): Silence duration ({elapsed_silence:.2f}s) >= threshold ({self.dictationMode_silenceDurationToOutput}s). Triggering transcription.")

                    if len(self.audioBuffer) > 0:
                        audioData = self.audioBuffer.copy()  # Get the Transcription Window
                        self.audioBuffer = np.array([],
                                                    dtype=np.float32)  # Clear buffer for next utterance
                        self.lastTranscriptionTime = currentTime  # Mark transcription time
                        perform_transcription = True

                        self.isCurrentlySpeakingFlag = False
                        self.silence_start_time = None
                        if self.debugPrint:
                            print("DEBUG (dictationMode): State reset after transcription trigger.")
                    else:
                        if self.debugPrint:
                            print(
                                "DEBUG (dictationMode): Silence trigger met, but buffer is empty. Resetting state without transcription.")
                        self.isCurrentlySpeakingFlag = False
                        self.silence_start_time = None

        if perform_transcription and audioData is not None and len(audioData) > 0:
            segment_duration = len(audioData) / self.actualSampleRate
            if self.debugPrint:
                print(f"DEBUG: Preparing Transcription Window. Duration: {segment_duration:.2f}s")

            segmentMean = np.mean(np.abs(audioData))
            if self.debugPrint:
                print(f"DEBUG: Transcription Window Avg Loudness = {segmentMean:.6f}")

            perform_asr = True  # Developer: Default is to perform ASR
            if segmentMean < self.silenceSkip_threshold:
                trailing_samples = 0
                if self.skipSilence_afterNSecSilence > 0:
                    trailing_samples = int(
                        self.skipSilence_afterNSecSilence * self.actualSampleRate)

                if trailing_samples > 0 and len(audioData) >= trailing_samples:
                    trailing_audio = audioData[-trailing_samples:]
                    trailing_loudness = np.mean(np.abs(trailing_audio))

                    if trailing_loudness >= self.dictationModeSilenceThreshold:
                        perform_asr = True
                        if self.debugPrint:
                            print(
                                f"DEBUG: Overriding silence skip: segmentMean ({segmentMean:.6f} < {self.silenceSkip_threshold:.6f}) "
                                f"but trailing {self.skipSilence_afterNSecSilence}s loudness ({trailing_loudness:.6f}) >= dictationModeSilenceThreshold ({self.dictationModeSilenceThreshold:.6f})")
                    else:
                        perform_asr = False
                        if self.debugPrint:
                            print(
                                f"DEBUG: Silence skip CONFIRMED: segmentMean ({segmentMean:.6f} < {self.silenceSkip_threshold:.6f}) "
                                f"AND trailing {self.skipSilence_afterNSecSilence}s loudness ({trailing_loudness:.6f}) < dictationModeSilenceThreshold ({self.dictationModeSilenceThreshold:.6f})")
                else:
                    perform_asr = False
                    if self.debugPrint:
                        print(
                            f"DEBUG: Silence skip: segmentMean ({segmentMean:.6f} < {self.silenceSkip_threshold:.6f}). "
                            f"(Segment shorter than {self.skipSilence_afterNSecSilence}s or trailing check disabled).")

            transcription = ""  # Developer: Initialize transcription for this segment
            if perform_asr:
                if self.debugPrint:
                    if not (
                            segmentMean < self.silenceSkip_threshold):  # Standard case (loud enough)
                        print(
                            f"DEBUG: Segment mean loudness ({segmentMean:.6f}) >= silenceSkip_threshold ({self.silenceSkip_threshold:.6f}). Proceeding to ASR.")
                    else:  # Skip was overridden case
                        print(
                            f"DEBUG: Proceeding to ASR despite low segment mean due to louder trailing audio.")

                transcription = super().transcribeAudio(audioData, self.actualSampleRate)
            else:
                if self.debugPrint:
                    print("DEBUG: Skipping ASR call due to low loudness conditions.")
                transcription = ""  # Ensure transcription is empty if skipped

            self.processAndHandleTranscription(transcription, segmentMean)

            self.lastActivityTime = currentTime

        elif perform_transcription and (audioData is None or len(audioData) == 0):
            if self.debugPrint:
                print(
                    "DEBUG: Transcription triggered but audioData is None or empty. Skipping output handling.")

    def processAndHandleTranscription(self, transcription, loudnessValue):
        """
        Developer: Processes the transcription result after ASR.
        Handles filtering of common false words based on loudness,
        prints output, types output via PyAutoGUI if enabled,
        and manages the idle timeout check.
        """
        now = time.time()

        cleaned = transcription.strip().lower()
        isEmpty = (cleaned == "") or (cleaned == ".")  # Check if result is effectively empty
        isFalseWord = cleaned in [w.lower() for w in self.commonFalseDetectedWords]
        loudnessThresh = self.loudnessThresholdOfCommonFalseDetectedWords
        isBelowLoudness = loudnessValue < loudnessThresh
        isFalseDetection = isFalseWord and isBelowLoudness

        if isFalseDetection and self.debugPrint:
            print(
                f"DEBUG: Filtering false word '{cleaned}'. Segment loudness {loudnessValue:.6f} < threshold {loudnessThresh:.6f}")

        if not isEmpty and not isFalseDetection:
            final_transcription = transcription.lstrip(" ")  # Remove leading space for typing
            print("Transcription:", final_transcription)  # User: Show transcription

            if self.outputEnabled and not keyboard.is_pressed("ctrl"):
                try:
                    pyautogui.write(final_transcription + " ")  # Add space after typing
                except Exception as e:
                    print(f"Warning: PyAutoGUI write failed: {e}")

            self.lastValidTranscriptionTime = now
            if self.debugPrint:
                print(f"DEBUG: Valid speech detected. Resetting idle timer.")
            return  # Developer: Exit function after handling valid speech

        silentFor = now - self.lastValidTranscriptionTime  # Calculate time since last valid output
        if self.debugPrint:
            print(
                f"DEBUG: No valid output. Time since last valid speech: {silentFor:.1f}s (Idle timeout: {self.consecutiveIdleTime}s)")

        if silentFor >= self.consecutiveIdleTime:
            print(
                f"\nReached {self.consecutiveIdleTime} seconds of effective silence, stopping recording...")
            self.stopRecording()
            self.recordingStartTime = 0  # Reset recording start time

    def cleanupInactiveRecording(self):
        """Developer: Clears the audio buffer if recording is stopped but buffer might contain data."""
        if not self.isRecordingActive and len(self.audioBuffer) > 0:
            self._debugPrint(f"Clearing {len(self.audioBuffer)} samples from inactive buffer.")
            self.audioBuffer = np.array([], dtype=np.float32)

    def run(self, deviceId=None):
        """
        User/Developer: Main execution loop for the real-time transcriber.
        Initializes components, starts threads, manages the audio stream,
        and orchestrates the processing loop.
        """
        try:
            print("Initializing real-time transcriber...")
            print("Warming up ASR model...")
            self.loadModel()
            if not self.modelLoaded:
                print("!!! CRITICAL ERROR: Model failed to load. Exiting. !!!")
                return  # Exit if model loading failed

            self.setupDeviceInfo(deviceId)

            self.startThreads()

            print("Starting audio stream...")
            with sd.InputStream(samplerate=self.actualSampleRate,
                                channels=self.actualChannels,
                                device=deviceId,
                                blocksize=self.blockSize,  # Use configured block size
                                callback=self.audioCallback):  # Route audio to callback

                self.recordingStartTime = time.time() if self.isRecordingActive else 0
                self.lastTranscriptionTime = time.time()  # Initialize timer for constantIntervalMode
                print("Ready. Waiting for input or hotkeys...")

                while self.isProgramActive:
                    self.handleRecordingTiming()

                    self.processAudioChunks()

                    self.transcribeAudio()

                    self.cleanupInactiveRecording()

                    time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received. Stopping...")
            self.isProgramActive = False  # Signal threads to stop
        except Exception as e:
            print(f"\n!!! UNEXPECTED ERROR in main loop: {e}")
            import traceback
            print(traceback.format_exc())  # Print detailed traceback for debugging
            self.isProgramActive = False  # Signal threads to stop

        finally:
            print("Stopping program...")
            self.isRecordingActive = False  # Ensure recording flag is off
            self.isProgramActive = False  # Ensure loop flag is off
            super().cleanup()
            if pygame.mixer.get_init():
                pygame.mixer.quit()
            print("Program stopped.")


if __name__ == "__main__":

    print("Starting SpeechToText Transcriber script...")
    try:
        transcriber = SpeechToTextTranscriber(

            modelName="nvidia/canary-180m-flash",
            language="en",  # User: Set transcription language
            onlyCpu=False,  # User: Set to False to try using GPU (if available), True to force CPU

            transcriptionMode="dictationMode",  # User: "dictationMode" or "constantIntervalMode"

            dictationMode_silenceDurationToOutput=0.6,
            dictationMode_silenceLoudnessThreshold=0.0004,

            constantIntervalMode_transcriptionInterval=4,

            silenceSkip_threshold=0.0002,
            skipSilence_afterNSecSilence=0.3,
            commonFalseDetectedWords=["you", "thank you", "bye", 'amen'],
            loudnessThresholdOf_commonFalseDetectedWords=0.0008,

            removeTrailingDots=True,  # User: Clean trailing dots from output
            outputEnabled=False,  # User: Start with text output (typing) disabled
            isRecordingActive=True,  # User: Start with recording disabled (wait for hotkey)
            playEnableSounds=False,  # User: Disable sounds for 'recording on' / 'output enabled'

            recordingToggleKey="win+alt+l",  # User: Hotkey to toggle recording on/off.
            outputToggleKey="ctrl+q",  # User: Set your preferred hotkey for output toggle

            maxDuration_recording=10000,
            maxDuration_programActive=2 * 60 * 60,
            model_unloadTimeout=10 * 60,  # User: Unload model after 10 mins of inactivity
            consecutiveIdleTime=2 * 60,  # User: Stop recording after 2 mins of silence

            sampleRate=16000,  # User/Developer: Standard rate for Whisper
            channels=1,  # User/Developer: Use mono audio

            debugPrint=True  # Developer: Enable verbose console logs
        )

        transcriber.run()

    except Exception as e:
        print(f"\n!!! PROGRAM CRITICAL ERROR: {e}")
        import traceback

        print(traceback.format_exc())
