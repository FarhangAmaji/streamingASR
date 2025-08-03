import gc
import os
import queue
import threading
import time
from pathlib import Path

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
    Handles common model management and basic transcription operations.
    """

    def __init__(self,
                 modelName="openai/whisper-large-v3",
                 language="en",
                 removeTrailingDots=True,
                 onlyCpu=False,
                 debugPrint=False):
        """
        Initialize the base transcriber with common parameters.

        Args:
            modelName (str): Name of the Whisper model to use
            language (str): Language code for transcription
            removeTrailingDots (bool): Whether to remove trailing dots from transcriptions
            debugPrint (bool): Enable debug printing for memory monitoring
        """
        self.modelName = modelName
        self.language = language
        self.removeTrailingDots = removeTrailingDots
        self.debugPrint = debugPrint
        self.asr = None
        self.modelLoaded = False

        if onlyCpu:
            self.device = torch.device('cpu')  # Force CPU, ignore CUDA availability
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _debugPrint(self, message):
        if self.debugPrint:
            print(message)

    def loadModel(self):
        """Load the ASR model to GPU."""
        if not self.modelLoaded:
            self._debugPrint("Loading model to GPU...")

            self._monitorMemory()
            self._cudaClean()

            self.asr = pipeline("automatic-speech-recognition",
                                model=self.modelName,
                                generate_kwargs={"language": self.language},
                                device=self.device)
            self.modelLoaded = True

            dummyAudio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            self.asr({"raw": dummyAudio, "sampling_rate": 16000})

            self._monitorMemory()

    def unloadModel(self):
        """Unload the ASR model from GPU."""
        if self.modelLoaded:
            self._debugPrint("Unloading model from GPU...")

            del self.asr
            self.asr = None

            self._cudaClean()

            self.modelLoaded = False
            self._monitorMemory()

    def _cudaClean(self):
        """Clean CUDA memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            with torch.no_grad():
                torch.cuda.synchronize()

    def _monitorMemory(self):
        """Monitor GPU memory usage."""
        if torch.cuda.is_available() and self.debugPrint:
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    def transcribeAudio(self, audioData, sampleRate):
        """
        Transcribe audio data using the loaded model.

        Args:
            audioData (numpy.ndarray): Audio data (as float32 NumPy array).
            sampleRate (int): Sample rate of the audio data (e.g., 16000).

        Returns:
            str: The transcribed text. Returns an empty string if transcription fails
                 or the model is not loaded.
        """
        if not self.modelLoaded:
            self._debugPrint("Transcription skipped: Model not loaded.")
            return ""  # Return empty if model isn't loaded

        if len(audioData.shape) > 1 and audioData.shape[1] > 1:
            if self.debugPrint:
                print(f"DEBUG: Converting stereo audio (shape {audioData.shape}) to mono.")
            audioData = np.mean(audioData, axis=1)  # Average channels for mono

        if audioData.dtype != np.float32:
            self._debugPrint(
                f"DEBUG: Converting audio data type from {audioData.dtype} to float32.")
            audioData = audioData.astype(np.float32)

        transcription = ""  # Default empty transcription
        try:
            asr_input = {"raw": audioData, "sampling_rate": sampleRate}
            self._debugPrint(
                f"DEBUG: Sending {len(audioData) / sampleRate:.2f}s audio to ASR pipeline...")
            result = self.asr(asr_input)
            self._debugPrint("DEBUG: ASR pipeline processing finished.")

            self._debugPrint(f"ASR Raw Result: {result}")

            if isinstance(result, dict) and "text" in result:
                transcription = result["text"]
            else:
                print(
                    f"Warning: Unexpected ASR result structure: {type(result)}. Could not extract text.")
                transcription = ""  # Assign empty string if text cannot be found

            if self.removeTrailingDots and isinstance(transcription, str):
                transcription = transcription.strip('. ')  # Also strip leading/trailing spaces

        except Exception as e:
            print(f"!!! ERROR during transcription: {e}")
            transcription = ""  # Return empty string on error

        if self.debugPrint:
            print(
                f"DEBUG: Transcription result (cleaned): '{transcription[:100]}...'")  # Print start of result

        return transcription

    def cleanup(self):
        """Clean up resources before exiting."""
        if self.modelLoaded:
            self.unloadModel()


class FileTranscriber(BaseTranscriber):
    """
    Class for transcribing audio files using the Whisper model.
    Inherits common model management from BaseTranscriber.
    """

    def __init__(self,
                 modelName="openai/whisper-large-v3",
                 language="en",
                 removeTrailingDots=True,
                 debugPrint=False):
        """
        Initialize the file transcriber.

        Args:
            modelName (str): Name of the Whisper model to use
            language (str): Language code for transcription
            removeTrailingDots (bool): Whether to remove trailing dots from transcriptions
            debugPrint (bool): Enable debug printing for memory monitoring
        """
        super().__init__(modelName=modelName,
                         language=language,
                         removeTrailingDots=removeTrailingDots,
                         debugPrint=debugPrint)

    def transcribeFile(self, audioFilePath, outputFilePath=None):
        """
        Transcribe an audio file and optionally save the transcription to a file.

        Args:
            audioFilePath (str): Path to the input audio file
            outputFilePath (str, optional): Path to save the transcription. If None, prints to console.

        Returns:
            str: Transcribed text, or None if transcription fails
        """
        try:
            audioData, sampleRate = sf.read(audioFilePath)

            transcription = self.transcribeAudio(audioData, sampleRate)

            if outputFilePath:
                with open(outputFilePath, 'w', encoding='utf-8') as outputFile:
                    outputFile.write(transcription)
                self._debugPrint(f"Transcription saved to: {outputFilePath}")
            else:
                print("Transcription:", transcription)

            return transcription

        except Exception as e:
            print(f"Error transcribing file: {e}")
            return None

    def cleanup(self):
        """
        Clean up resources before exiting.
        Calls base class cleanup method.
        """
        super().cleanup()
        self._debugPrint("File transcriber cleanup complete.")


class SpeechToTextTranscriber(BaseTranscriber):
    """
    Class for real-time speech-to-text transcription.
    Inherits common model management from BaseTranscriber and adds real-time specific functionality.
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
                 isRecordingActive=True,
                 isProgramActive=True,
                 outputEnabled=False,
                 sampleRate=16000,
                 channels=1,
                 removeTrailingDots=True,
                 playEnableSounds=True,
                 onlyCpu=False,
                 debugPrint=False,
                 recordingToggleKey="win+alt+l",
                 outputToggleKey="ctrl+q"):
        """
        Initialize the real-time speech-to-text transcriber.

        Args:
            modelName (str): Name of the Whisper model to use
            constantIntervalMode_transcriptionInterval (int): Interval for transcription processing (constantIntervalMode mode)
            dictationMode_silenceDurationToOutput (float): Duration of silence required after speech to trigger
                                        transcription (dictationMode mode).
            transcriptionMode (str): "constantIntervalMode" or "dictationMode".
            skipSilence_afterNSecSilence (float): If a transcription segment's overall loudness
                                                         is below 'lowLoudnessSkip_threshold', only skip it
                                                         if the average loudness of the *last* N seconds
                                                         (this value) was also below
                                                         'dictationModeSilenceThreshold'. Set to 0 to disable
                                                         this trailing check and revert to only checking overall loudness.
            maxDuration_recording (int): Maximum duration for a single recording session (seconds).
            maxDuration_programActive (int): Maximum duration for program activity (seconds).
            model_unloadTimeout (int): Timeout for unloading model when inactive (seconds).
            consecutiveIdleTime (int): Time of effective silence before stopping recording (seconds).
            isRecordingActive (bool): Initial recording state.
            isProgramActive (bool): Initial program state.
            outputEnabled (bool): Initial output state (typing transcription).
            sampleRate (int): Audio sample rate (Hz).
            silenceSkip_threshold (float): Average loudness threshold below which transcription
                                                of a segment is potentially skipped (see above).
            dictationMode_silenceLoudnessThreshold (float): Average loudness threshold below which an incoming
                                                    audio chunk is considered silent (dictationMode mode),
                                                    AND used for the trailing check of the low loudness skip feature.
            channels (int): Number of audio channels (1 for mono, 2 for stereo).
            removeTrailingDots (bool): Whether to remove trailing dots from transcriptions.
            language (str): Language code for transcription (e.g., "en", "es").
            commonFalseDetectedWords (list): List of commonly falsely detected words to filter.
            loudnessThresholdOf_commonFalseDetectedWords (float): Average loudness threshold below which
                                                                  common false words are ignored.
            playEnableSounds (bool): Allow playing sounds for enabling actions (recording on, output on).
            debugPrint (bool): Enable detailed debug printing.
            recordingToggleKey (str): Key combination to toggle recording (e.g., "win+alt+l").
            outputToggleKey (str): Key combination to toggle text output (e.g., "ctrl+q").
        """
        super().__init__(modelName=modelName, language=language,
                         removeTrailingDots=removeTrailingDots,
                         onlyCpu=onlyCpu,
                         debugPrint=debugPrint)

        self.transcriptionMode = transcriptionMode
        self.dictationMode_silenceDurationToOutput = dictationMode_silenceDurationToOutput
        self.skipSilence_afterNSecSilence = skipSilence_afterNSecSilence

        self.sampleRate = sampleRate
        self.channels = channels
        self.blockSize = 1024  # Number of frames per block (affects callback frequency)
        self.transcriptionInterval = constantIntervalMode_transcriptionInterval
        self.maxDurationRecording = maxDuration_recording
        self.maxDurationProgramActive = maxDuration_programActive
        self.consecutiveIdleTime = consecutiveIdleTime
        self.modelUnloadTimeout = model_unloadTimeout
        self.lowLoudnessSkip_threshold = silenceSkip_threshold
        self.dictationModeSilenceThreshold = dictationMode_silenceLoudnessThreshold

        self.commonFalseDetectedWords = commonFalseDetectedWords if commonFalseDetectedWords else []
        self.loudnessThresholdOfCommonFalseDetectedWords = loudnessThresholdOf_commonFalseDetectedWords

        self.scriptDir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.audioFiles = {
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
            self.audioFiles = {}

        self.recordingToggleKey = recordingToggleKey
        self.outputToggleKey = outputToggleKey
        self.playEnableSounds = playEnableSounds
        self.enablingSounds = {"outputEnabled", "recordingOn"}

        self.audioQueue = queue.Queue()

        self.isRecordingActive = isRecordingActive
        self.isProgramActive = isProgramActive
        self.outputEnabled = outputEnabled
        self.lastActivityTime = time.time()

        self.audioBuffer = np.array([], dtype=np.float32)
        self.emptyTranscriptionCount = 0
        self.recordingStartTime = 0
        self.lastTranscriptionTime = 0
        self.actualSampleRate = self.sampleRate
        self.actualChannels = self.channels

        self.lastValidTranscriptionTime = time.time()

        self.isCurrentlySpeakingFlag = False
        self.silence_start_time = None

        print("--- Available Audio Devices ---")
        try:
            devices = sd.query_devices()
            print(devices)
        except Exception as e:
            print(f"Could not query audio devices: {e}")
        print("-----------------------------")

    def playNotification(self, soundName):
        if not self.playEnableSounds and soundName in self.enablingSounds:
            return
        if soundName in self.audioFiles:
            try:
                sound = pygame.mixer.Sound(self.audioFiles[soundName])
                sound.play()
            except Exception as e:
                print(f"Error playing notification sound: {e}")

    def audioCallback(self, inData, frames, time, status):
        """Callback function for audio stream."""
        if status:
            print(f"Audio callback status: {status}")
        if self.isRecordingActive:
            self.audioQueue.put(inData.copy())

    def toggleOutput(self):
        """Toggle output when outputToggleKey is pressed."""
        self.outputEnabled = not self.outputEnabled
        if self.outputEnabled:
            self.playNotification("outputEnabled")
        else:
            self.playNotification("outputDisabled")
        print(f"Output {'enabled' if self.outputEnabled else 'disabled'}")

    def startRecording(self):
        """Start recording when recordingToggleKey is pressed."""
        self.isRecordingActive = True
        self.lastActivityTime = time.time()
        self.recordingStartTime = time.time()  # Reset session timer
        self.lastValidTranscriptionTime = time.time()  # Reset idle timeout timer
        if not self.modelLoaded:  # ensure model is ready
            self.loadModel()

        self.isCurrentlySpeakingFlag = False
        self.silence_start_time = None
        self.audioBuffer = np.array([], dtype=np.float32)  # Clear buffer on new start

        print("Recording started...")
        self.playNotification("recordingOn")

    def stopRecording(self):
        """Stop recording."""
        self.isRecordingActive = False
        self.playNotification("recordingOff")
        print("Recording stopped...")

    def monitorKeyboardShortcuts(self):
        """Monitor keyboard shortcuts."""
        startTime = time.time()

        while self.isProgramActive and (time.time() - startTime) < self.maxDurationProgramActive:
            if keyboard.is_pressed(self.recordingToggleKey):
                self.isRecordingActive = not self.isRecordingActive

                if self.isRecordingActive:
                    self.startRecording()
                    startTime = time.time()  # still needed to reset program timeout
                else:
                    self.stopRecording()

                while keyboard.is_pressed(self.recordingToggleKey):
                    time.sleep(0.1)

            if keyboard.is_pressed(self.outputToggleKey):
                self.toggleOutput()
                while keyboard.is_pressed(self.outputToggleKey):
                    time.sleep(0.1)

            time.sleep(0.1)

        self.isProgramActive = False
        print("Program timeout reached. Exiting...")

    def modelManager(self):
        """Monitor model usage and unload when inactive for too long."""
        while self.isProgramActive:
            currentTime = time.time()

            if not self.isRecordingActive and self.modelLoaded:
                if (currentTime - self.lastActivityTime) >= self.modelUnloadTimeout:
                    print(f"Model inactive for {self.modelUnloadTimeout} seconds, unloading...")
                    self.unloadModel()
                    self.playNotification("modelUnloaded")

            if self.isRecordingActive and not self.modelLoaded:
                self.loadModel()
                self.lastActivityTime = currentTime

            time.sleep(10)  # Check every 10 seconds

    def setupDeviceInfo(self, deviceId=None):
        """Set up audio device information."""
        if deviceId is not None:
            deviceInfo = sd.query_devices(deviceId)
            print(f"Device info: {deviceInfo}")
            self.actualSampleRate = deviceInfo.get("default_samplerate", self.sampleRate)
            self.actualChannels = min(self.channels,
                                      deviceInfo.get("max_input_channels", self.channels))
        else:
            self.actualSampleRate = self.sampleRate
            self.actualChannels = self.channels

        print(
            f"Audio processing started with sample rate {self.actualSampleRate} Hz, channels {self.actualChannels}...")
        print(
            f"Press '{self.recordingToggleKey}' to start recording (max {self.maxDurationRecording} seconds per session)")
        print(f"Press '{self.outputToggleKey}' to toggle text output")
        print(f"Recording will stop after {self.consecutiveIdleTime} seconds of silence")
        print(f"Program will exit after {self.maxDurationProgramActive} seconds of inactivity")
        print(f"Model will be unloaded after {self.modelUnloadTimeout} seconds of inactivity")

    def startThreads(self):
        """Start monitoring threads."""
        keyboardThread = threading.Thread(target=self.monitorKeyboardShortcuts)
        keyboardThread.daemon = True
        keyboardThread.start()

        modelThread = threading.Thread(target=self.modelManager)
        modelThread.daemon = True
        modelThread.start()

    def processAudioChunks(self):
        """
        Process audio chunks from the queue, update speaking state for dictationMode mode,
        and append chunks to the main audio buffer.
        """
        processed_chunk = False  # Flag to track if any work was done
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
                        f"DEBUG (Chunk): Loudness={chunk_loudness:.4f}, Threshold={self.dictationModeSilenceThreshold}")

                if chunk_loudness >= self.dictationModeSilenceThreshold:
                    if not self.isCurrentlySpeakingFlag:
                        if self.debugPrint:
                            print(
                                f"DEBUG: Speech detected (Loudness {chunk_loudness:.4f} >= {self.dictationModeSilenceThreshold})")
                        self.isCurrentlySpeakingFlag = True

                    self.silence_start_time = None

                else:

                    if self.isCurrentlySpeakingFlag:
                        if self.silence_start_time is None:
                            if self.debugPrint:
                                print(
                                    f"DEBUG: Silence detected after speech (Loudness {chunk_loudness:.4f}), starting silence timer ({self.dictationMode_silenceDurationToOutput}s)")
                            self.silence_start_time = time.time()

            self.audioBuffer = np.concatenate((self.audioBuffer, mono_chunk))
            processed_chunk = True  # Mark that at least one chunk was processed

        return processed_chunk

    def handleRecordingTiming(self):
        """Handle recording session timing."""
        if self.isRecordingActive:
            if self.recordingStartTime == 0:
                self.recordingStartTime = time.time()
                self.lastTranscriptionTime = self.recordingStartTime
                self.lastActivityTime = time.time()

            currentTime = time.time()
            if (currentTime - self.recordingStartTime) >= self.maxDurationRecording:
                print(
                    f"Recording reached maximum duration of {self.maxDurationRecording} seconds, stopping...")
                self.stopRecording()
                self.recordingStartTime = 0
                self.emptyTranscriptionCount = 0

    def transcribeAudio(self):
        """
        Handle transcription according to the selected mode.
        - constantIntervalMode: Transcribes fixed intervals based on 'transcriptionInterval'.
        - dictationMode: Accumulates audio while speech is detected and transcribes
                          the entire segment only after speech stops and a silence
                          period defined by 'dictationMode_silenceDurationToOutput' elapses.
        """
        if not self.isRecordingActive:
            return

        currentTime = time.time()
        audioData = None  # Holds the audio data segment to be transcribed
        perform_transcription = False  # Flag to indicate if transcription should proceed

        if self.transcriptionMode == "constantIntervalMode":
            if (currentTime - self.lastTranscriptionTime) >= self.transcriptionInterval:
                if len(self.audioBuffer) > 0:
                    audioData = self.audioBuffer.copy()
                    self.audioBuffer = np.array([], dtype=np.float32)
                    self.lastTranscriptionTime = currentTime  # Reset the timer for the next interval
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
                if self.debugPrint and elapsed_silence < self.dictationMode_silenceDurationToOutput:
                    pass  # Optional debug for silence timer progress

                if elapsed_silence >= self.dictationMode_silenceDurationToOutput:
                    if self.debugPrint:
                        print(
                            f"DEBUG (dictationMode): Silence duration ({elapsed_silence:.2f}s) >= threshold ({self.dictationMode_silenceDurationToOutput}s). Triggering transcription.")

                    if len(self.audioBuffer) > 0:
                        audioData = self.audioBuffer.copy()
                        self.audioBuffer = np.array([], dtype=np.float32)
                        self.lastTranscriptionTime = currentTime
                        perform_transcription = True

                        self.isCurrentlySpeakingFlag = False
                        self.silence_start_time = None
                        if self.debugPrint:
                            print(
                                "DEBUG (dictationMode): State reset after transcription trigger.")
                    else:
                        if self.debugPrint:
                            print(
                                "DEBUG (dictationMode): Silence trigger met, but buffer is empty. Resetting state without transcription.")
                        self.isCurrentlySpeakingFlag = False
                        self.silence_start_time = None

        if perform_transcription and audioData is not None and len(audioData) > 0:
            segment_duration = len(audioData) / self.actualSampleRate
            if self.debugPrint:
                print(f"DEBUG: Processing segment of duration {segment_duration:.2f}s")

            segmentMean = np.mean(np.abs(audioData))
            if self.debugPrint:
                print(f"DEBUG: Segment mean loudness = {segmentMean:.4f}")

            perform_asr = True  # Assume we will transcribe unless skip conditions met
            if segmentMean < self.lowLoudnessSkip_threshold:

                trailing_samples = 0
                if self.skipSilence_afterNSecSilence > 0:
                    trailing_samples = int(
                        self.skipSilence_afterNSecSilence * self.actualSampleRate)

                if trailing_samples > 0 and len(audioData) >= trailing_samples:
                    trailing_audio = audioData[-trailing_samples:]
                    trailing_loudness = np.mean(np.abs(trailing_audio))

                    if trailing_loudness >= self.dictationModeSilenceThreshold:
                        perform_asr = True  # Override the skip decision
                        if self.debugPrint:
                            print(
                                f"DEBUG: Overriding low loudness skip: segmentMean ({segmentMean:.4f} < {self.lowLoudnessSkip_threshold}) "
                                f"but trailing {self.skipSilence_afterNSecSilence}s loudness ({trailing_loudness:.4f}) >= dictationModeSilenceThreshold ({self.dictationModeSilenceThreshold})")
                    else:
                        perform_asr = False
                        if self.debugPrint:
                            print(
                                f"DEBUG: Low loudness skip CONFIRMED: segmentMean ({segmentMean:.4f} < {self.lowLoudnessSkip_threshold}) "
                                f"AND trailing {self.skipSilence_afterNSecSilence}s loudness ({trailing_loudness:.4f}) < dictationModeSilenceThreshold ({self.dictationModeSilenceThreshold})")
                else:
                    perform_asr = False
                    if self.debugPrint:
                        print(
                            f"DEBUG: Low loudness skip: segmentMean ({segmentMean:.4f} < {self.lowLoudnessSkip_threshold}). "
                            f"Segment shorter than {self.skipSilence_afterNSecSilence}s or trailing check disabled.")

            if perform_asr:
                if self.debugPrint and not (segmentMean < self.lowLoudnessSkip_threshold):
                    print(
                        f"DEBUG: Segment mean loudness ({segmentMean:.4f}) >= lowLoudnessSkip_threshold. Proceeding to ASR.")
                elif self.debugPrint and (segmentMean < self.lowLoudnessSkip_threshold):
                    print(
                        f"DEBUG: Proceeding to ASR despite low segment mean due to louder trailing audio.")

                transcription = super().transcribeAudio(audioData, self.actualSampleRate)
            else:
                transcription = ""  # Treat as empty transcription
                if self.debugPrint:
                    print("DEBUG: Skipping transcription due to low loudness conditions.")

            self.processAndHandleTranscription(transcription, segmentMean)

            self.lastActivityTime = currentTime
        elif perform_transcription and (audioData is None or len(audioData) == 0):
            if self.debugPrint:
                print(
                    "DEBUG: Transcription triggered but audioData is None or empty. Skipping output handling.")

    def processAndHandleTranscription(self, transcription, loudnessValue):
        """Post-process transcription and manage silence time-out."""
        now = time.time()
        effectiveInterval = (
            self.dictationMode_silenceDurationToOutput
            if self.transcriptionMode == "dictationMode"
            else self.transcriptionInterval
        )

        cleaned = transcription.strip().lower()
        isEmpty = (cleaned == "") or (cleaned == ".")
        isFalseWord = cleaned in [w.lower() for w in self.commonFalseDetectedWords]
        loudnessThresh = self.loudnessThresholdOfCommonFalseDetectedWords
        isBelow = loudnessValue < loudnessThresh
        isFalseDetection = isFalseWord and isBelow

        if not isEmpty and not isFalseDetection:
            print("Transcription:", transcription)

            if self.outputEnabled and not keyboard.is_pressed("ctrl"):
                pyautogui.write(transcription.lstrip(" ") + " ")

            self.emptyTranscriptionCount = 0  # optional, still reset
            self.lastValidTranscriptionTime = now  # ← mark last real speech
            return

        silentFor = now - self.lastValidTranscriptionTime
        if self.debugPrint:
            print(f"Silent for {silentFor:.1f}s "
                  f"(threshold {self.consecutiveIdleTime}s)")

        if silentFor >= self.consecutiveIdleTime:
            print(f"Reached {self.consecutiveIdleTime} seconds of silence, "
                  f"stopping recording...")
            self.stopRecording()
            self.recordingStartTime = 0
            self.emptyTranscriptionCount = 0

    def cleanupInactiveRecording(self):
        """Clean up when recording is inactive."""
        if not self.isRecordingActive and len(self.audioBuffer) > 0:
            self.audioBuffer = np.array([], dtype=np.float32)
            self.emptyTranscriptionCount = 0

    def run(self, deviceId=None):
        """
        Main method to run the transcriber.
        Continuously record audio, transcribe it, and type the transcription.
        """
        try:
            print("Warming up model...")
            self.loadModel()

            self.setupDeviceInfo(deviceId)

            self.startThreads()

            with sd.InputStream(samplerate=self.actualSampleRate,
                                channels=self.actualChannels,
                                device=deviceId,
                                blocksize=self.blockSize,
                                callback=self.audioCallback):

                self.recordingStartTime = 0
                self.lastTranscriptionTime = 0

                while self.isProgramActive:
                    self.handleRecordingTiming()

                    self.processAudioChunks()

                    self.transcribeAudio()

                    self.cleanupInactiveRecording()

                    time.sleep(0.01)

        except Exception as e:
            print(f"Error during audio processing: {e}")
            raise
        finally:
            self.isRecordingActive = False
            self.isProgramActive = False
            super().cleanup()
            print("Program stopped.")


if __name__ == "__main__":
    try:
        transcriber = SpeechToTextTranscriber(
            modelName="openai/whisper-large-v3",
            transcriptionMode="dictationMode",
            onlyCpu=True,
            constantIntervalMode_transcriptionInterval=4,
            dictationMode_silenceDurationToOutput=.6,
            commonFalseDetectedWords=["you", "thank you", "bye", 'amen'],
            loudnessThresholdOf_commonFalseDetectedWords=.0008,
            silenceSkip_threshold=0.0002,
            dictationMode_silenceLoudnessThreshold=.0004,
            playEnableSounds=False,
            maxDuration_recording=10000,
            maxDuration_programActive=2 * 60 * 60,
            model_unloadTimeout=20 * 60,
            consecutiveIdleTime=3 * 60,
            isRecordingActive=True,
            outputEnabled=False,
            sampleRate=16000,
            channels=1,
            debugPrint=True
        )

        transcriber.run()
    except Exception as e:
        print(f"Program error: {e}")
