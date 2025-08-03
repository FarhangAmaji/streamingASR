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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.asr = None
        self.modelLoaded = False

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
            audioData (numpy.ndarray): Audio data to transcribe
            sampleRate (int): Sample rate of the audio data

        Returns:
            str: Transcribed text
        """
        if not self.modelLoaded:
            self.loadModel()

        if len(audioData.shape) > 1:
            audioData = np.mean(audioData, axis=1)

        result = self.asr({"raw": audioData, "sampling_rate": sampleRate})
        transcription = result["text"]

        if self.removeTrailingDots:
            transcription = transcription.rstrip('.')

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
                 transcriptionInterval=3,
                 busyContinuousTime=0.6,
                 transcriptionMode="constantInterval",
                 maxDuration_recording=10000,
                 maxDuration_programActive=60 * 60,
                 model_unloadTimeout=5 * 60,
                 consecutiveIdleTime=100,
                 isRecordingActive=True,
                 isProgramActive=True,
                 outputEnabled=False,
                 sampleRate=16000,
                 lowLoudnessSkip_threshold=4,
                 busyContinuousSilenceThreshold=2,
                 channels=1,
                 removeTrailingDots=True,
                 language="en",
                 commonFalseDetectedWords=None,
                 loudnessThresholdOf_commonFalseDetectedWords=2.4,
                 playEnableSounds=True,
                 debugPrint=False,
                 recordingToggleKey="win+alt+l",
                 outputToggleKey="ctrl+q"):
        """
        Initialize the real-time speech-to-text transcriber.

        Args:
            modelName (str): Name of the Whisper model to use
            transcriptionInterval (int): Interval for transcription processing
            maxDuration_recording (int): Maximum duration for a single recording session
            maxDuration_programActive (int): Maximum duration for program activity
            model_unloadTimeout (int): Timeout for unloading model when inactive
            consecutiveIdleTime (int): Time of silence before stopping recording
            isRecordingActive (bool): Initial recording state
            isProgramActive (bool): Initial program state
            outputEnabled (bool): Initial output state
            sampleRate (int): Audio sample rate
            channels (int): Number of audio channels
            removeTrailingDots (bool): Whether to remove trailing dots from transcriptions
            language (str): Language code for transcription
            commonFalseDetectedWords (list): List of commonly falsely detected words
            loudnessThresholdOf_commonFalseDetectedWords (float): Loudness threshold for false detection
            debugPrint (bool): Enable debug printing for memory monitoring
            recordingToggleKey (str): Key combination to toggle recording
            outputToggleKey (str): Key to toggle output
        """
        super().__init__(modelName=modelName,
                         language=language,
                         removeTrailingDots=removeTrailingDots,
                         debugPrint=debugPrint)

        self.transcriptionMode = transcriptionMode  # "constantInterval" | "busyContinuous"
        self.busyContinuousTime = busyContinuousTime  # seconds to look-back for silence

        self.sampleRate = sampleRate
        self.channels = channels
        self.blockSize = 1024  # Number of frames per block
        self.transcriptionInterval = transcriptionInterval
        self.maxDurationRecording = maxDuration_recording
        self.maxDurationProgramActive = maxDuration_programActive
        self.consecutiveIdleTime = consecutiveIdleTime
        self.modelUnloadTimeout = model_unloadTimeout
        self.lowLoudnessSkip_threshold = lowLoudnessSkip_threshold
        self.busyContinuousSilenceThreshold = busyContinuousSilenceThreshold

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

        self.recordingToggleKey = recordingToggleKey
        self.outputToggleKey = outputToggleKey
        self.playEnableSounds = playEnableSounds
        self.enablingSounds = {"outputEnabled", "recordingOn"}

        pygame.mixer.init()

        self.audioQueue = queue.Queue()

        self.isRecordingActive = isRecordingActive
        self.isProgramActive = isProgramActive
        self.outputEnabled = outputEnabled
        self.lastActivityTime = time.time()

        self.audioBuffer = np.array([], dtype=np.float32)
        self.actualSampleRate = self.sampleRate
        self.maxBufferSize = int(
            busyContinuousTime * 2 * self.actualSampleRate)  # to maintain context
        self.emptyTranscriptionCount = 0
        self.recordingStartTime = 0
        self.lastTranscriptionTime = 0
        self.actualChannels = self.channels
        self.lastProcessedIndex = 0
        self.silenceStartTime = None  # Tracks start of *current* silent stretch
        self.lastSoundTime = 0  # Marks time of last detected sound

        self.lastValidTranscriptionTime = time.time()

        print("Available audio devices:")
        devices = sd.query_devices()
        print(devices)

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
        self.recordingStartTime = 0  # <-- Optional: reset session timer
        self.lastValidTranscriptionTime = time.time()  # ✅ Reset silence timer
        if not self.modelLoaded:  # ensure model is ready
            self.loadModel()
        print("Recording started...")

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
        """Process audio chunks from the queue."""
        while not self.audioQueue.empty():
            audioChunk = self.audioQueue.get()
            if self.actualChannels > 1:
                audioChunk = np.mean(audioChunk, axis=1)  # Convert stereo to mono
            self.audioBuffer = np.concatenate((self.audioBuffer, audioChunk.flatten()))

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
        if not self.isRecordingActive or len(self.audioBuffer) == 0:
            return

        now = time.time()
        audioData = None

        if self.transcriptionMode == "constantInterval":
            if (now - self.lastTranscriptionTime) >= self.transcriptionInterval:
                audioData = self.audioBuffer.copy()
                self.audioBuffer = np.array([], dtype=np.float32)
                self.lastTranscriptionTime = now

        elif self.transcriptionMode == "busyContinuous":
            newAudio = self.audioBuffer[self.lastProcessedIndex:]

            if len(newAudio) > int(0.1 * self.actualSampleRate):  # Process >100ms
                frameSize = int(0.1 * self.actualSampleRate)
                frames = [newAudio[i:i + frameSize] for i in range(0, len(newAudio), frameSize)]
                rmsValues = [np.sqrt(np.mean(np.square(f))) if len(f) == frameSize else 0 for f in
                             frames]

                frameTime = 0
                for i, rms in enumerate(rmsValues):
                    frameTime = i * 0.1
                    absoluteFrameTime = now - frameTime

                    if rms >= self.busyContinuousSilenceThreshold:
                        self.lastSoundTime = absoluteFrameTime
                        self.silenceStartTime = None
                    else:
                        if self.silenceStartTime is None:
                            self.silenceStartTime = absoluteFrameTime

                        if absoluteFrameTime <= self.lastSoundTime + self.busyContinuousTime:
                            continue

                        finalFrameIdx = self.lastProcessedIndex + i * frameSize
                        audioData = self.audioBuffer[:finalFrameIdx].copy()

                        self.lastProcessedIndex = finalFrameIdx
                        self.silenceStartTime = None
                        self.lastTranscriptionTime = now
                        break

                if self.lastProcessedIndex < len(self.audioBuffer):
                    self.audioBuffer = self.audioBuffer[self.lastProcessedIndex:]
                self.lastProcessedIndex = 0

                if len(self.audioBuffer) > self.maxBufferSize:
                    self.audioBuffer = self.audioBuffer[-self.maxBufferSize:]

        if audioData is None:
            return

        segmentMean = np.mean(np.abs(audioData))
        segmentDuration = len(audioData) / self.actualSampleRate
        if segmentMean < (self.lowLoudnessSkip_threshold * segmentDuration):
            transcription = ""
            self._debugPrint(f"lower than loudness threshold {segmentMean}")
        else:
            transcription = super().transcribeAudio(audioData, self.actualSampleRate)

        self.handleTranscriptionOutput(transcription, segmentMean)
        self.lastActivityTime = now  # ← Fixed: Use 'now', not 'currentTime'

    def handleTranscriptionOutput(self, transcription, loudnessValue):
        """Post-process transcription and manage silence time-out."""
        now = time.time()
        effectiveInterval = (
            self.busyContinuousTime
            if self.transcriptionMode == "busyContinuous"
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
            transcriptionMode="busyContinuous",  # |constantInterval
            transcriptionInterval=4,  # Longer interval between transcriptions
            busyContinuousTime=6,
            commonFalseDetectedWords=["you", "thank you", "bye", 'amen', "you"],
            loudnessThresholdOf_commonFalseDetectedWords=32,  # ← Updated
            lowLoudnessSkip_threshold=0,
            playEnableSounds=False,
            busyContinuousSilenceThreshold=3.5,  # ← Updated
            maxDuration_recording=10000,  # 10000s max recording
            maxDuration_programActive=2 * 60 * 60,  # 1 hour program active time
            model_unloadTimeout=20 * 60,  # time to Unload model from gpu
            consecutiveIdleTime=3 * 60,  # Stop after n seconds of silence
            isRecordingActive=True,  # Start with recording off
            outputEnabled=False,  # Start with output off
            sampleRate=16000,  # Higher sample rate
            channels=1,
            debugPrint=True
        )

        transcriber.run()
    except Exception as e:
        print(f"Program error: {e}")
