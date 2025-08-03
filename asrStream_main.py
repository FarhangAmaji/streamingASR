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

            # Force garbage collection before loading
            self._monitorMemory()
            self._cudaClean()

            # Load the model with specific configuration
            self.asr = pipeline("automatic-speech-recognition",
                                model=self.modelName,
                                generate_kwargs={"language": self.language},
                                device=self.device)
            self.modelLoaded = True

            # Warm up the model
            dummyAudio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            self.asr({"raw": dummyAudio, "sampling_rate": 16000})

            self._monitorMemory()

    def unloadModel(self):
        """Unload the ASR model from GPU."""
        if self.modelLoaded:
            self._debugPrint("Unloading model from GPU...")

            # Delete the model and pipeline
            del self.asr
            self.asr = None

            self._cudaClean()

            self.modelLoaded = False
            self._monitorMemory()

    def _cudaClean(self):
        """Clean CUDA memory."""
        # Force garbage collection
        gc.collect()
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            # Additional forceful memory cleanup
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

        # Convert to mono if stereo
        if len(audioData.shape) > 1:
            audioData = np.mean(audioData, axis=1)

        # Transcribe
        result = self.asr({"raw": audioData, "sampling_rate": sampleRate})
        transcription = result["text"]

        # Clean transcription
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
            # Read audio file
            audioData, sampleRate = sf.read(audioFilePath)

            # Transcribe using base class method
            transcription = self.transcribeAudio(audioData, sampleRate)

            # Handle output
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

        # Audio recording parameters
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

        # Parameters for false word detection handling
        self.commonFalseDetectedWords = commonFalseDetectedWords if commonFalseDetectedWords else []
        self.loudnessThresholdOfCommonFalseDetectedWords = loudnessThresholdOf_commonFalseDetectedWords

        # Audio notifications setup
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

        # Initialize pygame for audio playback
        pygame.mixer.init()

        # Queue to store audio chunks
        self.audioQueue = queue.Queue()

        # Flags to control recording and output
        self.isRecordingActive = isRecordingActive
        self.isProgramActive = isProgramActive
        self.outputEnabled = outputEnabled
        self.lastActivityTime = time.time()

        # Runtime variables
        self.audioBuffer = np.array([], dtype=np.float32)
        self.emptyTranscriptionCount = 0
        self.recordingStartTime = 0
        self.lastTranscriptionTime = 0
        self.actualSampleRate = self.sampleRate
        self.actualChannels = self.channels

        self.lastValidTranscriptionTime = time.time()

        # List available audio devices
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
        self.lastValidTranscriptionTime = time.time()  # ← reset silence timer
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
                # Toggle recording state
                self.isRecordingActive = not self.isRecordingActive

                if self.isRecordingActive:
                    print("Recording started...")
                    self.playNotification("recordingOn")
                    # Reset the program timer when recording starts
                    startTime = time.time()
                    self.lastActivityTime = time.time()
                    # Ensure model is loaded when recording starts
                    if not self.modelLoaded:
                        self.loadModel()
                else:
                    print("Recording stopped...")
                    self.playNotification("recordingOff")

                # Wait for key release to prevent multiple triggers
                while keyboard.is_pressed(self.recordingToggleKey):
                    time.sleep(0.1)

            if keyboard.is_pressed(self.outputToggleKey):
                self.toggleOutput()
                # Wait for key release to prevent multiple triggers
                while keyboard.is_pressed(self.outputToggleKey):
                    time.sleep(0.1)

            time.sleep(0.1)

        self.isProgramActive = False
        print("Program timeout reached. Exiting...")

    def modelManager(self):
        """Monitor model usage and unload when inactive for too long."""
        while self.isProgramActive:
            currentTime = time.time()

            # If recording is inactive and model is loaded
            if not self.isRecordingActive and self.modelLoaded:
                if (currentTime - self.lastActivityTime) >= self.modelUnloadTimeout:
                    print(f"Model inactive for {self.modelUnloadTimeout} seconds, unloading...")
                    self.unloadModel()
                    self.playNotification("modelUnloaded")

            # If recording is active but model isn't loaded
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
        # Start keyboard monitor in a separate thread
        keyboardThread = threading.Thread(target=self.monitorKeyboardShortcuts)
        keyboardThread.daemon = True
        keyboardThread.start()

        # Start model manager in a separate thread
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

            # Check if recording duration exceeds max duration
            currentTime = time.time()
            if (currentTime - self.recordingStartTime) >= self.maxDurationRecording:
                print(
                    f"Recording reached maximum duration of {self.maxDurationRecording} seconds, stopping...")
                self.stopRecording()
                self.recordingStartTime = 0
                self.emptyTranscriptionCount = 0

    def transcribeAudio(self):
        """Handle transcription according to the selected mode."""
        if not self.isRecordingActive or len(self.audioBuffer) == 0:
            return

        currentTime = time.time()
        audioData = None

        # ────────────── MODE: constantInterval ──────────────
        if self.transcriptionMode == "constantInterval":
            if (currentTime - self.lastTranscriptionTime) >= self.transcriptionInterval:
                audioData = self.audioBuffer.copy()
                self.audioBuffer = np.array([], dtype=np.float32)
                self.lastTranscriptionTime = currentTime

        # ────────────── MODE: busyContinuous ──────────────
        elif self.transcriptionMode == "busyContinuous":
            requiredSamples = int(self.busyContinuousTime * self.actualSampleRate)
            if len(self.audioBuffer) >= requiredSamples:
                trailingChunk = self.audioBuffer[-requiredSamples:]
                trailingMean = np.mean(np.abs(trailingChunk))  # ← use mean, not sum
                if trailingMean < self.busyContinuousSilenceThreshold:  # ← new threshold
                    audioData = self.audioBuffer.copy()
                    self.audioBuffer = np.array([], dtype=np.float32)
                    self.lastTranscriptionTime = currentTime

        # ───────────────────────────────────────────────────
        if audioData is None:
            return

        segmentMean = np.mean(np.abs(audioData))
        if segmentMean < self.lowLoudnessSkip_threshold:
            transcription = ""
            self._debugPrint(f"lower than loudness threshold {segmentMean}")
        else:
            transcription = super().transcribeAudio(audioData, self.actualSampleRate)

        self.handleTranscriptionOutput(transcription, segmentMean)
        self.lastActivityTime = currentTime

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
        loudnessThresh = self.loudnessThresholdOfCommonFalseDetectedWords * effectiveInterval
        isBelow = loudnessValue < loudnessThresh
        isFalseDetection = isFalseWord and isBelow

        # ─────────────────────────── valid speech ────────────────────────────
        if not isEmpty and not isFalseDetection:
            print("Transcription:", transcription)

            if self.outputEnabled and not keyboard.is_pressed("ctrl"):
                pyautogui.write(transcription.lstrip(" ") + " ")

            self.emptyTranscriptionCount = 0  # optional, still reset
            self.lastValidTranscriptionTime = now  # ← mark last real speech
            return

        # ─────────────────────── silence / false detection ───────────────────
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
            # Load model initially
            print("Warming up model...")
            self.loadModel()

            # Setup device info
            self.setupDeviceInfo(deviceId)

            # Start monitoring threads
            self.startThreads()

            # Start audio stream
            with sd.InputStream(samplerate=self.actualSampleRate,
                                channels=self.actualChannels,
                                device=deviceId,
                                blocksize=self.blockSize,
                                callback=self.audioCallback):

                self.recordingStartTime = 0
                self.lastTranscriptionTime = 0

                while self.isProgramActive:
                    # Handle recording timing
                    self.handleRecordingTiming()

                    # Process audio chunks
                    self.processAudioChunks()

                    # Transcribe audio
                    self.transcribeAudio()

                    # Cleanup if recording inactive
                    self.cleanupInactiveRecording()

                    # Small sleep to prevent high CPU usage
                    time.sleep(0.01)

        except Exception as e:
            print(f"Error during audio processing: {e}")
            raise
        finally:
            self.isRecordingActive = False
            self.isProgramActive = False
            # Call base class cleanup
            super().cleanup()
            print("Program stopped.")


# Main execution
if __name__ == "__main__":
    try:
        transcriber = SpeechToTextTranscriber(
            modelName="openai/whisper-large-v3",
            transcriptionMode="busyContinuous",  # |constantInterval
            transcriptionInterval=4,  # Longer interval between transcriptions
            busyContinuousTime=1.4,
            commonFalseDetectedWords=["you", "thank you", "bye", 'amen'],
            loudnessThresholdOf_commonFalseDetectedWords=20,
            lowLoudnessSkip_threshold=0,
            playEnableSounds=False,
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

        # Use default input device
        transcriber.run()
    except Exception as e:
        print(f"Program error: {e}")

    # # fileTranscriber example
    # # Create file transcriber
    # transcriber = FileTranscriber(
    #     modelName="openai/whisper-large-v3",
    #     language="en",
    #     removeTrailingDots=True,
    #     debugPrint=True
    # )
    #
    # # Transcribe file to console
    # transcription = transcriber.transcribeFile(r"C:\Users\pc\Documents\Sound Recordings\shitSampleToSeeCanItBeTranscribedOrNot.mp3")
    #
    # # Transcribe file and save to output file
    # transcription = transcriber.transcribeFile(r"C:\Users\pc\Documents\Sound Recordings\shitSampleToSeeCanItBeTranscribedOrNot.mp3", r"C:\Users\pc\Documents\Sound Recordings\transcription_shitSampleToSeeCanItBeTranscribedOrNot_mp3.txt")
    #
    # # Clean up
    # transcriber.cleanup()
