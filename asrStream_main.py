import math
import os
import queue
import threading
import time
from pathlib import Path

import keyboard
import noisereduce as nr
import numpy as np
import pyautogui
import pygame
import sounddevice as sd
import torch
from transformers import pipeline


class SpeechToTextTranscriber:
    def __init__(self,
                 modelName="openai/whisper-large-v3",
                 transcriptionInterval=3,
                 maxDuration_recording=10000,
                 maxDuration_programActive=60 * 60,
                 model_unloadTimeout=5 * 60,
                 consecutiveIdleTime=100,
                 isRecordingActive=True,
                 isProgramActive=True,
                 outputEnabled=False,
                 sampleRate=16000,
                 channels=1,
                 removeTrailingDots=True,
                 language="en",
                 commonFalseDetectedWords=None,  # New parameter
                 loudnessThresholdOf_commonFalseDetectedWords=2.4,
                 debugPrint=False,
                 recordingToggleKey="win+alt+l",
                 outputToggleKey="q",
                 ):

        # Audio recording parameters
        self.sampleRate = sampleRate
        self.channels = channels
        self.blockSize = 1024  # Number of frames per block
        self.transcriptionInterval = transcriptionInterval
        self.maxDuration_recording = maxDuration_recording
        self.maxDuration_programActive = maxDuration_programActive
        self.consecutiveIdleTime = consecutiveIdleTime
        self.model_unloadTimeout = model_unloadTimeout
        self.modelName = modelName
        self.removeTrailingDots = removeTrailingDots
        self.language = language

        # parameters for false word detection handling
        self.commonFalseDetectedWords = commonFalseDetectedWords if commonFalseDetectedWords else []
        self.loudnessThresholdOf_commonFalseDetectedWords = loudnessThresholdOf_commonFalseDetectedWords

        # Audio notifications setup
        self.scriptDir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.audio_files = {
            "modelUnloaded": str(self.scriptDir / "modelUnloaded.mp3"),
            "outputDisabled": str(self.scriptDir / "outputDisabled.mp3"),
            "outputEnabled": str(self.scriptDir / "outputEnabled.mp3"),
            "recordingOff": str(self.scriptDir / "recordingOff.mp3"),
            "recordingOn": str(self.scriptDir / "recordingOn.mp3")
        }

        self.debugPrint = debugPrint
        self.recordingToggleKey = recordingToggleKey
        self.outputToggleKey = outputToggleKey

        # Initialize pygame for audio playback
        pygame.mixer.init()

        # Queue to store audio chunks
        self.audioQueue = queue.Queue()

        # Flags to control recording and output
        self.isRecordingActive = isRecordingActive
        self.isProgramActive = isProgramActive
        self.outputEnabled = outputEnabled  # Flag to toggle pyautogui output
        self.modelLoaded = False
        self.lastActivityTime = time.time()

        # Model variables
        self.asr = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Runtime variables
        self.audioBuffer = np.array([], dtype=np.float32)
        self.emptyTranscriptionCount = 0
        self.recordingStartTime = 0
        self.lastTranscriptionTime = 0
        self.actualSampleRate = self.sampleRate
        self.actualChannels = self.channels

        # List available audio devices
        print("Available audio devices:")
        devices = sd.query_devices()
        print(devices)

    def playNotification(self, soundName):
        """Play notification sound"""
        if soundName in self.audio_files:
            try:
                sound = pygame.mixer.Sound(self.audio_files[soundName])
                sound.play()
            except Exception as e:
                print(f"Error playing notification sound: {e}")

    def loadModel(self):
        """Load the ASR model to GPU"""
        if not self.modelLoaded:
            if self.debugPrint:
                print("Loading model to GPU...")
            self.asr = pipeline("automatic-speech-recognition",
                                model=self.modelName,
                                generate_kwargs={"language": self.language},
                                # Explicitly set language to English
                                device=self.device)
            self.modelLoaded = True
            # Warm up the model
            dummyAudio = np.zeros(self.sampleRate, dtype=np.float32)  # 1 second of silence
            self.asr({"raw": dummyAudio, "sampling_rate": self.sampleRate})  # Corrected for API
            if self.debugPrint:
                print("Model loaded and warmed up")

    def unloadModel(self):
        """Unload the ASR model from GPU"""
        if self.modelLoaded:
            print("Unloading model from GPU...")
            self.asr = None
            torch.cuda.empty_cache()  # Free GPU memory
            # This frees up GPU memory that was allocated by PyTorch in this process and doesn't interrupt my other codes or other programmes which use gpu
            self.modelLoaded = False
            self.playNotification("modelUnloaded")
            print("Model unloaded from GPU")

    def audioCallback(self, inData, frames, time, status):
        """Callback function for audio stream."""
        if status:
            print(f"Audio callback status: {status}")
        if self.isRecordingActive:
            self.audioQueue.put(inData.copy())

    def toggleOutput(self):
        """Toggle output when 'q' is pressed."""
        self.outputEnabled = not self.outputEnabled
        if self.outputEnabled:
            self.playNotification("outputEnabled")
        else:
            self.playNotification("outputDisabled")
        print(f"Output {'enabled' if self.outputEnabled else 'disabled'}")

    def startRecording(self):
        """Start recording when Win+alt+L is pressed."""
        self.isRecordingActive = True
        self.lastActivityTime = time.time()
        # Make sure model is loaded when recording starts
        if not self.modelLoaded:
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

        while self.isProgramActive and (time.time() - startTime) < self.maxDuration_programActive:
            if keyboard.is_pressed(self.recordingToggleKey):
                # Toggle recording state
                self.isRecordingActive = not self.isRecordingActive

                if self.isRecordingActive:
                    print("Recording started...")  # Keep this print regardless of debugPrint
                    self.playNotification("recordingOn")
                    # Reset the program timer when recording starts
                    startTime = time.time()
                    self.lastActivityTime = time.time()
                    # Ensure model is loaded when recording starts
                    if not self.modelLoaded:
                        self.loadModel()
                else:
                    print("Recording stopped...")  # Keep this print regardless of debugPrint
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
        print("Program timeout reached. Exiting...")  # Keep this print regardless of debugPrint

    def modelManager(self):
        """Monitor model usage and unload when inactive for too long"""
        while self.isProgramActive:
            current_time = time.time()

            # If recording is inactive and model is loaded
            if not self.isRecordingActive and self.modelLoaded:
                if (current_time - self.lastActivityTime) >= self.model_unloadTimeout:
                    print(f"Model inactive for {self.model_unloadTimeout} seconds, unloading...")
                    self.unloadModel()

            # If recording is active but model isn't loaded
            if self.isRecordingActive and not self.modelLoaded:
                self.loadModel()
                self.lastActivityTime = current_time

            time.sleep(10)  # Check every 10 seconds

    def setupDeviceInfo(self, deviceId=None):
        """Set up audio device information"""
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
            f"Press 'Win+alt+L' to start recording (max {self.maxDuration_recording} seconds per session)")
        print(f"Press 'q' to toggle text output")
        print(f"Recording will stop after {self.consecutiveIdleTime} seconds of silence")
        print(f"Program will exit after {self.maxDuration_programActive} seconds of inactivity")
        print(f"Model will be unloaded after {self.model_unloadTimeout} seconds of inactivity")

    def startThreads(self):
        """Start monitoring threads"""
        # Start keyboard monitor in a separate thread
        keyboardThread = threading.Thread(target=self.monitorKeyboardShortcuts)
        keyboardThread.daemon = True
        keyboardThread.start()

        # Start model manager in a separate thread
        modelThread = threading.Thread(target=self.modelManager)
        modelThread.daemon = True
        modelThread.start()

    def processAudioChunks(self):
        """Process audio chunks from the queue"""
        while not self.audioQueue.empty():
            audioChunk = self.audioQueue.get()
            if self.actualChannels > 1:
                audioChunk = np.mean(audioChunk, axis=1)  # Convert stereo to mono
            self.audioBuffer = np.concatenate((self.audioBuffer, audioChunk.flatten()))

    def handleRecordingTiming(self):
        """Handle recording session timing"""
        if self.isRecordingActive:
            if self.recordingStartTime == 0:
                self.recordingStartTime = time.time()
                self.lastTranscriptionTime = self.recordingStartTime
                self.lastActivityTime = time.time()

            # Check if recording duration exceeded
            if (time.time() - self.recordingStartTime) >= self.maxDuration_recording:
                print(f"Maximum recording duration of {self.maxDuration_recording} seconds reached")
                self.stopRecording()
                self.recordingStartTime = 0

    def transcribeAudio(self):
        """Transcribe audio buffer and handle output"""
        if self.isRecordingActive:
            currentTime = time.time()
            if (currentTime - self.lastTranscriptionTime) >= self.transcriptionInterval and len(
                    self.audioBuffer) > 0:
                try:
                    # Ensure model is loaded before transcription
                    if not self.modelLoaded:
                        self.loadModel()

                    # Update activity timestamp
                    self.lastActivityTime = currentTime

                    # Calculate and store the loudness sum
                    loudnessSum = np.sum(np.abs(self.audioBuffer))
                    if self.debugPrint:
                        print(f"Sum of loudness for audio buffer: {loudnessSum}")

                    # Apply noise reduction to the audio buffer
                    reducedNoiseAudio = nr.reduce_noise(
                        y=self.audioBuffer,
                        sr=self.actualSampleRate,
                        stationary=False,
                        prop_decrease=0.8
                    )

                    # Transcription call using raw
                    transcription = self.asr({
                        "raw": reducedNoiseAudio,
                        "sampling_rate": self.actualSampleRate
                    })["text"]

                    # Pass loudnessSum to handleTranscriptionOutput
                    self.handleTranscriptionOutput(transcription, loudnessSum)
                except Exception as e:
                    print(f"Error during transcription: {e}")

                # Reset audio buffer and update last transcription time
                self.audioBuffer = np.array([], dtype=np.float32)
                self.lastTranscriptionTime = currentTime

    def handleTranscriptionOutput(self, transcription, loudnessSum):
        """Process transcription output with false detection handling and Ctrl key management."""
        # Remove trailing dots if the option is enabled
        cleanedText = transcription.rstrip('.') if self.removeTrailingDots else transcription
        cleanedText = cleanedText.strip().lower()  # Normalize for comparison

        # Calculate the loudness threshold for this transcription interval
        loudnessThreshold = self.loudnessThresholdOf_commonFalseDetectedWords * self.transcriptionInterval

        # Check if the transcription is empty or just periods
        isEmpty = not cleanedText or cleanedText == "."

        # Check if the transcription is in the common false detection list
        isInFalseDetectionList = cleanedText in [word.lower() for word in
                                                 self.commonFalseDetectedWords]

        # Check if the loudness is below threshold
        isBelowThreshold = loudnessSum < loudnessThreshold

        # Determine if this is a false detection
        isFalseDetection = isInFalseDetectionList and isBelowThreshold

        # Print when word is in false detection list but above threshold
        if isInFalseDetectionList and not isBelowThreshold:
            print(
                f"Potential false detection but above threshold: '{cleanedText}'. Loudness: {loudnessSum}, Threshold: {loudnessThreshold}")

        if isEmpty or isFalseDetection:
            # Increment empty transcription count
            self.emptyTranscriptionCount += 1
            maxEmptyTranscriptions = math.ceil(
                self.consecutiveIdleTime / self.transcriptionInterval)
            if self.debugPrint:
                print(
                    f"Empty transcription detected"
                    f"({self.emptyTranscriptionCount}/{maxEmptyTranscriptions})"
                )

            # Check if we've reached the maximum number of empty transcriptions
            if self.emptyTranscriptionCount >= maxEmptyTranscriptions:
                print(
                    f"Reached {self.consecutiveIdleTime} seconds of silence, stopping recording..."
                )
                self.stopRecording()
                self.recordingStartTime = 0
                self.emptyTranscriptionCount = 0
        else:
            print("Transcription:", transcription)  # Always print this regardless of debugPrint
            # Valid transcription
            if self.outputEnabled:
                # Restore original case and formatting for output
                outputText = transcription.rstrip('.') if self.removeTrailingDots else transcription
                outputText = outputText.lstrip(" ") + " "

                ctrl_was_pressed = keyboard.is_pressed('ctrl')  # Check if Ctrl is pressed
                if not ctrl_was_pressed:
                    pyautogui.write(outputText)

            # Reset consecutive empty transcription count
            self.emptyTranscriptionCount = 0

    def cleanupInactiveRecording(self):
        """Clean up when recording is inactive"""
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
            # Make sure to unload the model before exiting
            if self.modelLoaded:
                self.unloadModel()
            print("Program stopped.")


# Main execution
if __name__ == "__main__":
    try:
        transcriber = SpeechToTextTranscriber(
            modelName="openai/whisper-large-v3",
            transcriptionInterval=1,  # Longer interval between transcriptions
            commonFalseDetectedWords=["you", "thank you", "bye", 'amen'],
            loudnessThresholdOf_commonFalseDetectedWords=20,
            maxDuration_recording=10000,  # 10000s max recording
            maxDuration_programActive=3600,  # 1 hour program active time
            model_unloadTimeout=5 * 60,  # Unload after 5 minutes
            consecutiveIdleTime=40,  # Stop after 20 seconds of silence
            isRecordingActive=True,  # Start with recording off
            outputEnabled=False,  # Start with output off
            sampleRate=16000,  # Higher sample rate
            channels=1
        )

        # Use default input device
        transcriber.run()
    except Exception as e:
        print(f"Program error: {e}")
