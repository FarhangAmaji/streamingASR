import queue
import threading
import time
import math

import keyboard
import numpy as np
import pyautogui
import sounddevice as sd
import torch
from transformers import pipeline


class SpeechToTextTranscriber:
    def __init__(self,
                 modelName="openai/whisper-large-v3",
                 transcriptionInterval=3,
                 maxDuration_recording=460,
                 maxDuration_programActive=60 * 60,
                 model_unloadTimeout=5 * 60,
                 consecutiveIdleTime=100,
                 isRecordingActive=True,
                 isProgramActive=True,
                 outputEnabled=False,
                 sampleRate=16000,
                 channels=1,
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

    def loadModel(self):
        """Load the ASR model to GPU"""
        if not self.modelLoaded:
            print("Loading model to GPU...")
            self.asr = pipeline("automatic-speech-recognition",
                                model=self.modelName,
                                generate_kwargs={"language": "en"},
                                # Explicitly set language to English
                                device=self.device)
            self.modelLoaded = True
            # Warm up the model
            dummyAudio = np.zeros(self.sampleRate, dtype=np.float32)  # 1 second of silence
            self.asr({"raw": dummyAudio, "sampling_rate": self.sampleRate})  # Corrected for API
            print("Model loaded and warmed up")

    def unloadModel(self):
        """Unload the ASR model from GPU"""
        if self.modelLoaded:
            print("Unloading model from GPU...")
            self.asr = None
            torch.cuda.empty_cache()  # Free GPU memory
            # This frees up GPU memory that was allocated by PyTorch in this process and doesn't interrupt my other codes or other programmes which use gpu
            self.modelLoaded = False
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
        print("Recording stopped...")

    def monitorKeyboardShortcuts(self):
        """Monitor keyboard shortcuts."""
        startTime = time.time()

        while self.isProgramActive and (time.time() - startTime) < self.maxDuration_programActive:
            if keyboard.is_pressed('win+alt+l'):
                # Toggle recording state
                self.isRecordingActive = not self.isRecordingActive

                if self.isRecordingActive:
                    print("Recording started...")
                    # Reset the program timer when recording starts
                    startTime = time.time()
                    self.lastActivityTime = time.time()
                    # Ensure model is loaded when recording starts
                    if not self.modelLoaded:
                        self.loadModel()
                else:
                    print("Recording stopped...")

                # Wait for key release to prevent multiple triggers
                while keyboard.is_pressed('win+alt+l'):
                    time.sleep(0.1)

            if keyboard.is_pressed('q'):
                self.toggleOutput()
                # Wait for key release to prevent multiple triggers
                while keyboard.is_pressed('q'):
                    time.sleep(0.1)

            time.sleep(0.1)

        self.isProgramActive = False
        print("Program timeout reached. Exiting...")

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
                print("Transcribing audio buffer...")
                try:
                    # Ensure model is loaded before transcription
                    if not self.modelLoaded:
                        self.loadModel()

                    # Update activity timestamp
                    self.lastActivityTime = currentTime

                    # Transcription call using raw
                    transcription = self.asr({
                        "raw": self.audioBuffer,
                        "sampling_rate": self.actualSampleRate
                    })["text"]
                    print("Transcription:", transcription)

                    self.handleTranscriptionOutput(transcription)
                except Exception as e:
                    print(f"Error during transcription: {e}")

                # Reset audio buffer and update last transcription time
                self.audioBuffer = np.array([], dtype=np.float32)
                self.lastTranscriptionTime = currentTime

    def handleTranscriptionOutput(self, transcription):
        """Process transcription output"""
        # Filter out empty transcriptions or just periods
        if transcription.strip() and transcription.strip() != ".":
            # Write the text only if output is enabled
            if self.outputEnabled:
                cleanedText = transcription.strip()
                pyautogui.write(cleanedText + " ")
            else:
                print("Output is disabled. Text not typed.")

            # Reset consecutive empty transcription count
            self.emptyTranscriptionCount = 0
        else:
            # Increment empty transcription count
            self.emptyTranscriptionCount += 1
            maxEmptyTranscriptions = math.ceil(
                self.consecutiveIdleTime / self.transcriptionInterval)
            print(
                f"Empty transcription detected ({self.emptyTranscriptionCount}/{maxEmptyTranscriptions})")

            # Check if we've reached the maximum consecutive empty transcriptions
            if self.emptyTranscriptionCount >= maxEmptyTranscriptions:
                print(f"Stopping recording after {self.consecutiveIdleTime} seconds of silence")
                self.stopRecording()
                self.recordingStartTime = 0

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
            transcriptionInterval=5,  # Longer interval between transcriptions
            maxDuration_recording=200,  # 200s max recording
            maxDuration_programActive=3600,  # 1 hour program active time
            model_unloadTimeout=5 * 60,  # Unload after 5 minutes
            consecutiveIdleTime=100,  # Stop after 20 seconds of silence
            isRecordingActive=True,  # Start with recording off
            outputEnabled=False,  # Start with output off
            sampleRate=16000,  # Higher sample rate
            channels=1
        )

        # Use default input device
        transcriber.run()
    except Exception as e:
        print(f"Program error: {e}")
