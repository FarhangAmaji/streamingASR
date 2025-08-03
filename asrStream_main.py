import logging
import queue
import threading

import keyboard
import math
import numpy as np
import pyautogui
import sounddevice as sd
import time
import torch
from transformers import pipeline

logging.basicConfig(level=logging.DEBUG)

# List available audio devices
print("Available audio devices:")
devices = sd.query_devices()
print(devices)

# Audio recording parameters
sampleRate = 16000  # Hz
channels = 1  # Mono
blockSize = 1024  # Number of frames per block
transcriptionInterval = 3  # Seconds to accumulate audio before transcription
maxDuration_recording = 460  # Maximum recording duration in seconds
maxDuration_programActive = 1777  # Maximum time the program stays active
consecutiveIdleTime = 13  # Maximum consecutive seconds of silence before stopping
model_unload_timeout = 1222  # Seconds of inactivity before unloading model from GPU

# Queue to store audio chunks
audioQueue = queue.Queue()

# Flags to control recording and output
isRecordingActive = True
isProgramActive = True
outputEnabled = False  # Flag to toggle pyautogui output
modelLoaded = False
lastActivityTime = time.time()

# Model variables
asr = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loadModel():
    """Load the ASR model to GPU"""
    global asr, modelLoaded
    if not modelLoaded:
        print("Loading model to GPU...")
        asr = pipeline("automatic-speech-recognition",
                       model="openai/whisper-large-v3",
                       generate_kwargs={"language": "en"},  # Explicitly set language to English
                       device=device)
        modelLoaded = True
        # Warm up the model
        dummyAudio = np.zeros(sampleRate, dtype=np.float32)  # 1 second of silence
        asr({"raw": dummyAudio, "sampling_rate": sampleRate})  # Corrected for API
        print("Model loaded and warmed up")


def unloadModel():
    """Unload the ASR model from GPU"""
    global asr, modelLoaded
    if modelLoaded:
        print("Unloading model from GPU...")
        asr = None
        torch.cuda.empty_cache()  # Free GPU memory
        modelLoaded = False
        print("Model unloaded from GPU")


def audioCallback(inData, frames, time, status):
    """Callback function for audio stream."""
    if status:
        print(f"Audio callback status: {status}")
    if isRecordingActive:
        audioQueue.put(inData.copy())


def toggleOutput():
    """Toggle output when 'q' is pressed."""
    global outputEnabled
    outputEnabled = not outputEnabled
    print(f"Output {'enabled' if outputEnabled else 'disabled'}")


def startRecording():
    """Start recording when Win+alt+L is pressed."""
    global isRecordingActive, lastActivityTime
    isRecordingActive = True
    lastActivityTime = time.time()
    # Make sure model is loaded when recording starts
    if not modelLoaded:
        loadModel()
    print("Recording started...")


def stopRecording():
    """Stop recording."""
    global isRecordingActive
    isRecordingActive = False
    print("Recording stopped...")


def monitorKeyboardShortcuts():
    """Monitor keyboard shortcuts."""
    global isProgramActive, isRecordingActive, lastActivityTime

    startTime = time.time()

    while isProgramActive and (time.time() - startTime) < maxDuration_programActive:
        if keyboard.is_pressed('win+alt+l'):
            # Toggle recording state
            isRecordingActive = not isRecordingActive

            if isRecordingActive:
                print("Recording started...")
                # Reset the program timer when recording starts
                startTime = time.time()
                lastActivityTime = time.time()
                # Ensure model is loaded when recording starts
                if not modelLoaded:
                    loadModel()
            else:
                print("Recording stopped...")

            # Wait for key release to prevent multiple triggers
            while keyboard.is_pressed('win+alt+l'):
                time.sleep(0.1)

        if keyboard.is_pressed('q'):
            toggleOutput()
            # Wait for key release to prevent multiple triggers
            while keyboard.is_pressed('q'):
                time.sleep(0.1)

        time.sleep(0.1)

    isProgramActive = False
    print("Program timeout reached. Exiting...")


def modelManager():
    """Monitor model usage and unload when inactive for too long"""
    global isProgramActive, isRecordingActive, lastActivityTime

    while isProgramActive:
        current_time = time.time()

        # If recording is inactive and model is loaded
        if not isRecordingActive and modelLoaded:
            if (current_time - lastActivityTime) >= model_unload_timeout:
                print(f"Model inactive for {model_unload_timeout} seconds, unloading...")
                unloadModel()

        # If recording is active but model isn't loaded
        if isRecordingActive and not modelLoaded:
            loadModel()
            lastActivityTime = current_time

        time.sleep(10)  # Check every 10 seconds


# Load model initially
print("Warming up model...")
loadModel()


def processAudio(deviceId=None):
    """
    Continuously record audio, transcribe it, and type the transcription.
    Stops when recording is deactivated or after maxDuration_recording seconds or after consecutive idle time.
    """
    global isRecordingActive, isProgramActive, lastActivityTime

    try:
        # Query device info
        actualSampleRate = sampleRate  # Initialize with default
        actualChannels = channels  # Initialize with default

        if deviceId is not None:
            deviceInfo = sd.query_devices(deviceId)
            print(f"Device info: {deviceInfo}")
            actualSampleRate = deviceInfo.get("default_samplerate", sampleRate)
            actualChannels = min(channels, deviceInfo.get("max_input_channels", channels))

        print(
            f"Audio processing started with sample rate {actualSampleRate} Hz, channels {actualChannels}...")
        print(
            f"Press 'Win+alt+L' to start recording (max {maxDuration_recording} seconds per session)")
        print(f"Press 'q' to toggle text output")
        print(f"Recording will stop after {consecutiveIdleTime} seconds of silence")
        print(f"Program will exit after {maxDuration_programActive} seconds of inactivity")
        print(f"Model will be unloaded after {model_unload_timeout} seconds of inactivity")

        # Start keyboard monitor in a separate thread
        keyboardThread = threading.Thread(target=monitorKeyboardShortcuts)
        keyboardThread.daemon = True
        keyboardThread.start()

        # Start model manager in a separate thread
        modelThread = threading.Thread(target=modelManager)
        modelThread.daemon = True
        modelThread.start()

        # Initialize audio buffer
        audioBuffer = np.array([], dtype=np.float32)

        # Track empty transcriptions
        emptyTranscriptionCount = 0
        maxEmptyTranscriptions = math.ceil(consecutiveIdleTime / transcriptionInterval)

        # Start audio stream
        with sd.InputStream(samplerate=actualSampleRate,
                            channels=actualChannels,
                            device=deviceId,
                            blocksize=blockSize,
                            callback=audioCallback):

            recordingStartTime = 0
            lastTranscriptionTime = 0

            while isProgramActive:
                # Handle recording session timing
                if isRecordingActive:
                    if recordingStartTime == 0:
                        recordingStartTime = time.time()
                        lastTranscriptionTime = recordingStartTime
                        lastActivityTime = time.time()

                    # Check if recording duration exceeded
                    if (time.time() - recordingStartTime) >= maxDuration_recording:
                        print(
                            f"Maximum recording duration of {maxDuration_recording} seconds reached")
                        stopRecording()
                        recordingStartTime = 0

                # Process audio chunks from the queue
                while not audioQueue.empty():
                    audioChunk = audioQueue.get()
                    if actualChannels > 1:
                        audioChunk = np.mean(audioChunk, axis=1)  # Convert stereo to mono
                    audioBuffer = np.concatenate((audioBuffer, audioChunk.flatten()))

                # Transcribe audio buffer periodically when recording is active
                if isRecordingActive:
                    currentTime = time.time()
                    if (currentTime - lastTranscriptionTime) >= transcriptionInterval and len(
                            audioBuffer) > 0:
                        print("Transcribing audio buffer...")
                        try:
                            # Ensure model is loaded before transcription
                            if not modelLoaded:
                                loadModel()

                            # Update activity timestamp
                            lastActivityTime = currentTime

                            # Corrected transcription call using raw
                            transcription = asr({
                                "raw": audioBuffer,
                                "sampling_rate": actualSampleRate
                            })["text"]
                            print("Transcription:", transcription)

                            # Filter out empty transcriptions or just periods
                            if transcription.strip() and transcription.strip() != ".":

                                # Write the text only if output is enabled
                                if outputEnabled:
                                    cleanedText = transcription.strip()
                                    pyautogui.write(cleanedText + " ")
                                else:
                                    print("Output is disabled. Text not typed.")

                                # Reset consecutive empty transcription count
                                emptyTranscriptionCount = 0
                            else:
                                # Increment empty transcription count
                                emptyTranscriptionCount += 1
                                print(
                                    f"Empty transcription detected ({emptyTranscriptionCount}/{maxEmptyTranscriptions})")

                                # Check if we've reached the maximum consecutive empty transcriptions
                                if emptyTranscriptionCount >= maxEmptyTranscriptions:
                                    print(
                                        f"Stopping recording after {consecutiveIdleTime} seconds of silence")
                                    stopRecording()
                                    recordingStartTime = 0
                        except Exception as e:
                            print(f"Error during transcription: {e}")

                        # Reset audio buffer and update last transcription time
                        audioBuffer = np.array([], dtype=np.float32)
                        lastTranscriptionTime = currentTime

                # If recording stopped, clear the audio buffer
                if not isRecordingActive and len(audioBuffer) > 0:
                    audioBuffer = np.array([], dtype=np.float32)
                    emptyTranscriptionCount = 0

                # Small sleep to prevent high CPU usage
                time.sleep(0.01)
    except Exception as e:
        print(f"Error during audio processing: {e}")
        raise
    finally:
        isRecordingActive = False
        isProgramActive = False
        # Make sure to unload the model before exiting
        if modelLoaded:
            unloadModel()
        print("Program stopped.")


# Main execution
if __name__ == "__main__":
    try:
        # Use default input device
        processAudio()
    except Exception as e:
        print(f"Program error: {e}")
    finally:
        # Ensure model is unloaded when program exits
        if modelLoaded:
            unloadModel()
