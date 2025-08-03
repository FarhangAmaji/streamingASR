import logging
import queue
import threading
import time

import keyboard
import numpy as np
import pyautogui
import sounddevice as sd
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
transcriptionInterval = 5  # Seconds to accumulate audio before transcription
maxDuration = 60  # Maximum recording duration in seconds (optional)

# Queue to store audio chunks
audioQueue = queue.Queue()

# Flag to control recording
isRecordingActive = True
# Load ASR model with explicit language setting
asr = pipeline("automatic-speech-recognition",
               model="openai/whisper-large-v3",
               generate_kwargs={"language": "en"},  # Explicitly set language to English
               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


def audioCallback(inData, frames, time, status):
    """Callback function for audio stream."""
    if status:
        print(f"Audio callback status: {status}")
    if isRecordingActive:
        audioQueue.put(inData.copy())


def stopRecording():
    """Stop recording when 'q' is pressed."""
    global isRecordingActive
    keyboard.wait("q")
    print("Stopping recording...")
    isRecordingActive = False


print("Warming up model...")
dummyAudio = np.zeros(sampleRate, dtype=np.float32)  # 1 second of silence
# Use raw instead of input_features for warmup
asr({"raw": dummyAudio, "sampling_rate": sampleRate})  # Corrected for API


def processAudio(deviceId=None, maxDuration=maxDuration):
    """
    Continuously record audio, transcribe it, and type the transcription.
    Stops when 'q' is pressed or after maxDuration seconds.
    """
    global isRecordingActive
    try:
        # Query device info
        actualSampleRate = sampleRate  # Initialize with default
        actualChannels = channels  # Initialize with default

        if deviceId is not None:
            deviceInfo = sd.query_devices(deviceId)
            print(f"Device info: {deviceInfo}")
            actualSampleRate = deviceInfo.get("default_samplerate", sampleRate)
            actualChannels = min(channels, deviceInfo.get("max_input_channels", channels))

        print(f"Recording with sample rate {actualSampleRate} Hz, channels {actualChannels}...")
        print(f"Press 'q' to stop recording, or recording will stop after {maxDuration} seconds.")

        # Start stop key listener in a separate thread
        stopThread = threading.Thread(target=stopRecording)
        stopThread.daemon = True
        stopThread.start()

        # Initialize audio buffer
        audioBuffer = np.array([], dtype=np.float32)

        # Start audio stream
        with sd.InputStream(samplerate=actualSampleRate,
                            channels=actualChannels,
                            device=deviceId,
                            blocksize=blockSize,
                            callback=audioCallback):
            startTime = time.time()
            lastTranscriptionTime = startTime

            while isRecordingActive and (time.time() - startTime) < maxDuration:
                # Process audio chunks from the queue
                while not audioQueue.empty():
                    audioChunk = audioQueue.get()
                    if actualChannels > 1:
                        audioChunk = np.mean(audioChunk, axis=1)  # Convert stereo to mono
                    audioBuffer = np.concatenate((audioBuffer, audioChunk.flatten()))

                # Transcribe audio buffer periodically
                currentTime = time.time()
                if (currentTime - lastTranscriptionTime) >= transcriptionInterval and len(
                        audioBuffer) > 0:
                    print("Transcribing audio buffer...")
                    try:
                        # Corrected transcription call using raw
                        transcription = asr({
                            "raw": audioBuffer,
                            "sampling_rate": actualSampleRate
                        })["text"]
                        print("Transcription:", transcription)

                        # Filter out empty transcriptions or just periods
                        if transcription.strip() and transcription.strip() != ".":
                            # Give time to switch to the target window
                            time.sleep(0.5)

                            # Simply write the text with a space at the end (no Enter key)
                            # This prevents newlines and subsequent indentation
                            cleaned_text = transcription.strip()
                            pyautogui.write(cleaned_text + " ")
                    except Exception as e:
                        print(f"Error during transcription: {e}")

                    # Reset audio buffer and update last transcription time
                    audioBuffer = np.array([], dtype=np.float32)
                    lastTranscriptionTime = currentTime

                # Small sleep to prevent high CPU usage
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("Recording interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"Error during recording: {e}")
        raise
    finally:
        isRecordingActive = False
        print("Recording stopped.")


if __name__ == "__main__":
    # Replace with the correct device ID or use None for default
    # processAudio(deviceId=0, maxDuration=5)  # e.g., deviceId=1
    processAudio(deviceId=None, maxDuration=60)
