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
transcriptionInterval = 5  # Seconds to accumulate audio before transcription
maxDuration = 60  # Maximum recording duration in seconds (optional)
consecutiveIdleTime = 28  # Maximum consecutive seconds of silence before stopping

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
    Stops when 'q' is pressed or after maxDuration seconds or after consecutive idle time.
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
        print(f"Recording will also stop after {consecutiveIdleTime} seconds of silence.")

        # Start stop key listener in a separate thread
        stopThread = threading.Thread(target=stopRecording)
        stopThread.daemon = True
        stopThread.start()

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
                            cleanedText = transcription.strip()
                            pyautogui.write(cleanedText + " ")

                            # Reset consecutive empty transcription count
                            emptyTranscriptionCount = 0
                        else:
                            # IMPORTANT: This handles the consecutiveIdleTime logic
                            # Single words like "you" or common phrases like "Thank you" when alone
                            # are likely false positives from background noise
                            # If we get multiple consecutive empty/minimal transcriptions,
                            # we'll stop recording after reaching maxEmptyTranscriptions
                            # (calculated as ceil(consecutiveIdleTime/transcriptionInterval))

                            # Consider also checking word count or implementing energy threshold here
                            # to better distinguish between silence and actual speech

                            # Increment empty transcription count
                            emptyTranscriptionCount += 1
                            print(
                                f"Empty transcription detected ({emptyTranscriptionCount}/{maxEmptyTranscriptions})")

                            # Check if we've reached the maximum consecutive empty transcriptions
                            if emptyTranscriptionCount >= maxEmptyTranscriptions:
                                print(f"Stopping after {consecutiveIdleTime} seconds of silence")
                                isRecordingActive = False
                                break
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
    processAudio(deviceId=None, maxDuration=460)
