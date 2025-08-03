# audioProcesses.py
# ==============================================================================
# Audio Input Handling and Real-Time Processing
# ==============================================================================
#
# Purpose:
# - AudioHandler: Manages audio input stream using sounddevice library.
#   Handles device selection, stream starting/stopping, and queuing raw audio chunks.
# - RealTimeAudioProcessor: Accumulates audio chunks, implements transcription
#   trigger logic based on selected mode (dictation or constant interval),
#   and prepares audio segments for the ASR handler.
# ==============================================================================
import queue
import sys  # Needed for stderr print during early error
import time

import numpy as np

# Import sounddevice conditionally
try:
    import sounddevice as sd

    sounddeviceAvailable = True
except ImportError:
    sd = None
    sounddeviceAvailable = False
except Exception as e:  # Catch other potential init errors
    sd = None
    sounddeviceAvailable = False
    # Log error here as logger might not be set up yet in main module
    # This print remains as it's for a very early, critical dependency failure.
    print(f"ERROR: Failed to import or initialize sounddevice: {e}", file=sys.stderr)

# Import logging helpers from utils
from utils import logWarning, logDebug, logInfo, logError, logCritical


# ==================================
# Audio Handling (Input Stream)
# ==================================
class AudioHandler:
    """Manages audio input stream using sounddevice."""

    def __init__(self, config, stateManager):
        self.config = config
        self.stateManager = stateManager
        self.audioQueue = queue.Queue()  # Queue for raw audio chunks from callback
        self.stream = None  # Holds the sounddevice InputStream object
        self.streamInfo = {}  # Store info about the active stream
        # Check dependency and perform initial device setup
        if not sounddeviceAvailable:
            logCritical(
                "sounddevice library not found or failed to initialize. Audio input disabled.")
            # Optionally raise error to prevent app start without audio
            # raise ImportError("sounddevice library is required for audio input.")
        else:
            self._setupDeviceInfo()  # Query devices and set actual rates/channels

    def _printAudioDevices(self):
        """Helper function to log available audio devices."""
        if not sounddeviceAvailable: return
        logInfo("--- Querying Available Audio Devices ---")
        try:
            devices = sd.query_devices()
            if devices:
                logInfo(f"\n{devices}")  # Log the full device list object representation
            else:
                logInfo("No audio devices found by sounddevice.")
        except Exception as e:
            logError(f"Could not query audio devices: {e}", exc_info=True)
        logInfo("--------------------------------------")

    def _setupDeviceInfo(self):
        """Queries audio device info and sets actual sample rate/channels in config."""
        if not sounddeviceAvailable: return
        self._printAudioDevices()
        deviceId = self.config.get('deviceId')
        requestedRate = self.config.get('sampleRate', 16000)
        requestedChannels = self.config.get('channels', 1)
        actualRate = requestedRate
        actualChannels = requestedChannels
        logDebug(
            f"Setting up device info. Requested Device ID: {deviceId}, Rate: {requestedRate}, Channels: {requestedChannels}")
        try:
            deviceInfo = sd.query_devices(device=deviceId, kind='input')
            if isinstance(deviceInfo, dict):
                logDebug(f"Device info found: {deviceInfo}")
                defaultRate = deviceInfo.get("default_samplerate")
                maxInChannels = deviceInfo.get("max_input_channels")
                if maxInChannels is not None and requestedChannels > maxInChannels:
                    logWarning(
                        f"Requested {requestedChannels} channels, but device '{deviceInfo.get('name')}' only supports up to {maxInChannels}. Using {maxInChannels}.")
                    actualChannels = maxInChannels
                elif requestedChannels < 1:
                    logWarning(
                        f"Requested invalid channel count ({requestedChannels}). Using 1 (mono).")
                    actualChannels = 1
                else:
                    actualChannels = requestedChannels
                actualRate = requestedRate
                logInfo(
                    f"Selected Device: '{deviceInfo.get('name')}', Max Input Channels: {maxInChannels}, Default Rate: {defaultRate}")
            else:
                logWarning(
                    f"Could not retrieve detailed info for requested device ID '{deviceId}'. Using configured defaults or system default.")
                actualRate = requestedRate
                actualChannels = requestedChannels
                if actualChannels < 1: actualChannels = 1
        except ValueError as e:
            logError(f"Error querying audio device '{deviceId}': {e}. Check device ID/name.")
            logError("Using configured defaults/system default, but stream start might fail.")
            actualRate = requestedRate
            actualChannels = requestedChannels
            if actualChannels < 1: actualChannels = 1
        except Exception as e:
            logError(f"Unexpected error querying audio device information: {e}", exc_info=True)
            actualRate = requestedRate
            actualChannels = requestedChannels
            if actualChannels < 1: actualChannels = 1
        self.config.set('actualSampleRate', actualRate)
        self.config.set('actualChannels', actualChannels)
        logInfo(f"Audio configured for: Rate={actualRate} Hz, Channels={actualChannels}")

    def _audioCallback(self, indata: np.ndarray, frames: int, timeInfo, status: 'sd.CallbackFlags'):
        """
        Callback function executed by sounddevice stream thread for each audio block.
        Adds received audio data to the queue if recording is active.
        Uses string literal for CallbackFlags type hint for robustness if sounddevice fails import.
        """
        if status:
            try:
                if sd and isinstance(status, sd.CallbackFlags):
                    logWarning(f"Audio callback status flags: {status}")
                else:
                    logWarning(
                        f"Audio callback received status flags object (type: {type(status)}), but sounddevice module (sd) might be unavailable.")
            except Exception as e:
                logWarning(f"Error processing audio callback status flags: {e}")
        if self.stateManager and self.stateManager.isRecording():
            try:
                self.audioQueue.put_nowait(indata.copy())
            except queue.Full:
                logWarning("Audio input queue is full! Potential audio data loss.")
                try:
                    self.audioQueue.get_nowait()
                    self.audioQueue.put_nowait(indata.copy())
                except queue.Empty:
                    pass
                except Exception as qe:
                    logError(f"Error managing full audio queue: {qe}")

    def startStream(self) -> bool:
        """Starts the sounddevice input stream if not already active."""
        if not sounddeviceAvailable:
            logError("Cannot start audio stream: sounddevice library not available.")
            return False
        if self.stream is not None and self.stream.active:
            logDebug("Audio stream is already active.")
            return True
        if self.stream is not None:
            self.stopStream()
        try:
            rate = self.config.get('actualSampleRate')
            channels = self.config.get('actualChannels')
            deviceId = self.config.get('deviceId')
            blockSize = self.config.get('blockSize', 0)
            logInfo(
                f"Attempting to start audio stream (Device: {deviceId or 'Default'}, Rate: {rate}, Channels: {channels}, BlockSize: {blockSize or 'Auto'})...")
            self.clearQueue()
            self.stream = sd.InputStream(
                samplerate=rate,
                channels=channels,
                device=deviceId,
                blocksize=blockSize,
                dtype='float32',
                callback=self._audioCallback
            )
            self.stream.start()
            self.streamInfo = {
                'device': self.stream.device,
                'samplerate': self.stream.samplerate,
                'channels': self.stream.channels,
                'dtype': self.stream.dtype,
                'blocksize': self.stream.blocksize
            }
            logInfo(
                f"Audio stream started successfully on device {self.streamInfo['device']} ({self.streamInfo}).")
            return True
        except sd.PortAudioError as pae:
            logError(
                f"PortAudioError starting stream: {pae}. Is a microphone connected and configured?")
            logError(
                "Hints: Check system audio settings, device ID in config, sample rate/channel compatibility.")
            self.stream = None
            return False
        except ValueError as ve:
            logError(f"ValueError starting stream: {ve}. Check audio parameters.")
            self.stream = None
            return False
        except Exception as e:
            logError(f"Failed to start audio stream: {e}", exc_info=True)
            self.stream = None
            return False

    def stopStream(self):
        """Stops and closes the sounddevice input stream if active."""
        if self.stream is not None and self.stream.active:
            logInfo("Stopping audio stream...")
            try:
                self.stream.stop()
                self.stream.close()
                logInfo(f"Audio stream stopped and closed ({self.streamInfo}).")
            except Exception as e:
                logError(f"Error stopping/closing audio stream: {e}", exc_info=True)
            finally:
                self.stream = None
                self.streamInfo = {}
                self.clearQueue()
        elif self.stream is not None and not self.stream.active:
            logDebug("Audio stream object exists but is not active. Closing.")
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None
            self.streamInfo = {}
            self.clearQueue()
        else:
            logDebug("Audio stream already stopped or not initialized.")

    def getAudioChunk(self) -> np.ndarray | None:
        """Retrieves the next available audio chunk from the queue (non-blocking)."""
        try:
            return self.audioQueue.get_nowait()
        except queue.Empty:
            return None
        except Exception as e:
            logError(f"Error getting audio chunk from queue: {e}", exc_info=True)
            return None

    def getQueueSize(self) -> int:
        """Returns the approximate number of items in the audio queue."""
        return self.audioQueue.qsize()

    def clearQueue(self):
        """Clears all items from the audio input queue."""
        qsize = self.audioQueue.qsize()
        if qsize > 0:
            logDebug(f"Clearing {qsize} items from audio input queue...")
            with self.audioQueue.mutex:
                self.audioQueue.queue.clear()
            logDebug("Audio input queue cleared.")


# ==================================
# Real-Time Audio Processing Logic
# ==================================
class RealTimeAudioProcessor:
    """
    Handles audio buffer accumulation, processing modes (dictation, constant interval),
    and determines when transcription should be triggered based on audio analysis.
    """

    def __init__(self, config, stateManager):
        self.config = config
        self.stateManager = stateManager
        self.audioBuffer = np.array([], dtype=np.float32)
        self.lastTranscriptionTriggerTime = time.time()
        self.isCurrentlySpeaking = False
        self.silenceStartTime = None
        logDebug("RealTimeAudioProcessor initialized.")

    def _calculateChunkLoudness(self, audioChunk: np.ndarray) -> float:
        """Calculates the average absolute amplitude (proxy for loudness) of an audio chunk."""
        if audioChunk is None or audioChunk.size == 0:
            return 0.0
        if audioChunk.dtype.kind != 'f':
            if audioChunk.dtype.kind in ('i', 'u'):
                maxVal = np.iinfo(audioChunk.dtype).max
                minVal = np.iinfo(audioChunk.dtype).min
                if maxVal > minVal:
                    audioChunk = (audioChunk.astype(np.float32) - minVal) / (
                            maxVal - minVal) * 2.0 - 1.0
                else:
                    audioChunk = audioChunk.astype(np.float32)
            else:
                audioChunk = audioChunk.astype(np.float32)
        return np.mean(np.abs(audioChunk))

    def processIncomingChunk(self, audioChunk: np.ndarray) -> bool:
        """
        Processes a new raw audio chunk: converts to float32, ensures mono,
        updates internal state (like dictation mode), and appends to the buffer.
        Args:
            audioChunk (np.ndarray): Raw audio data chunk from AudioHandler.
        Returns:
            bool: True if the chunk was successfully processed and added to the buffer, False otherwise.
        """
        if audioChunk is None or audioChunk.size == 0:
            return False
        processedChunk = audioChunk
        if processedChunk.dtype != np.float32:
            try:
                if processedChunk.dtype.kind in ('i', 'u'):
                    maxVal = np.iinfo(processedChunk.dtype).max
                    minVal = np.iinfo(processedChunk.dtype).min
                    if maxVal > minVal:
                        processedChunk = (processedChunk.astype(np.float32) - minVal) / (
                                maxVal - minVal) * 2.0 - 1.0
                    else:
                        processedChunk = processedChunk.astype(np.float32)
                else:
                    processedChunk = processedChunk.astype(np.float32)
            except Exception as e:
                logError(f"Failed converting chunk to float32: {e}")
                return False

        numChannels = self.config.get('actualChannels', 1)
        if numChannels > 1:
            if len(processedChunk.shape) > 1 and processedChunk.shape[1] == numChannels:
                processedChunk = np.mean(processedChunk, axis=1)
            elif len(processedChunk.shape) == 1 and numChannels > 1:
                logWarning(
                    f"Received 1D audio data but expected {numChannels} channels. Proceeding as mono.")

        if processedChunk.ndim > 1:
            processedChunk = processedChunk.flatten()

        if processedChunk is None or processedChunk.size == 0:
            logWarning("Audio chunk became empty after processing (float/mono/flatten).")
            return False

        if self.config.get('transcriptionMode') == "dictationMode":
            self._updateDictationState(processedChunk)

        try:
            self.audioBuffer = np.concatenate((self.audioBuffer, processedChunk))
            return True
        except ValueError as e:
            logError(
                f"Error concatenating audio chunk to buffer: {e}. Buffer shape: {self.audioBuffer.shape}, Chunk shape: {processedChunk.shape}")
            return False
        except Exception as e:
            logError(f"Unexpected error appending chunk to buffer: {e}", exc_info=True)
            return False

    def _updateDictationState(self, monoChunk: np.ndarray):
        """Updates the speaking flag and silence timer for dictation mode based on chunk loudness."""
        chunkLoudness = self._calculateChunkLoudness(monoChunk)
        silenceThreshold = self.config.get('dictationMode_silenceLoudnessThreshold', 0.001)
        if chunkLoudness >= silenceThreshold:
            if not self.isCurrentlySpeaking:
                logDebug(
                    f"Speech detected (Chunk Loudness {chunkLoudness:.6f} >= {silenceThreshold:.6f})")
            self.isCurrentlySpeaking = True
            if self.silenceStartTime is not None:
                logDebug("Speech resumed, resetting silence timer.")
                self.silenceStartTime = None
        else:
            if self.isCurrentlySpeaking and self.silenceStartTime is None:
                logDebug(
                    f"Silence detected after speech (Chunk Loudness {chunkLoudness:.6f} < {silenceThreshold:.6f}). Starting silence timer ({self.config.get('dictationMode_silenceDurationToOutput', 0.6)}s)...")
                self.silenceStartTime = time.time()

    def checkTranscriptionTrigger(self) -> np.ndarray | None:
        """
        Checks if conditions are met to trigger transcription based on the current mode
        and the accumulated audio buffer.
        Returns:
            numpy.ndarray | None: A copy of the audio data segment (Transcription Window)
                                   to be transcribed if trigger conditions are met.
                                   Returns None otherwise.
        """
        mode = self.config.get('transcriptionMode')
        audioDataToTranscribe = None
        if mode == "constantIntervalMode":
            audioDataToTranscribe = self._checkTriggerConstantInterval()
        elif mode == "dictationMode":
            audioDataToTranscribe = self._checkTriggerDictationMode()
        else:
            logWarning(
                f"Unsupported transcriptionMode: {mode}. No transcription will be triggered.")
            return None

        if audioDataToTranscribe is not None:
            if audioDataToTranscribe.size == 0:
                logDebug("Trigger occurred but yielded empty audio data. Ignoring.")
                return None
            logInfo(
                f"Triggering transcription ({mode}). Buffer duration: {len(audioDataToTranscribe) / self.config.get('actualSampleRate', 1):.2f}s.")
            if self.stateManager: self.stateManager.updateLastActivityTime()
            if mode == "constantIntervalMode":
                self.clearBuffer()
                self.lastTranscriptionTriggerTime = time.time()
            elif mode == "dictationMode":
                # Buffer and state reset happens inside _checkTriggerDictationMode
                pass
            return audioDataToTranscribe
        else:
            if mode == "constantIntervalMode" and self._isConstantIntervalTimeReached():
                self.lastTranscriptionTriggerTime = time.time()
        return None

    def _isConstantIntervalTimeReached(self) -> bool:
        """Helper method to check if the time interval for constant mode has passed."""
        interval = self.config.get('constantIntervalMode_transcriptionInterval', 3.0)
        if interval <= 0: return False
        return (time.time() - self.lastTranscriptionTriggerTime) >= interval

    def _checkTriggerConstantInterval(self) -> np.ndarray | None:
        """Checks trigger conditions for constant interval mode."""
        if self._isConstantIntervalTimeReached() and self.audioBuffer.size > 0:
            logDebug(
                f"Constant interval trigger condition met. Buffer size: {self.audioBuffer.size} samples.")
            return self.audioBuffer.copy()
        return None

    def _checkTriggerDictationMode(self) -> np.ndarray | None:
        """Checks trigger conditions for dictation mode (silence after speech)."""
        if self.isCurrentlySpeaking and self.silenceStartTime is not None:
            requiredSilence = self.config.get('dictationMode_silenceDurationToOutput', 0.6)
            if requiredSilence <= 0: return None
            elapsedSilence = time.time() - self.silenceStartTime
            if elapsedSilence >= requiredSilence:
                logDebug(
                    f"Dictation mode trigger: Silence duration ({elapsedSilence:.2f}s) >= threshold ({requiredSilence}s).")
                if self.audioBuffer.size > 0:
                    audioData = self.audioBuffer.copy()
                    self.clearBuffer()
                    self.isCurrentlySpeaking = False
                    self.silenceStartTime = None
                    logDebug("Dictation mode state reset after successful trigger.")
                    return audioData
                else:
                    logWarning(
                        "Dictation mode trigger met, but audio buffer is empty. Resetting state anyway.")
                    self.clearBuffer()
                    self.isCurrentlySpeaking = False
                    self.silenceStartTime = None
                    return None
        return None

    def getAudioBufferCopyAndClear(self) -> np.ndarray | None:
        """
        Returns a copy of the current audio buffer and then clears the buffer
        and resets associated dictation state. Used for forced transcription.
        Returns:
            numpy.ndarray | None: A copy of the audio data if buffer is not empty, else None.
        """
        logDebug("getAudioBufferCopyAndClear called (e.g., by force transcription hotkey).")
        if self.audioBuffer.size > 0:
            audio_to_transcribe = self.audioBuffer.copy()
            duration = audio_to_transcribe.size / self.config.get('actualSampleRate', 16000) if self.config.get('actualSampleRate', 16000) > 0 else 0
            logInfo(f"Retrieved {duration:.2f}s of audio from buffer for forced transcription.")
            self.clearBuffer() # This also resets dictation state
            # Reset constant interval timer as well, as this segment is now handled
            self.lastTranscriptionTriggerTime = time.time()
            logDebug("Forced transcription: Constant interval timer reset.")
            return audio_to_transcribe
        else:
            logInfo("getAudioBufferCopyAndClear: Buffer is empty, nothing to return.")
            return None

    def getBufferDuration(self) -> float:
        """Calculates the duration of the current audio buffer in seconds."""
        if self.audioBuffer.size == 0:
            return 0.0
        sampleRate = self.config.get('actualSampleRate')
        return self.audioBuffer.size / sampleRate if sampleRate and sampleRate > 0 else 0.0

    def clearBufferIfOutputDisabled(self):
        """Clears the processor's audio buffer if text output state is currently disabled."""
        if self.stateManager and not self.stateManager.isOutputEnabled():
            if self.audioBuffer.size > 0:
                logInfo("Clearing audio buffer because output is disabled.")
                self.clearBuffer()

    def clearBuffer(self):
        """Clears the internal audio buffer and resets associated dictation state."""
        bufferLen = self.audioBuffer.size
        if bufferLen > 0:
            sampleRate = self.config.get('actualSampleRate', 16000)
            duration = bufferLen / sampleRate if sampleRate > 0 else 0
            logDebug(f"Clearing audio buffer with {bufferLen} samples ({duration:.2f}s).")
        self.audioBuffer = np.array([], dtype=np.float32)
        if self.isCurrentlySpeaking or self.silenceStartTime is not None:
            logDebug("Resetting dictation state due to buffer clear.")
            self.isCurrentlySpeaking = False
            self.silenceStartTime = None