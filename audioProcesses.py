# audioProcesses.py

import queue
import time

import numpy as np
# Pygame and PyAutoGUI are imported conditionally later where needed
import sounddevice as sd

from utils import logWarning, logDebug, logInfo, logError


# ==================================
# Audio Handling
# ==================================
class AudioHandler:
    """Manages audio input stream using sounddevice."""

    def __init__(self, config, stateManager):
        self.config = config
        self.stateManager = stateManager
        self.audioQueue = queue.Queue()
        self.stream = None
        # Device info setup might happen before or after ASR handler determines device.
        # Ensure actual rates are used correctly.
        self._setupDeviceInfo()

    def _logDebug(self, message):
        logDebug(message, self.config.get('debugPrint'))

    def _printAudioDevices(self):
        """Helper function to print available audio devices."""
        logInfo("--- Available Audio Devices ---")
        try:
            devices = sd.query_devices()
            print(devices)
        except Exception as e:
            logError(f"Could not query audio devices: {e}")
        logInfo("-----------------------------")

    def _setupDeviceInfo(self):
        """Queries audio device info and sets actual sample rate/channels in config."""
        self._printAudioDevices()
        deviceId = self.config.get('deviceId')
        requestedRate = self.config.get('sampleRate')
        requestedChannels = self.config.get('channels')
        actualRate = requestedRate
        actualChannels = requestedChannels

        self._logDebug(
            f"Setting up device info. Requested device ID: {deviceId}, Rate: {requestedRate}, Channels: {requestedChannels}")
        try:
            deviceInfo = sd.query_devices(deviceId, kind='input')
            self._logDebug(f"Device info: {deviceInfo}")
            if isinstance(deviceInfo, dict):
                actualRate = int(deviceInfo.get("default_samplerate", requestedRate))
                actualChannels = min(requestedChannels,
                                     int(deviceInfo.get("max_input_channels", requestedChannels)))
            else:
                logWarning("Could not retrieve detailed device info, using configured defaults.")
        except Exception as e:
            logWarning(f"Could not query audio device information: {e}. Using configured defaults.")

        if actualChannels < 1:
            logWarning(f"Determined invalid input channels ({actualChannels}), defaulting to 1.")
            actualChannels = 1

        self.config.set('actualSampleRate', actualRate)
        self.config.set('actualChannels', actualChannels)
        logInfo(f"Using Sample Rate: {actualRate} Hz, Channels: {actualChannels}")

    def _audioCallback(self, inData, frames, timeInfo, status):
        """Callback function executed by sounddevice stream thread."""
        if status:
            logWarning(f"Audio callback status: {status}")
        if self.stateManager.isRecording():
            self.audioQueue.put(inData.copy())

    def startStream(self):
        """Starts the sounddevice input stream."""
        if self.stream is not None and self.stream.active:
            self._logDebug("Audio stream already active.")
            return True
        try:
            logInfo("Starting audio stream...")
            self.stream = sd.InputStream(
                samplerate=self.config.get('actualSampleRate'),
                channels=self.config.get('actualChannels'),
                device=self.config.get('deviceId'),
                blocksize=self.config.get('blockSize', 1024),
                callback=self._audioCallback
            )
            self.stream.start()
            logInfo("Audio stream started successfully.")
            return True
        except Exception as e:
            logError(f"Failed to start audio stream: {e}")
            # Provide more specific error guidance if possible
            if "invalid channel count" in str(e).lower():
                logError(
                    "Hint: Check if the configured 'channels' count is supported by your selected audio device.")
            elif "invalid sample rate" in str(e).lower():
                logError(
                    "Hint: Check if the configured 'sampleRate' is supported by your selected audio device.")
            elif "device unavailable" in str(e).lower():
                logError(
                    f"Hint: Check if audio device ID '{self.config.get('deviceId')}' is correct and available.")
            self.stream = None
            return False

    def stopStream(self):
        """Stops the sounddevice input stream."""
        if self.stream is not None and self.stream.active:
            try:
                logInfo("Stopping audio stream...")
                self.stream.stop()
                self.stream.close()
                logInfo("Audio stream stopped.")
            except Exception as e:
                logError(f"Error stopping audio stream: {e}")
            finally:
                self.stream = None
        else:
            self._logDebug("Audio stream already stopped or not initialized.")

    def getAudioChunk(self):
        """Retrieves the next available audio chunk from the queue (non-blocking)."""
        try:
            return self.audioQueue.get_nowait()
        except queue.Empty:
            return None

    def getQueueSize(self):
        """Returns the approximate number of items in the audio queue."""
        return self.audioQueue.qsize()

    def clearQueue(self):
        """Clears all items from the audio queue."""
        while not self.audioQueue.empty():
            try:
                self.audioQueue.get_nowait()
            except queue.Empty:
                break
        self._logDebug("Audio queue cleared.")


# ==================================
# Real-Time Audio Processing Logic
# ==================================
class RealTimeAudioProcessor:
    """
    Handles audio buffer accumulation, processing modes (dictation, constant interval),
    and determines when transcription should be triggered based on audio analysis.
    This component is agnostic to where the transcription happens (local/remote).
    """

    def __init__(self, config, stateManager):
        self.config = config
        self.stateManager = stateManager
        self.audioBuffer = np.array([], dtype=np.float32)
        self.lastTranscriptionTriggerTime = time.time()  # For constant interval mode

        # Dictation mode state
        self.isCurrentlySpeaking = False
        self.silenceStartTime = None

    def _logDebug(self, message):
        logDebug(message, self.config.get('debugPrint'))

    def _calculateChunkLoudness(self, audioChunk):
        """Calculates the average absolute amplitude of an audio chunk."""
        if audioChunk is None or len(audioChunk) == 0:
            return 0.0
        # Ensure float input for abs
        if audioChunk.dtype.kind != 'f':
            audioChunk = audioChunk.astype(np.float32) / np.iinfo(
                audioChunk.dtype).max  # Basic normalization if int
        return np.mean(np.abs(audioChunk))

    def processIncomingChunk(self, audioChunk):
        """Processes a new audio chunk: converts to float32, mono, updates state, appends to buffer."""
        if audioChunk is None:
            return False  # No chunk processed

        # --- Ensure Float32 Audio ---
        if audioChunk.dtype != np.float32:
            # Attempt normalization if integer type
            if audioChunk.dtype.kind in ('i', 'u'):
                max_val = np.iinfo(audioChunk.dtype).max
                min_val = np.iinfo(audioChunk.dtype).min
                # Avoid division by zero for empty range
                if max_val > min_val:
                    audioChunk = (audioChunk.astype(np.float32) - min_val) / (
                            max_val - min_val) * 2.0 - 1.0
                else:
                    audioChunk = audioChunk.astype(np.float32)  # Just cast if range is zero
            else:
                # Direct cast for other float types
                audioChunk = audioChunk.astype(np.float32)

        # --- Ensure Mono Audio ---
        monoChunk = audioChunk.flatten()
        if self.config.get('actualChannels') > 1:
            if len(audioChunk.shape) > 1 and audioChunk.shape[1] > 1:
                monoChunk = np.mean(audioChunk, axis=1)

        # --- Update Dictation Mode State (if applicable) ---
        if self.config.get('transcriptionMode') == "dictationMode":
            self._updateDictationState(monoChunk)

        # --- Append Chunk to Buffer ---
        if monoChunk is not None and len(monoChunk) > 0:
            self.audioBuffer = np.concatenate((self.audioBuffer, monoChunk))
            return True  # Chunk was processed and added
        return False

    def _updateDictationState(self, monoChunk):
        """Updates the speaking flag and silence timer for dictation mode."""
        chunkLoudness = self._calculateChunkLoudness(monoChunk)
        silenceThreshold = self.config.get('dictationMode_silenceLoudnessThreshold', 0.001)

        if chunkLoudness >= silenceThreshold:
            if not self.isCurrentlySpeaking:
                self._logDebug(
                    f"Speech detected (Loudness {chunkLoudness:.6f} >= {silenceThreshold:.6f})")
            self.isCurrentlySpeaking = True
            self.silenceStartTime = None
        else:
            if self.isCurrentlySpeaking and self.silenceStartTime is None:
                self._logDebug(
                    f"Silence detected after speech (Loudness {chunkLoudness:.6f}). Starting silence timer ({self.config.get('dictationMode_silenceDurationToOutput')}s)")
                self.silenceStartTime = time.time()

    def checkTranscriptionTrigger(self):
        """
        Checks if conditions are met to trigger transcription based on the current mode.

        Returns:
            numpy.ndarray or None: The audio data (Transcription Window) to be transcribed,
                                   or None if transcription should not be triggered yet.
        """
        mode = self.config.get('transcriptionMode')
        audioDataToTranscribe = None

        if mode == "constantIntervalMode":
            audioDataToTranscribe = self._checkTriggerConstantInterval()
        elif mode == "dictationMode":
            audioDataToTranscribe = self._checkTriggerDictationMode()

        if audioDataToTranscribe is not None and len(audioDataToTranscribe) > 0:
            self.stateManager.updateLastActivityTime()  # Mark activity when preparing data for ASR
            if mode == "constantIntervalMode":
                self.clearBuffer()
                self.lastTranscriptionTriggerTime = time.time()
            # Dictation mode buffer is cleared *within* _checkTriggerDictationMode on success

            return audioDataToTranscribe
        else:
            # Check and potentially reset constant interval timer even if buffer is empty
            if mode == "constantIntervalMode" and self._isConstantIntervalTimeReached():
                self.lastTranscriptionTriggerTime = time.time()
                # self._logDebug("Constant interval reached, buffer empty. Resetting timer.") # Can be noisy

        return None

    def _isConstantIntervalTimeReached(self):
        """Checks if the time interval for constant mode has passed."""
        interval = self.config.get('constantIntervalMode_transcriptionInterval', 3.0)
        return (time.time() - self.lastTranscriptionTriggerTime) >= interval

    def _checkTriggerConstantInterval(self):
        """Checks trigger conditions for constant interval mode."""
        if self._isConstantIntervalTimeReached() and len(self.audioBuffer) > 0:
            self._logDebug(f"Constant interval trigger. Buffer: {self.getBufferDuration():.2f}s.")
            return self.audioBuffer.copy()
        return None

    def _checkTriggerDictationMode(self):
        """Checks trigger conditions for dictation mode."""
        if self.isCurrentlySpeaking and self.silenceStartTime is not None:
            elapsedSilence = time.time() - self.silenceStartTime
            requiredSilence = self.config.get('dictationMode_silenceDurationToOutput', 0.6)

            if elapsedSilence >= requiredSilence:
                self._logDebug(
                    f"Dictation mode trigger: Silence duration ({elapsedSilence:.2f}s) >= threshold ({requiredSilence}s). Buffer: {self.getBufferDuration():.2f}s.")
                if len(self.audioBuffer) > 0:
                    audioData = self.audioBuffer.copy()
                    # Reset state *after* confirming trigger and getting data
                    self.clearBuffer()  # Clear buffer for next utterance
                    self.isCurrentlySpeaking = False
                    self.silenceStartTime = None
                    self._logDebug("Dictation mode state reset after trigger.")
                    return audioData
                else:
                    self._logDebug(
                        "Dictation mode trigger met, but buffer is empty. Resetting state.")
                    self.isCurrentlySpeaking = False
                    self.silenceStartTime = None
                    return None
        return None

    def getBufferDuration(self):
        """Calculates the duration of the current audio buffer in seconds."""
        if len(self.audioBuffer) == 0:
            return 0.0
        sampleRate = self.config.get('actualSampleRate')
        return len(self.audioBuffer) / sampleRate if sampleRate > 0 else 0.0

    def clearBufferIfOutputDisabled(self):
        """Clears the buffer if output is disabled to prevent backlog."""
        if not self.stateManager.isOutputEnabled():
            if len(self.audioBuffer) > 0:
                self._logDebug("Clearing audio buffer because output is disabled.")
                self.clearBuffer()

    def clearBuffer(self):
        """Clears the internal audio buffer."""
        bufferLen = len(self.audioBuffer)
        if bufferLen > 0:
            self._logDebug(
                f"Clearing audio buffer with {bufferLen} samples ({bufferLen / self.config.get('actualSampleRate', 16000):.2f}s).")
        self.audioBuffer = np.array([], dtype=np.float32)
        # Reset dictation state associated with the buffer being cleared
        # This seems safer here than in the trigger logic sometimes
        # self.isCurrentlySpeaking = False # Careful, might prematurely stop detecting if called mid-speech
        # self.silenceStartTime = None
