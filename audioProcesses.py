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
import time
import logging  # Needed for checking log level
import traceback  # For detailed error logging

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
    print(f"ERROR: Failed to import or initialize sounddevice: {e}", file=sys.stderr)

# Import logging helpers from utils
from utils import logWarning, logDebug, logInfo, logError


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
            logError(
                "CRITICAL: sounddevice library not found or failed to initialize. Audio input disabled.")
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
            # Log devices clearly, perhaps one per line if many
            if devices:
                logInfo(f"\n{devices}")  # Log the full device list object representation
                # Or iterate for more detail:
                # for i, device in enumerate(devices):
                #     logInfo(f"Device {i}: {device['name']} (In: {device['max_input_channels']}, Out: {device['max_output_channels']}, Rate: {device['default_samplerate']})")
            else:
                logInfo("No audio devices found by sounddevice.")
        except Exception as e:
            logError(f"Could not query audio devices: {e}", exc_info=True)
        logInfo("--------------------------------------")

    def _setupDeviceInfo(self):
        """Queries audio device info and sets actual sample rate/channels in config."""
        if not sounddeviceAvailable: return

        self._printAudioDevices()  # Log available devices first
        deviceId = self.config.get('deviceId')  # Can be None, int, or string part
        requestedRate = self.config.get('sampleRate', 16000)
        requestedChannels = self.config.get('channels', 1)
        actualRate = requestedRate  # Default to requested
        actualChannels = requestedChannels  # Default to requested

        logDebug(
            f"Setting up device info. Requested Device ID: {deviceId}, Rate: {requestedRate}, Channels: {requestedChannels}")
        try:
            # Query device details based on the requested ID (or default if None)
            # sounddevice handles None deviceId to mean default input device
            deviceInfo = sd.query_devices(device=deviceId, kind='input')

            # deviceInfo can be dict or None if device not found/no input
            if isinstance(deviceInfo, dict):
                logDebug(f"Device info found: {deviceInfo}")
                # Use device's default sample rate if our requested one isn't ideal? Risky.
                # Stick to requested rate unless known issues. Let's store device defaults though.
                defaultRate = deviceInfo.get("default_samplerate")
                maxInChannels = deviceInfo.get("max_input_channels")

                # Validate requested channels against device max input channels
                if maxInChannels is not None and requestedChannels > maxInChannels:
                    logWarning(
                        f"Requested {requestedChannels} channels, but device '{deviceInfo.get('name')}' only supports up to {maxInChannels}. Using {maxInChannels}.")
                    actualChannels = maxInChannels
                elif requestedChannels < 1:
                    logWarning(
                        f"Requested invalid channel count ({requestedChannels}). Using 1 (mono).")
                    actualChannels = 1
                else:
                    actualChannels = requestedChannels  # Use requested if valid

                # Let's use the requested sample rate for now, sounddevice will raise error if unsupported
                actualRate = requestedRate
                logInfo(
                    f"Selected Device: '{deviceInfo.get('name')}', Max Input Channels: {maxInChannels}, Default Rate: {defaultRate}")

            else:  # Device not found or no default input
                logWarning(
                    f"Could not retrieve detailed info for requested device ID '{deviceId}'. Using configured defaults or system default.")
                # Keep requested values, sounddevice might still find a default
                actualRate = requestedRate
                actualChannels = requestedChannels
                if actualChannels < 1: actualChannels = 1  # Ensure at least 1 channel

        except ValueError as e:
            # sd.query_devices raises ValueError if device ID is invalid format or not found
            logError(f"Error querying audio device '{deviceId}': {e}. Check device ID/name.")
            logError("Using configured defaults/system default, but stream start might fail.")
            # Keep requested values as fallback
            actualRate = requestedRate
            actualChannels = requestedChannels
            if actualChannels < 1: actualChannels = 1
        except Exception as e:
            logError(f"Unexpected error querying audio device information: {e}", exc_info=True)
            # Keep requested values as fallback
            actualRate = requestedRate
            actualChannels = requestedChannels
            if actualChannels < 1: actualChannels = 1

        # Store the determined 'actual' values back into configuration for other components
        self.config.set('actualSampleRate', actualRate)
        self.config.set('actualChannels', actualChannels)
        logInfo(f"Audio configured for: Rate={actualRate} Hz, Channels={actualChannels}")

    def _audioCallback(self, indata: np.ndarray, frames: int, timeInfo, status: sd.CallbackFlags):
        """
        Callback function executed by sounddevice stream thread for each audio block.
        Adds received audio data to the queue if recording is active.
        """
        if status:  # Log any status flags (e.g., input overflow/underflow)
            logWarning(f"Audio callback status flags: {status}")
        # Add data to queue only if application logic wants recording active
        if self.stateManager and self.stateManager.isRecording():
            # indata is a numpy array. Make a copy to avoid issues if sounddevice reuses buffer.
            try:
                # Simple put, RealTimeAudioProcessor handles consumption rate
                self.audioQueue.put_nowait(indata.copy())
            except queue.Full:
                # Handle queue full scenario - drop oldest? Log warning?
                logWarning("Audio input queue is full! Potential audio data loss.")
                # Example: Drop oldest item to make space for newest
                try:
                    self.audioQueue.get_nowait()  # Remove oldest
                    self.audioQueue.put_nowait(indata.copy())  # Add newest
                except queue.Empty:
                    pass  # Should not happen if full, but handle defensively
                except Exception as qe:
                    logError(f"Error managing full audio queue: {qe}")

    def startStream(self) -> bool:
        """Starts the sounddevice input stream if not already active."""
        if not sounddeviceAvailable:
            logError("Cannot start audio stream: sounddevice library not available.")
            return False

        if self.stream is not None and self.stream.active:
            logDebug("Audio stream is already active.")
            return True  # Already running

        # Ensure previous stream (if any) is properly closed before starting new one
        if self.stream is not None:
            self.stopStream()  # Attempt cleanup of previous stream

        try:
            # Get configured parameters
            rate = self.config.get('actualSampleRate')
            channels = self.config.get('actualChannels')
            deviceId = self.config.get('deviceId')  # Can be None
            blockSize = self.config.get('blockSize', 0)  # 0 lets sounddevice choose optimal

            logInfo(
                f"Attempting to start audio stream (Device: {deviceId or 'Default'}, Rate: {rate}, Channels: {channels}, BlockSize: {blockSize or 'Auto'})...")

            # Clear the queue before starting
            self.clearQueue()

            self.stream = sd.InputStream(
                samplerate=rate,
                channels=channels,
                device=deviceId,
                blocksize=blockSize,
                dtype='float32',  # Explicitly request float32, common for ASR
                callback=self._audioCallback
            )
            self.stream.start()  # Start the stream & callback mechanism
            # Store stream info
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
            # Often indicates incompatible parameters (rate, channels, device)
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
                self.streamInfo = {}  # Clear info
                # Clear the queue after stopping to discard remaining data
                self.clearQueue()
        elif self.stream is not None and not self.stream.active:
            logDebug("Audio stream object exists but is not active. Closing.")
            try:
                self.stream.close()  # Close even if not active
            except Exception:
                pass  # Ignore errors closing already closed stream
            self.stream = None
            self.streamInfo = {}
            self.clearQueue()  # Clear queue as well
        else:
            logDebug("Audio stream already stopped or not initialized.")

    def getAudioChunk(self) -> np.ndarray | None:
        """Retrieves the next available audio chunk from the queue (non-blocking)."""
        try:
            # Get audio chunk without waiting
            return self.audioQueue.get_nowait()
        except queue.Empty:
            # This is expected when no new audio has arrived
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
            # Efficiently clear the queue
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
        # Main buffer to accumulate audio chunks between transcription triggers
        self.audioBuffer = np.array([], dtype=np.float32)
        # Timestamp for constant interval mode trigger logic
        self.lastTranscriptionTriggerTime = time.time()

        # --- Dictation mode specific state ---
        self.isCurrentlySpeaking = False  # Flag if audio is currently above silence threshold
        self.silenceStartTime = None  # Timestamp when silence started *after* speech

        logDebug("RealTimeAudioProcessor initialized.")

    def _calculateChunkLoudness(self, audioChunk: np.ndarray) -> float:
        """Calculates the average absolute amplitude (proxy for loudness) of an audio chunk."""
        if audioChunk is None or audioChunk.size == 0:
            return 0.0
        # Ensure calculation is done on float data
        if audioChunk.dtype.kind != 'f':
            # Basic normalization if int type (e.g., int16)
            if audioChunk.dtype.kind in ('i', 'u'):
                max_val = np.iinfo(audioChunk.dtype).max
                min_val = np.iinfo(audioChunk.dtype).min
                if max_val > min_val:
                    audioChunk = (audioChunk.astype(np.float32) - min_val) / (
                                max_val - min_val) * 2.0 - 1.0
                else:
                    audioChunk = audioChunk.astype(np.float32)
            else:
                audioChunk = audioChunk.astype(np.float32)  # Just cast other types

        # Use Root Mean Square (RMS) for a better loudness measure? Optional.
        # rms = np.sqrt(np.mean(audioChunk**2))
        # return rms
        # Or stick to mean absolute amplitude (simpler)
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
            logDebug("processIncomingChunk received empty chunk, skipping.")
            return False  # No chunk processed

        processedChunk = audioChunk  # Start with the input chunk

        # --- Ensure Float32 Audio ---
        if processedChunk.dtype != np.float32:
            logDebug(f"Chunk received with dtype {processedChunk.dtype}, converting to float32.")
            try:
                if processedChunk.dtype.kind in ('i', 'u'):
                    max_val = np.iinfo(processedChunk.dtype).max
                    min_val = np.iinfo(processedChunk.dtype).min
                    if max_val > min_val:
                        processedChunk = (processedChunk.astype(np.float32) - min_val) / (
                                    max_val - min_val) * 2.0 - 1.0
                    else:
                        processedChunk = processedChunk.astype(np.float32)
                else:
                    processedChunk = processedChunk.astype(np.float32)
            except Exception as e:
                logError(f"Failed converting chunk to float32: {e}")
                return False

        # --- Ensure Mono Audio ---
        numChannels = self.config.get('actualChannels', 1)
        if numChannels > 1:
            if len(processedChunk.shape) > 1 and processedChunk.shape[1] == numChannels:
                logDebug(f"Averaging {numChannels} channels to mono.")
                processedChunk = np.mean(processedChunk, axis=1)
            elif len(processedChunk.shape) == 1 and numChannels > 1:
                logWarning(
                    f"Received 1D audio data but expected {numChannels} channels. Proceeding as mono.")

        # --- *** NEW: Ensure processedChunk is 1D before concatenation *** ---
        if processedChunk.ndim > 1:
            # Example: Flatten a potential (N, 1) shape to (N,)
            logDebug(f"Flattening processed chunk from {processedChunk.shape} to 1D.")
            processedChunk = processedChunk.flatten()
        # --- *** End of Change *** ---

        # Ensure chunk is not empty after processing
        if processedChunk is None or processedChunk.size == 0:
            logWarning("Audio chunk became empty after processing (float/mono/flatten).")
            return False

        # --- Update Dictation Mode State (if applicable) ---
        if self.config.get('transcriptionMode') == "dictationMode":
            # Pass the potentially flattened chunk here
            self._updateDictationState(processedChunk)

        # --- Append Processed Chunk to Main Buffer ---
        try:
            # Now concatenation should work as both self.audioBuffer and processedChunk are 1D
            self.audioBuffer = np.concatenate((self.audioBuffer, processedChunk))
            return True
        except ValueError as e:
            # Log specific error if it still occurs (shouldn't for dimensions now)
            logError(
                f"Error concatenating audio chunk to buffer: {e}. Buffer shape: {self.audioBuffer.shape}, Chunk shape: {processedChunk.shape}")
            return False
        except Exception as e:
            logError(f"Unexpected error appending chunk to buffer: {e}", exc_info=True)
            return False

    def _updateDictationState(self, monoChunk: np.ndarray):
        """Updates the speaking flag and silence timer for dictation mode based on chunk loudness."""
        chunkLoudness = self._calculateChunkLoudness(monoChunk)
        # Use a configurable threshold for detecting 'silence' within dictation mode
        silenceThreshold = self.config.get('dictationMode_silenceLoudnessThreshold',
                                           0.001)  # Example default

        if chunkLoudness >= silenceThreshold:
            # Audio is above threshold - considered speaking
            if not self.isCurrentlySpeaking:
                logDebug(
                    f"Speech detected (Chunk Loudness {chunkLoudness:.6f} >= {silenceThreshold:.6f})")
            self.isCurrentlySpeaking = True
            # If we were timing silence, reset the timer because speech resumed
            if self.silenceStartTime is not None:
                logDebug("Speech resumed, resetting silence timer.")
                self.silenceStartTime = None
        else:
            # Audio is below threshold - considered silence for this chunk
            # Only start timing silence *if* we were previously speaking
            if self.isCurrentlySpeaking and self.silenceStartTime is None:
                logDebug(
                    f"Silence detected after speech (Chunk Loudness {chunkLoudness:.6f} < {silenceThreshold:.6f}). Starting silence timer ({self.config.get('dictationMode_silenceDurationToOutput', 0.6)}s)...")
                self.silenceStartTime = time.time()
            # If silence continues while timer is running, do nothing here - let checkTranscriptionTrigger handle it.
            # If silence occurs when not previously speaking, also do nothing (ignore leading silence).

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

        # --- Route to mode-specific trigger logic ---
        if mode == "constantIntervalMode":
            audioDataToTranscribe = self._checkTriggerConstantInterval()
        elif mode == "dictationMode":
            audioDataToTranscribe = self._checkTriggerDictationMode()
        else:
            logWarning(
                f"Unsupported transcriptionMode: {mode}. No transcription will be triggered.")
            return None  # Unknown mode

        # --- Process Trigger Result ---
        if audioDataToTranscribe is not None:
            # Ensure we have substantial audio data (avoid tiny fragments unless intended)
            # Example: Check minimum duration? Or handled by filtering later?
            # Let's assume filtering handles very short segments, just check non-empty here.
            if audioDataToTranscribe.size == 0:
                logDebug("Trigger occurred but yielded empty audio data. Ignoring.")
                return None

            logInfo(
                f"Triggering transcription ({mode}). Buffer duration: {len(audioDataToTranscribe) / self.config.get('actualSampleRate', 1):.2f}s.")
            # Mark activity when preparing data for ASR
            if self.stateManager: self.stateManager.updateLastActivityTime()

            # Clear buffer / reset state specific to the mode *after* getting data
            if mode == "constantIntervalMode":
                self.clearBuffer()  # Clear entire buffer
                self.lastTranscriptionTriggerTime = time.time()  # Reset timer
            elif mode == "dictationMode":
                # Buffer and state reset happens inside _checkTriggerDictationMode
                pass

            return audioDataToTranscribe  # Return the data segment
        else:
            # No trigger condition met for the current mode
            # Optionally reset constant interval timer even if buffer empty
            if mode == "constantIntervalMode" and self._isConstantIntervalTimeReached():
                self.lastTranscriptionTriggerTime = time.time()
                # logDebug("Constant interval reached, buffer empty. Resetting timer.") # Can be noisy

        return None  # No trigger

    def _isConstantIntervalTimeReached(self) -> bool:
        """Helper method to check if the time interval for constant mode has passed."""
        interval = self.config.get('constantIntervalMode_transcriptionInterval', 3.0)  # Default 3s
        if interval <= 0: return False  # Interval disabled
        return (time.time() - self.lastTranscriptionTriggerTime) >= interval

    def _checkTriggerConstantInterval(self) -> np.ndarray | None:
        """Checks trigger conditions for constant interval mode."""
        if self._isConstantIntervalTimeReached() and self.audioBuffer.size > 0:
            logDebug(
                f"Constant interval trigger condition met. Buffer size: {self.audioBuffer.size} samples.")
            # Return a copy of the current buffer
            return self.audioBuffer.copy()
        return None  # Not time yet, or buffer is empty

    def _checkTriggerDictationMode(self) -> np.ndarray | None:
        """Checks trigger conditions for dictation mode (silence after speech)."""
        # Trigger only if:
        # 1. We detected speech (isCurrentlySpeaking was True)
        # 2. Silence started after that speech (silenceStartTime is not None)
        # 3. Enough time has passed since silence started
        if self.isCurrentlySpeaking and self.silenceStartTime is not None:
            requiredSilence = self.config.get('dictationMode_silenceDurationToOutput',
                                              0.6)  # Default 0.6s
            if requiredSilence <= 0: return None  # Feature disabled

            elapsedSilence = time.time() - self.silenceStartTime

            if elapsedSilence >= requiredSilence:
                # Silence duration met! Trigger transcription.
                logDebug(
                    f"Dictation mode trigger: Silence duration ({elapsedSilence:.2f}s) >= threshold ({requiredSilence}s).")
                if self.audioBuffer.size > 0:
                    # Get the audio data *before* clearing buffer/state
                    audioData = self.audioBuffer.copy()
                    # Reset state for the next utterance *now*
                    self.clearBuffer()
                    self.isCurrentlySpeaking = False
                    self.silenceStartTime = None
                    logDebug("Dictation mode state reset after successful trigger.")
                    return audioData
                else:
                    # Trigger met, but somehow buffer is empty? Should not happen if isCurrentlySpeaking was true.
                    logWarning(
                        "Dictation mode trigger met, but audio buffer is empty. Resetting state anyway.")
                    self.clearBuffer()
                    self.isCurrentlySpeaking = False
                    self.silenceStartTime = None
                    return None  # Return None as there's no data
        return None  # Conditions not met

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
                self.clearBuffer()  # Resets buffer and dictation state if needed

    def clearBuffer(self):
        """Clears the internal audio buffer and resets associated dictation state."""
        bufferLen = self.audioBuffer.size
        if bufferLen > 0:
            sampleRate = self.config.get('actualSampleRate', 16000)  # Use actual or default
            duration = bufferLen / sampleRate if sampleRate > 0 else 0
            logDebug(f"Clearing audio buffer with {bufferLen} samples ({duration:.2f}s).")
        self.audioBuffer = np.array([], dtype=np.float32)  # Reset to empty float32 array

        # Also reset dictation state when buffer is explicitly cleared
        # This prevents stale state if cleared manually or due to output disable
        if self.isCurrentlySpeaking or self.silenceStartTime is not None:
            logDebug("Resetting dictation state due to buffer clear.")
            self.isCurrentlySpeaking = False
            self.silenceStartTime = None
