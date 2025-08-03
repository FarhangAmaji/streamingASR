import abc  # Abstract Base Classes
import gc
import os
import queue
import tempfile
import threading
import time
from pathlib import Path

import huggingface_hub
import keyboard
import numpy as np
import pyautogui
import pygame
import sounddevice as sd
import soundfile as sf
import torch
from transformers import pipeline

try:
    from nemo.collections.asr.models import ASRModel

    NEMO_AVAILABLE = True
except Exception:
    ASRModel = None  # Define as None if NeMo is not installed
    NEMO_AVAILABLE = False


def logDebug(message, debugPrintFlag):
    """Helper function for conditional debug printing."""
    if debugPrintFlag:
        print(f"DEBUG: {message}")


def logInfo(message):
    """Helper function for standard info messages."""
    print(f"INFO: {message}")


def logWarning(message):
    """Helper function for warning messages."""
    print(f"WARNING: {message}")


def logError(message):
    """Helper function for error messages."""
    print(f"ERROR: {message}")


class AbstractAsrModelHandler(abc.ABC):
    """
    Abstract Base Class defining the interface for ASR model handlers.
    Ensures that any ASR model implementation provides core functionalities.
    """

    def __init__(self, config):
        self.config = config
        self.modelLoaded = False
        self.asrPipeline = None  # The specific ASR object (e.g., Transformers pipeline)

    @abc.abstractmethod
    def loadModel(self):
        """Loads the ASR model into memory (CPU/GPU)."""
        pass

    @abc.abstractmethod
    def unloadModel(self):
        """Unloads the ASR model and cleans up resources."""
        pass

    @abc.abstractmethod
    def transcribeAudioSegment(self, audioData, sampleRate):
        """
        Transcribes a given audio data segment.

        Args:
            audioData (numpy.ndarray): The audio segment (Transcription Window).
            sampleRate (int): Sample rate of the audio data.

        Returns:
            str: The transcribed text, or an empty string on failure/if model not loaded.
        """
        pass

    def isModelLoaded(self):
        """Checks if the model is currently loaded."""
        return self.modelLoaded

    def getDevice(self):
        """Returns the compute device being used (e.g., 'cuda', 'cpu')."""
        return self.config.get('device')

    def cleanup(self):
        """Default cleanup action is to unload the model if loaded."""
        logDebug("AbstractAsrModelHandler cleanup initiated.", self.config.get('debugPrint'))
        if self.isModelLoaded():
            self.unloadModel()
        logDebug("AbstractAsrModelHandler cleanup complete.", self.config.get('debugPrint'))


class WhisperModelHandler(AbstractAsrModelHandler):
    """
    Concrete implementation of AbstractAsrModelHandler for Whisper models
    using the Hugging Face Transformers library.
    """

    def __init__(self, config):
        """
        Initializes the Whisper model handler.

        Args:
            config (ConfigurationManager): Application configuration object.
        """
        super().__init__(config)
        self._determineDevice()
        self.config.set('device', self.device)  # Update config with actual device

    def _determineDevice(self):
        """Determines the compute device (CUDA GPU or CPU)."""
        if self.config.get('onlyCpu'):
            self.device = torch.device('cpu')
            logInfo("CPU usage forced by 'onlyCpu=True'.")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cuda':
                logInfo("CUDA GPU detected and will be used.")
            else:
                logInfo("CUDA GPU not found or 'onlyCpu=True', using CPU.")

    def _logDebug(self, message):
        """Local debug logger using the config flag."""
        logDebug(message, self.config.get('debugPrint'))

    def _cudaClean(self):
        """Performs garbage collection and attempts to clear PyTorch's CUDA cache."""
        self._logDebug("Cleaning CUDA memory...")
        gc.collect()
        if torch.cuda.is_available() and self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as e:
                logWarning(f"CUDA memory cleaning attempt failed partially: {e}")
        self._logDebug("CUDA memory cleaning attempt finished.")

    def _monitorMemory(self):
        """Monitors and prints current GPU memory usage if debugPrint is enabled."""
        if torch.cuda.is_available() and self.config.get(
                'debugPrint') and self.device.type == 'cuda':
            try:
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            except Exception as e:
                logWarning(f"Failed to get GPU memory stats: {e}")

    def loadModel(self):
        """Loads the Whisper ASR model pipeline."""
        if self.modelLoaded:
            self._logDebug(f"Model '{self.config.get('modelName')}' already loaded.")
            return

        self._logDebug(f"Loading model '{self.config.get('modelName')}' to {self.device}...")
        self._monitorMemory()
        self._cudaClean()

        gen_kwargs = {"language": self.config.get('language')}
        gen_kwargs["return_timestamps"] = True
        self._logDebug(f"Pipeline generate_kwargs: {gen_kwargs}")

        try:
            self.asrPipeline = pipeline(
                "automatic-speech-recognition",
                model=self.config.get('modelName'),
                generate_kwargs=gen_kwargs,
                device=self.device
            )
            self.modelLoaded = True
            logInfo(f"Model '{self.config.get('modelName')}' loaded successfully.")
            self._warmUpModel()

        except Exception as e:
            logError(f"Failed loading ASR model '{self.config.get('modelName')}': {e}")
            logError("Check model name, internet connection, dependencies, and memory.")
            self.modelLoaded = False
            self.asrPipeline = None

        self._monitorMemory()

    def _warmUpModel(self):
        """Warms up the loaded model with a silent clip to reduce first inference latency."""
        if not self.modelLoaded or not self.asrPipeline:
            return
        try:
            self._logDebug("Warming up the model...")
            warmupSampleRate = 16000
            dummyAudio = np.zeros(warmupSampleRate, dtype=np.float32)  # 1 second silence
            _ = self.asrPipeline({"raw": dummyAudio, "sampling_rate": warmupSampleRate})
            self._logDebug("Model warm-up complete.")
        except Exception as e:
            logWarning(f"Model warm-up failed: {e}")

    def unloadModel(self):
        """Unloads the ASR model and cleans GPU cache."""
        if not self.modelLoaded:
            self._logDebug("Model already unloaded.")
            return

        self._logDebug(f"Unloading model '{self.config.get('modelName')}' from {self.device}...")
        if self.asrPipeline is not None:
            del self.asrPipeline
            self.asrPipeline = None

        self._cudaClean()  # Clean memory *after* deleting reference
        self.modelLoaded = False
        logInfo(f"Model '{self.config.get('modelName')}' unloaded.")
        self._monitorMemory()

    def transcribeAudioSegment(self, audioData, sampleRate):
        """Transcribes audio using the loaded Whisper pipeline."""
        if not self.modelLoaded or self.asrPipeline is None:
            self._logDebug("Transcription skipped: Model not loaded.")
            return ""
        if audioData is None or len(audioData) == 0:
            self._logDebug("Transcription skipped: No audio data provided.")
            return ""

        if len(audioData.shape) > 1 and audioData.shape[1] > 1:
            self._logDebug(f"Converting stereo audio (shape {audioData.shape}) to mono.")
            audioData = np.mean(audioData, axis=1)
        if audioData.dtype != np.float32:
            self._logDebug(f"Converting audio data type from {audioData.dtype} to float32.")
            audioData = audioData.astype(np.float32)

        transcription = ""
        try:
            segmentDurationSec = len(audioData) / sampleRate
            self._logDebug(
                f"Sending {segmentDurationSec:.2f}s audio Transcription Window to ASR...")

            asrInput = {"raw": audioData, "sampling_rate": sampleRate}
            result = self.asrPipeline(asrInput)
            self._logDebug("ASR pipeline processing finished.")
            self._logDebug(f"ASR Raw Result: {result}")  # Log raw result for debugging

            if isinstance(result, dict) and "text" in result:
                transcription = result["text"]
            else:
                logWarning(
                    f"Unexpected ASR result structure: {type(result)}. Could not extract text.")
                transcription = ""

        except Exception as e:
            logError(f"Error during Whisper transcription: {e}")
            transcription = ""

        return transcription

    @staticmethod
    def listAvailableModels():
        """Static method to retrieve Whisper/ASR models from Hugging Face Hub."""
        try:
            logInfo("Fetching list of available ASR models from Hugging Face Hub...")
            models = huggingface_hub.list_models(filter="automatic-speech-recognition")
            modelIds = [model.id for model in models]
            logInfo(f"Found {len(modelIds)} ASR models.")
            return modelIds
        except Exception as e:
            logError(f"Could not fetch models from Hugging Face Hub: {e}")
            return []


class NemoModelHandler(AbstractAsrModelHandler):
    """
    Concrete implementation of AbstractAsrModelHandler for NeMo ASR models
    provided by Nvidia.
    """

    def __init__(self, config):
        """
        Initializes the NeMo model handler.

        Args:
            config (ConfigurationManager): Application configuration object.
        """
        super().__init__(config)
        self._checkNemoAvailability()
        self._determineDevice()
        self.config.set('device', self.device)  # Update config with actual device
        self.model = None  # Explicitly define the model attribute

    def _checkNemoAvailability(self):
        """Checks if the NeMo toolkit is installed."""
        if not NEMO_AVAILABLE:
            errMsg = "NeMo toolkit not found, cannot use Nvidia ASR models. Install with: pip install nemo_toolkit[asr]"
            logError(errMsg)
            raise ImportError(errMsg)

    def _determineDevice(self):
        """Determines the compute device (CUDA GPU or CPU)."""
        if self.config.get('onlyCpu'):
            self.device = torch.device('cpu')
            logInfo("CPU usage forced by 'onlyCpu=True'.")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cuda':
                logInfo("CUDA GPU detected and will be used for NeMo model.")
            else:
                logInfo("CUDA GPU not found or 'onlyCpu=True', using CPU for NeMo model.")

    def _logDebug(self, message):
        """Local debug logger using the config flag."""
        logDebug(message, self.config.get('debugPrint'))

    def _cudaClean(self):
        """Performs garbage collection and attempts to clear PyTorch's CUDA cache."""
        self._logDebug("Cleaning CUDA memory (NeMo Handler)...")
        gc.collect()
        if torch.cuda.is_available() and self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logWarning(f"CUDA memory cleaning attempt failed partially (NeMo): {e}")
        self._logDebug("CUDA memory cleaning attempt finished (NeMo Handler).")

    def _monitorMemory(self):
        """Monitors and prints current GPU memory usage if debugPrint is enabled."""
        if torch.cuda.is_available() and self.config.get(
                'debugPrint') and self.device.type == 'cuda':
            try:
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            except Exception as e:
                logWarning(f"Failed to get GPU memory stats: {e}")

    def loadModel(self):
        """Loads the NeMo ASR model."""
        if self.modelLoaded:
            self._logDebug(f"NeMo model '{self.config.get('modelName')}' already loaded.")
            return
        if not NEMO_AVAILABLE:
            logError("Cannot load NeMo model: NeMo toolkit not installed.")
            return  # Prevent loading if dependency is missing

        self._logDebug(f"Loading NeMo model '{self.config.get('modelName')}' to {self.device}...")
        self._monitorMemory()
        self._cudaClean()  # Clean before loading

        try:
            self.model = ASRModel.from_pretrained(self.config.get('modelName')).to(self.device)
            self.model.eval()  # Set model to evaluation mode
            self.modelLoaded = True
            logInfo(f"NeMo model '{self.config.get('modelName')}' loaded successfully.")

        except Exception as e:
            logError(f"Failed loading NeMo ASR model '{self.config.get('modelName')}': {e}")
            logError(
                "Check model name, NeMo installation (pip install nemo_toolkit[asr]), internet connection, and memory.")
            self.modelLoaded = False
            self.model = None

        self._monitorMemory()  # Monitor after loading

    def unloadModel(self):
        """Unloads the NeMo ASR model and cleans GPU cache."""
        if not self.modelLoaded:
            self._logDebug("NeMo model already unloaded.")
            return

        self._logDebug(
            f"Unloading NeMo model '{self.config.get('modelName')}' from {self.device}...")
        if self.model is not None:
            del self.model
            self.model = None

        self._cudaClean()  # Clean memory *after* deleting reference
        self.modelLoaded = False
        logInfo(f"NeMo model '{self.config.get('modelName')}' unloaded.")
        self._monitorMemory()

    def transcribeAudioSegment(self, audioData, sampleRate):
        """
        Transcribes an audio data segment using the loaded NeMo model.

        Note: This implementation currently saves the audio segment to a
        temporary file, as NeMo's standard `transcribe` method often expects
        file paths. This might introduce I/O overhead for real-time streaming.
        Investigate model-specific direct tensor APIs for potential optimization.
        """
        if not self.modelLoaded or self.model is None:
            self._logDebug("NeMo transcription skipped: Model not loaded.")
            return ""
        if audioData is None or len(audioData) == 0:
            self._logDebug("NeMo transcription skipped: No audio data provided.")
            return ""

        if audioData.dtype != np.float32:
            self._logDebug(
                f"Converting audio data type from {audioData.dtype} to float32 for NeMo.")
            audioData = audioData.astype(np.float32)

        tempFilePath = None
        transcription = ""
        try:
            segmentDurationSec = len(audioData) / sampleRate
            self._logDebug(
                f"Sending {segmentDurationSec:.2f}s audio Transcription Window to NeMo ASR...")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpFile:
                tempFilePath = tmpFile.name
                sf.write(tmpFile.name, audioData, sampleRate, format='WAV', subtype='FLOAT')
                self._logDebug(f"Saved audio segment to temporary file: {tmpFile.name}")

            with torch.no_grad():  # Disable gradient calculations for inference
                transcriptions = self.model.transcribe(
                    paths2audio_files=[tempFilePath],
                    batch_size=1,  # Process one segment at a time
                    num_workers=0 if self.device.type == 'cpu' else 1  # Basic worker setting
                )

            self._logDebug(f"NeMo Raw Result: {transcriptions}")

            if transcriptions and isinstance(transcriptions, list):
                if transcriptions[0] and isinstance(transcriptions[0], list):
                    transcription = transcriptions[0][0]
                elif isinstance(transcriptions[0], str):
                    transcription = transcriptions[0]
                else:
                    logWarning(
                        f"Unexpected NeMo transcription result format inside list: {type(transcriptions[0])}")
                    transcription = ""
            elif isinstance(transcriptions,
                            str):  # Sometimes it might return a single string directly?
                transcription = transcriptions
            else:
                logWarning(
                    f"Unexpected NeMo transcription result format: {type(transcriptions)}. Could not extract text.")
                transcription = ""

        except Exception as e:
            logError(f"Error during NeMo transcription: {e}")
            import traceback
            logError(traceback.format_exc())  # Log full traceback for NeMo errors
            transcription = ""
        finally:
            if tempFilePath and os.path.exists(tempFilePath):
                try:
                    os.remove(tempFilePath)
                    self._logDebug(f"Removed temporary file: {tempFilePath}")
                except Exception as e:
                    logWarning(f"Failed to remove temporary file {tempFilePath}: {e}")

        return transcription.strip()  # Return stripped text


class ConfigurationManager:
    """Stores and provides access to all application settings."""

    def __init__(self, **kwargs):
        self._config = kwargs
        self._config['scriptDir'] = Path(os.path.dirname(os.path.abspath(__file__)))
        self._config['device'] = None  # Will be set by AsrModelHandler
        self._config['actualSampleRate'] = self._config.get('sampleRate', 16000)  # Default/Initial
        self._config['actualChannels'] = self._config.get('channels', 1)  # Default/Initial

    def get(self, key, default=None):
        """Gets a configuration value."""
        return self._config.get(key, default)

    def set(self, key, value):
        """Sets or updates a configuration value."""
        self._config[key] = value

    def getAll(self):
        """Returns the entire configuration dictionary."""
        return self._config.copy()


class StateManager:
    """Manages the dynamic state of the real-time transcriber."""

    def __init__(self, config):
        self.config = config
        self.isProgramActive = True  # Overall application loop control
        self.isRecordingActive = config.get('isRecordingActive', True)
        self.outputEnabled = config.get('outputEnabled', False)

        self.programStartTime = time.time()
        self.lastActivityTime = time.time()  # Used for model unloading timeout
        self.recordingStartTime = time.time() if self.isRecordingActive else 0
        self.lastValidTranscriptionTime = time.time()  # Used for consecutive idle timeout

    def _logDebug(self, message):
        logDebug(message, self.config.get('debugPrint'))

    def isRecording(self):
        return self.isRecordingActive

    def isOutputEnabled(self):
        return self.outputEnabled

    def shouldProgramContinue(self):
        return self.isProgramActive

    def startRecording(self):
        if not self.isRecordingActive:
            self._logDebug("Setting state to Recording: ON")
            self.isRecordingActive = True
            now = time.time()
            self.recordingStartTime = now
            self.lastActivityTime = now  # Mark activity for model manager
            self.lastValidTranscriptionTime = now  # Reset idle timer
            return True  # State changed
        return False  # No change

    def stopRecording(self):
        if self.isRecordingActive:
            self._logDebug("Setting state to Recording: OFF")
            self.isRecordingActive = False
            self.recordingStartTime = 0  # Reset session start time
            return True  # State changed
        return False  # No change

    def toggleOutput(self):
        self.outputEnabled = not self.outputEnabled
        status = 'enabled' if self.outputEnabled else 'disabled'
        self._logDebug(f"Setting state Output: {status.upper()}")
        logInfo(f"Output {status}")
        return self.outputEnabled  # Return new state

    def stopProgram(self):
        self._logDebug("Setting state Program Active: OFF")
        self.isProgramActive = False

    def updateLastActivityTime(self):
        """Updates the timestamp of the last significant activity."""
        self.lastActivityTime = time.time()

    def updateLastValidTranscriptionTime(self):
        """Updates the timestamp of the last valid transcription output."""
        self.lastValidTranscriptionTime = time.time()
        self._logDebug("Updated last valid transcription time (idle timer reset).")

    def checkProgramTimeout(self):
        """Checks if the maximum program duration has been exceeded."""
        elapsed = time.time() - self.programStartTime
        maxDuration = self.config.get('maxDurationProgramActive', 3600)
        if elapsed >= maxDuration:
            logInfo(f"Maximum program duration ({maxDuration}s) reached.")
            self.stopProgram()
            return True
        return False

    def checkRecordingTimeout(self):
        """Checks if the maximum recording session duration has been exceeded."""
        if not self.isRecordingActive or self.recordingStartTime == 0:
            return False
        elapsed = time.time() - self.recordingStartTime
        maxDuration = self.config.get('maxDurationRecording', 3600)
        if elapsed >= maxDuration:
            logInfo(f"Maximum recording session duration ({maxDuration}s) reached.")
            return True
        return False

    def checkIdleTimeout(self):
        """Checks if the consecutive idle time limit has been reached."""
        if not self.isRecordingActive:  # Only check if recording is supposed to be active
            return False
        silentFor = time.time() - self.lastValidTranscriptionTime
        idleTimeout = self.config.get('consecutiveIdleTime', 120)
        if silentFor >= idleTimeout:
            logInfo(f"Consecutive idle time ({idleTimeout}s) reached.")
            return True
        return False

    def timeSinceLastActivity(self):
        """Calculates the time elapsed since the last recorded activity."""
        return time.time() - self.lastActivityTime


class AudioHandler:
    """Manages audio input stream using sounddevice."""

    def __init__(self, config, stateManager):
        self.config = config
        self.stateManager = stateManager
        self.audioQueue = queue.Queue()
        self.stream = None
        self._setupDeviceInfo()  # Determine actual rates/channels

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
        self._printAudioDevices()  # Print devices for user convenience
        deviceId = self.config.get('deviceId')  # User might specify a device ID
        requestedRate = self.config.get('sampleRate')
        requestedChannels = self.config.get('channels')
        actualRate = requestedRate
        actualChannels = requestedChannels

        self._logDebug(
            f"Setting up device info. Requested device ID: {deviceId}, Rate: {requestedRate}, Channels: {requestedChannels}")
        try:
            deviceInfo = sd.query_devices(deviceId,
                                          kind='input')  # Query specific input device or default
            self._logDebug(f"Device info: {deviceInfo}")
            if isinstance(deviceInfo, dict):
                actualRate = int(deviceInfo.get("default_samplerate", requestedRate))
                actualChannels = min(requestedChannels,
                                     int(deviceInfo.get("max_input_channels", requestedChannels)))
            else:  # Fallback if query fails or returns unexpected format
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
                device=self.config.get('deviceId'),  # Use specific ID or None for default
                blocksize=self.config.get('blockSize', 1024),
                callback=self._audioCallback
            )
            self.stream.start()
            logInfo("Audio stream started successfully.")
            return True
        except Exception as e:
            logError(f"Failed to start audio stream: {e}")
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


class RealTimeAudioProcessor:
    """
    Handles audio buffer, processing modes (dictation, constant interval),
    and determines when transcription should be triggered based on audio.
    """

    def __init__(self, config, stateManager):
        self.config = config
        self.stateManager = stateManager
        self.audioBuffer = np.array([], dtype=np.float32)
        self.lastTranscriptionTriggerTime = time.time()  # For constant interval mode

        self.isCurrentlySpeaking = False
        self.silenceStartTime = None

    def _logDebug(self, message):
        logDebug(message, self.config.get('debugPrint'))

    def _calculateChunkLoudness(self, audioChunk):
        """Calculates the average absolute amplitude of an audio chunk."""
        if audioChunk is None or len(audioChunk) == 0:
            return 0.0
        return np.mean(np.abs(audioChunk))

    def processIncomingChunk(self, audioChunk):
        """Processes a new audio chunk: converts to mono, updates state, appends to buffer."""
        if audioChunk is None:
            return False  # No chunk processed

        monoChunk = audioChunk.flatten()
        if self.config.get('actualChannels') > 1:
            if len(audioChunk.shape) > 1 and audioChunk.shape[1] > 1:
                monoChunk = np.mean(audioChunk, axis=1)

        if self.config.get('transcriptionMode') == "dictationMode":
            self._updateDictationState(monoChunk)

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
            self.silenceStartTime = None  # Reset silence timer
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
            self.stateManager.updateLastActivityTime()  # Mark activity when sending data
            if mode == "constantIntervalMode":
                self.clearBuffer()  # Clear immediately for constant interval
                self.lastTranscriptionTriggerTime = time.time()  # Reset timer

            return audioDataToTranscribe
        else:
            if mode == "constantIntervalMode" and self._isConstantIntervalTimeReached():
                self.lastTranscriptionTriggerTime = time.time()
                self._logDebug("Constant interval reached, buffer empty. Resetting timer.")

        return None

    def _isConstantIntervalTimeReached(self):
        """Checks if the time interval for constant mode has passed."""
        interval = self.config.get('constantIntervalMode_transcriptionInterval', 3.0)
        return (time.time() - self.lastTranscriptionTriggerTime) >= interval

    def _checkTriggerConstantInterval(self):
        """Checks trigger conditions for constant interval mode."""
        if self._isConstantIntervalTimeReached() and len(self.audioBuffer) > 0:
            self._logDebug(
                f"Constant interval trigger. Processing buffer of {len(self.audioBuffer)} samples.")
            return self.audioBuffer.copy()
        return None

    def _checkTriggerDictationMode(self):
        """Checks trigger conditions for dictation mode."""
        if self.isCurrentlySpeaking and self.silenceStartTime is not None:
            elapsedSilence = time.time() - self.silenceStartTime
            requiredSilence = self.config.get('dictationMode_silenceDurationToOutput', 0.6)

            if elapsedSilence >= requiredSilence:
                self._logDebug(
                    f"Dictation mode trigger: Silence duration ({elapsedSilence:.2f}s) >= threshold ({requiredSilence}s).")
                if len(self.audioBuffer) > 0:
                    audioData = self.audioBuffer.copy()
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
                    return None  # Don't return empty data
        return None  # Conditions not met

    def getBufferDuration(self):
        """Calculates the duration of the current audio buffer in seconds."""
        if len(self.audioBuffer) == 0:
            return 0.0
        return len(self.audioBuffer) / self.config.get('actualSampleRate')

    def clearBufferIfOutputDisabled(self):
        """Clears the buffer if output is disabled."""
        if not self.stateManager.isOutputEnabled():
            self.clearBuffer()

    def clearBuffer(self):
        """Clears the internal audio buffer."""
        if len(self.audioBuffer) > 0:
            self._logDebug(f"Clearing audio buffer with {len(self.audioBuffer)} samples.")
        self.audioBuffer = np.array([], dtype=np.float32)


class TranscriptionOutputHandler:
    """Handles filtering, formatting, and outputting transcription results."""

    def __init__(self, config, stateManager, systemInteractionHandler):
        self.config = config
        self.stateManager = stateManager
        self.systemInteractionHandler = systemInteractionHandler  # For typing output

    def _logDebug(self, message):
        logDebug(message, self.config.get('debugPrint'))

    def _calculateSegmentLoudness(self, audioData):
        """Calculates the average absolute amplitude of the entire segment."""
        if audioData is None or len(audioData) == 0:
            return 0.0
        return np.mean(np.abs(audioData))

    def processTranscriptionResult(self, transcription, audioData):
        """
        Processes the ASR result: checks for silence, filters false positives,
        formats, and triggers output (print/type). Updates idle timer.
        """
        if audioData is None or len(audioData) == 0:
            self._logDebug("Processing skipped: No audio data provided with transcription.")
            self._handleSilentOrFilteredSegment()  # Treat as silent/filtered
            return

        segmentLoudness = self._calculateSegmentLoudness(audioData)
        self._logDebug(f"Processing transcription. Segment Avg Loudness = {segmentLoudness:.6f}")

        shouldOutput, finalText = self._filterAndFormatTranscription(transcription, segmentLoudness,
                                                                     audioData)

        if shouldOutput:
            self._handleValidOutput(finalText)
        else:
            self._handleSilentOrFilteredSegment()

    def _filterAndFormatTranscription(self, transcription, segmentLoudness, audioData):
        """
        Applies filtering rules (silence, false positives) and formatting to the raw transcription.

        Returns:
            tuple[bool, str]: (shouldOutput, formattedText)
                               - shouldOutput: True if the text should be outputted, False otherwise.
                               - formattedText: The final, cleaned text ready for output, or empty string.
        """
        cleanedText = transcription.strip() if isinstance(transcription, str) else ""
        if not cleanedText or cleanedText == ".":
            self._logDebug("Transcription is effectively empty after initial strip.")
            return False, ""  # Do not output empty or solely dot results.

        lowerCleanedText = cleanedText.lower()

        if self._shouldSkipTranscriptionDueToSilenceOrLowContent(segmentLoudness, audioData):
            return False, ""  # Do not output if determined to be silence or insufficient content.

        if self._isFalsePositive(lowerCleanedText, segmentLoudness):
            return False, ""  # Do not output if filtered as a false positive.

        formattedText = cleanedText
        if self.config.get('removeTrailingDots'):
            formattedText = formattedText.rstrip('. ')

        formattedText = formattedText.lstrip(" ")

        if not formattedText:
            self._logDebug("Text became empty after final formatting steps.")
            return False, ""

        self._logDebug(f"Final formatted text ready for output: '{formattedText}'")
        return True, formattedText

    def _shouldSkipTranscriptionDueToSilenceOrLowContent(self, segmentMeanLoudness, audioData):
        """
        Checks if the transcription should be ignored based on combined silence and
        minimum content duration rules.
        Returns True if the segment SHOULD be skipped, False otherwise.
        """
        if audioData is None or len(audioData) == 0:
            self._logDebug("Silence check skipped: No audio data.")
            return True  # Skip if no audio

        sampleRate = self.config.get('actualSampleRate')
        chunkSilenceThreshold = self.config.get('dictationMode_silenceLoudnessThreshold', 0.001)
        minLoudDuration = self.config.get('minLoudDurationForTranscription',
                                          0.6)  # New minimum duration threshold
        silenceSkipThreshold = self.config.get('silenceSkip_threshold',
                                               0.0002)  # Average loudness threshold
        checkLeadingSec = self.config.get('skipSilence_beforeNSecSilence', 0.0)
        checkTrailingSec = self.config.get('skipSilence_afterNSecSilence', 0.3)

        if minLoudDuration > 0:
            loudSamplesMask = np.abs(audioData) >= chunkSilenceThreshold
            numLoudSamples = np.sum(loudSamplesMask)
            totalLoudDuration = numLoudSamples / sampleRate

            if totalLoudDuration < minLoudDuration:
                self._logDebug(
                    f"Silence skip CONFIRMED: Total duration of loud samples ({totalLoudDuration:.2f}s) "
                    f"is less than the required minimum ({minLoudDuration:.2f}s). "
                    f"(Segment Avg Loudness: {segmentMeanLoudness:.6f})")
                return True  # SKIP: Not enough potentially meaningful content

        if segmentMeanLoudness >= silenceSkipThreshold:
            return False  # DO NOT SKIP

        self._logDebug(
            f"Segment passed min loud duration but mean loudness ({segmentMeanLoudness:.6f}) is below skip threshold ({silenceSkipThreshold:.6f}). Checking start/end overrides...")

        if checkLeadingSec > 0:
            leadingSamples = int(checkLeadingSec * sampleRate)
            leadingAudio = audioData[:leadingSamples]
            if len(leadingAudio) > 0:  # Avoid division by zero if leadingSamples=0 or audioData empty
                leadingLoudness = np.mean(np.abs(leadingAudio))
                if leadingLoudness >= chunkSilenceThreshold:
                    self._logDebug(
                        f"Silence skip OVERRIDDEN (Low Avg): Segment has sufficient total loud duration, "
                        f"and leading {checkLeadingSec:.2f}s loudness ({leadingLoudness:.6f}) >= "
                        f"chunk threshold ({chunkSilenceThreshold:.6f}).")
                    return False  # DO NOT SKIP

        if checkTrailingSec > 0:
            trailingSamples = int(checkTrailingSec * sampleRate)
            if len(audioData) >= trailingSamples:  # Check length for negative index
                trailingAudio = audioData[-trailingSamples:]
                trailingLoudness = np.mean(np.abs(trailingAudio))
                if trailingLoudness >= chunkSilenceThreshold:
                    self._logDebug(
                        f"Silence skip OVERRIDDEN (Low Avg): Segment has sufficient total loud duration, "
                        f"and trailing {checkTrailingSec:.2f}s loudness ({trailingLoudness:.6f}) >= "
                        f"chunk threshold ({chunkSilenceThreshold:.6f}).")
                    return False  # DO NOT SKIP

        self._logDebug(
            f"Silence skip CONFIRMED (Low Avg, No Overrides): Segment passed min loud duration but average loudness is low, "
            f"and loudness at start ({checkLeadingSec:.2f}s) and end ({checkTrailingSec:.2f}s) did not override.")
        return True  # SKIP the segment

    def _isFalsePositive(self, lowerCleanedText, segmentLoudness):
        """
        Checks if the transcription is a common false word detected in low loudness.
        Cleans the input text further (removing punctuation) for more robust matching.
        """
        commonFalseWords = self.config.get('commonFalseDetectedWords', [])
        if not commonFalseWords:
            return False

        import string

        translator = str.maketrans('', '', string.punctuation)
        checkText = lowerCleanedText.translate(translator).strip()
        checkText = ' '.join(checkText.split())

        normalizedFalseWords = [w.lower() for w in commonFalseWords]

        if checkText in normalizedFalseWords:
            loudnessThreshold = self.config.get('loudnessThresholdOf_commonFalseDetectedWords',
                                                0.0008)
            if segmentLoudness < loudnessThreshold:
                self._logDebug(
                    f"'{checkText}' IS considered a false positive (Loudness {segmentLoudness:.6f} < {loudnessThreshold:.6f}). Filtering.")
                return True
            else:
                self._logDebug(
                    f"'{checkText}' matches a false positive word BUT loudness ({segmentLoudness:.6f}) >= threshold ({loudnessThreshold:.6f}). Not filtering.")

        return False

    def _handleValidOutput(self, finalText):
        """Handles actions for valid, filtered transcription text."""
        print("Transcription:", finalText)  # Print to console

        if self.stateManager.isOutputEnabled() and not self.systemInteractionHandler.isModifierKeyPressed(
                "ctrl"):
            self.systemInteractionHandler.typeText(finalText + " ")  # Add trailing space

        self.stateManager.updateLastValidTranscriptionTime()

    def _handleSilentOrFilteredSegment(self):
        """Handles actions when transcription is empty, silent, or filtered."""
        pass


class SystemInteractionHandler:
    """Manages interactions with keyboard for hotkeys and pygame for sound."""

    def __init__(self, config):
        self.config = config
        self.audioFiles = {}
        self.isMixerInitialized = False
        self._setupAudioNotifications()

    def _logDebug(self, message):
        logDebug(message, self.config.get('debugPrint'))

    def _setupAudioNotifications(self):
        """Initializes pygame mixer and loads sound file paths."""
        soundMap = {
            "modelUnloaded": "modelUnloaded.mp3",
            "outputDisabled": "outputDisabled.mp3",
            "outputEnabled": "outputEnabled.mp3",
            "recordingOff": "recordingOff.mp3",
            "recordingOn": "recordingOn.mp3"
        }
        scriptDir = self.config.get('scriptDir')
        try:
            pygame.mixer.init()
            self.isMixerInitialized = True
            logInfo("Pygame mixer initialized for audio notifications.")
            for name, filename in soundMap.items():
                path = scriptDir / filename
                if path.is_file():
                    self.audioFiles[name] = str(path)
                else:
                    logWarning(f"Notification sound file not found: {path}")
        except pygame.error as e:
            logWarning(f"Failed to initialize pygame mixer: {e}. Notification sounds disabled.")
            self.isMixerInitialized = False
        except Exception as e:
            logError(f"Unexpected error during notification setup: {e}")
            self.isMixerInitialized = False

    def playNotification(self, soundName):
        """Plays a notification sound if available and audio notifications are globally enabled."""
        if not self.config.get('enableAudioNotifications', False):  # Default to False if not set
            self._logDebug(
                f"Skipping sound '{soundName}' because enableAudioNotifications is False.")
            return  # Exit early if all notifications are disabled

        if soundName in ['recordingOn', 'outputEnabled'] and not self.config.get('playEnableSounds',
                                                                                 False):
            self._logDebug(
                f"Skipping '{soundName}' because playEnableSounds is False.")
            return

        if not self.isMixerInitialized or soundName not in self.audioFiles:
            self._logDebug(
                f"Cannot play sound '{soundName}'. Mixer initialized: {self.isMixerInitialized}, Sound exists: {soundName in self.audioFiles}")
            return

        soundPath = self.audioFiles[soundName]
        try:
            sound = pygame.mixer.Sound(soundPath)
            sound.play()
            self._logDebug(f"Played notification sound: {soundName}")
        except Exception as e:
            logError(f"Error playing notification sound '{soundPath}': {e}")

    def monitorKeyboardShortcuts(self, orchestrator):
        """
        Runs in a thread to monitor global hotkeys. Calls methods on the orchestrator.
        """
        logInfo("Starting keyboard shortcut monitor thread.")
        threadStartTime = time.time()
        maxDuration = self.config.get('maxDurationProgramActive', 3600)
        recordingToggleKey = self.config.get('recordingToggleKey')
        outputToggleKey = self.config.get('outputToggleKey')

        while orchestrator.stateManager.shouldProgramContinue() and \
                (time.time() - threadStartTime) < maxDuration:
            try:
                if keyboard.is_pressed(recordingToggleKey):
                    self._logDebug(f"Hotkey '{recordingToggleKey}' pressed.")
                    orchestrator.toggleRecording()  # Call orchestrator method
                    self._waitForKeyRelease(recordingToggleKey)

                if keyboard.is_pressed(outputToggleKey):
                    self._logDebug(f"Hotkey '{outputToggleKey}' pressed.")
                    orchestrator.toggleOutput()  # Call orchestrator method
                    self._waitForKeyRelease(outputToggleKey)

            except ImportError:
                logError("Keyboard library not installed or permission denied. Hotkeys disabled.")
                logError("Try running with sudo (Linux/macOS) or installing 'keyboard'.")
                break  # Stop monitoring if library fails
            except Exception as e:
                logError(f"Error in keyboard monitoring thread: {e}")
                time.sleep(1)  # Avoid busy-looping on repeated errors

            time.sleep(0.05)  # Prevent high CPU usage

        logInfo("Keyboard shortcut monitor thread stopping.")
        orchestrator.stateManager.stopProgram()

    def _waitForKeyRelease(self, key):
        """Waits until the specified key is released to prevent rapid toggling."""
        startTime = time.time()
        timeout = 2.0  # Add a timeout to prevent getting stuck
        while keyboard.is_pressed(key):
            if time.time() - startTime > timeout:
                self._logDebug(f"Warning: Timeout waiting for key release '{key}'.")
                break
            time.sleep(0.05)
        self._logDebug(f"Hotkey '{key}' released.")

    def isModifierKeyPressed(self, key):
        """Checks if a specific modifier key (e.g., 'ctrl', 'alt', 'shift') is pressed."""
        try:
            return keyboard.is_pressed(key)
        except Exception as e:
            self._logDebug(f"Could not check modifier key '{key}': {e}")
            return False

    def typeText(self, text):
        """Uses pyautogui to simulate typing text."""
        try:
            pyautogui.write(text)
        except NameError:
            logError("PyAutoGUI library not found. Install it to enable typing output.")
        except Exception as e:
            logWarning(f"PyAutoGUI write failed: {e}")

    def cleanup(self):
        """Cleans up system interaction resources (pygame mixer)."""
        logDebug("SystemInteractionHandler cleanup.", self.config.get('debugPrint'))
        if self.isMixerInitialized:
            try:
                pygame.mixer.quit()
                logInfo("Pygame mixer quit.")
            except Exception as e:
                logError(f"Error quitting pygame mixer: {e}")


class ModelLifecycleManager:
    """Handles automatic loading and unloading of the ASR model based on activity."""

    def __init__(self, config, stateManager, asrModelHandler, systemInteractionHandler):
        self.config = config
        self.stateManager = stateManager
        self.asrModelHandler = asrModelHandler
        self.systemInteractionHandler = systemInteractionHandler  # For unload notification

    def _logDebug(self, message):
        logDebug(message, self.config.get('debugPrint'))

    def manageModelLifecycle(self):
        """
        Runs in a thread to monitor activity and load/unload the model.
        """
        logInfo("Starting Model Lifecycle Manager thread.")
        checkInterval = 10  # Seconds

        while self.stateManager.shouldProgramContinue():
            isRecording = self.stateManager.isRecording()
            modelLoaded = self.asrModelHandler.isModelLoaded()
            unloadTimeout = self.config.get('model_unloadTimeout', 1200)

            if not isRecording and modelLoaded:
                timeInactive = self.stateManager.timeSinceLastActivity()
                if timeInactive >= unloadTimeout:
                    self._logDebug(
                        f"Model inactive for {timeInactive:.1f}s (>= {unloadTimeout}s), unloading...")
                    self.asrModelHandler.unloadModel()
                    self.systemInteractionHandler.playNotification("modelUnloaded")

            if isRecording and not modelLoaded:
                logInfo("Recording active but model not loaded. Triggering model load...")
                self.asrModelHandler.loadModel()
                self.stateManager.updateLastActivityTime()

            startTime = time.time()
            while (
                    time.time() - startTime < checkInterval) and self.stateManager.shouldProgramContinue():
                time.sleep(0.5)  # Sleep in smaller chunks

        logInfo("Model Lifecycle Manager thread stopping.")


class FileTranscriber:
    """Handles transcription of pre-recorded audio files using a provided ASR handler."""

    def __init__(self, config, asrModelHandler):
        """
        Initialize the file transcriber.

        Args:
             config (ConfigurationManager): Application configuration object.
             asrModelHandler (AbstractAsrModelHandler): The ASR model handler to use.
        """
        self.config = config
        self.asrModelHandler = asrModelHandler

    def _logDebug(self, message):
        logDebug(message, self.config.get('debugPrint'))

    def transcribeFile(self, audioFilePath, outputFilePath=None):
        """
        Transcribe an audio file and optionally save the transcription.

        Args:
            audioFilePath (str | Path): Path to the input audio file.
            outputFilePath (str | Path, optional): Path to save the transcription text file.
                                                   If None, prints to console. Defaults to None.

        Returns:
            str | None: Transcribed text, or None if transcription fails.
        """
        self._logDebug(f"Attempting to transcribe file: {audioFilePath}")
        transcription = None
        audioFilePath = Path(audioFilePath)  # Ensure Path object

        try:
            if not self.asrModelHandler.isModelLoaded():
                logInfo("Model not loaded for file transcription, loading now...")
                self.asrModelHandler.loadModel()
            if not self.asrModelHandler.isModelLoaded():
                logError("Model could not be loaded. File transcription aborted.")
                return None

            if not audioFilePath.is_file():
                raise FileNotFoundError(f"Audio file not found at {audioFilePath}")

            audioData, sampleRate = sf.read(audioFilePath, dtype='float32',
                                            always_2d=False)  # Ensure float32, prefer 1D
            fileDuration = len(audioData) / sampleRate
            logInfo(
                f"File read successfully. Sample Rate: {sampleRate}, Duration: {fileDuration:.2f}s")

            transcription = self.asrModelHandler.transcribeAudioSegment(audioData, sampleRate)

            if self.config.get('removeTrailingDots') and isinstance(transcription, str):
                transcription = transcription.strip('. ')

            if transcription is not None:
                self._handleOutput(transcription, outputFilePath)
            else:
                logWarning("Transcription failed (ASR handler returned None or empty).")

            return transcription

        except FileNotFoundError as e:
            logError(str(e))
            return None
        except Exception as e:
            logError(f"Error transcribing file '{audioFilePath}': {e}")
            import traceback
            logError(traceback.format_exc())  # Log full traceback for file errors
            return None

    def _handleOutput(self, transcription, outputFilePath):
        """Saves transcription to file or prints to console."""
        if outputFilePath:
            try:
                outputFilePath = Path(outputFilePath)
                outputFilePath.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                with open(outputFilePath, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                logInfo(f"Transcription saved to: {outputFilePath}")
            except IOError as e:
                logError(f"Error writing transcription to file {outputFilePath}: {e}")
                print("\n--- Transcription ---")  # Print as fallback
                print(transcription)
                print("---------------------\n")
        else:
            print("\n--- Transcription ---")
            print(transcription)
            print("---------------------\n")

    def cleanup(self):
        """Cleans up resources (delegated to ASR handler)."""
        logInfo("FileTranscriber cleanup initiated.")
        logInfo("FileTranscriber cleanup complete.")


class SpeechToTextOrchestrator:
    """
    Main class orchestrating the real-time speech-to-text process.
    Connects and manages all components (Audio, State, Processing, Output, System, Model).
    Handles transcription in a separate thread to maintain main thread responsiveness.
    """

    def __init__(self, **userConfig):
        self.config = ConfigurationManager(**userConfig)
        self._logDebug = lambda msg: logDebug(msg, self.config.get('debugPrint'))
        self._logDebug("Initializing speechToText Orchestrator...")

        self.stateManager = StateManager(self.config)

        modelName = self.config.get('modelName', '')
        modelNameLower = modelName.lower()

        if modelNameLower.startswith("nvidia/"):
            logInfo(
                f"Detected Nvidia model prefix. Attempting to use NemoModelHandler for '{modelName}'.")
            if NEMO_AVAILABLE:
                try:
                    self.asrModelHandler = NemoModelHandler(self.config)
                except ImportError as e:
                    logError(f"Failed to instantiate NemoModelHandler: {e}")
                    logError(
                        "Falling back to whisperModelHandler - This may fail if the model is truly NeMo-only.")
                    self.asrModelHandler = WhisperModelHandler(self.config)  # Fallback attempt
                except Exception as e:
                    logError(f"Unexpected error instantiating NemoModelHandler: {e}")
                    logError(
                        "Falling back to whisperModelHandler - This may fail if the model is truly NeMo-only.")
                    self.asrModelHandler = WhisperModelHandler(self.config)  # Fallback attempt
            else:
                logError(
                    f"Cannot use model '{modelName}': NeMo toolkit not installed (required for 'nvidia/' prefix).")
                logError("Attempting to load with whisperModelHandler - This is unlikely to work.")
                self.asrModelHandler = WhisperModelHandler(self.config)
        elif "whisper" in modelNameLower or \
                "canary" in modelNameLower or \
                "parakeet" in modelNameLower:
            logInfo(f"Using whisperModelHandler for '{modelName}'.")
            self.asrModelHandler = WhisperModelHandler(self.config)
        else:
            logWarning(
                f"Model name '{modelName}' doesn't match known patterns for Nvidia or specific HF models.")
            logWarning("Attempting to load with whisperModelHandler as a default.")
            self.asrModelHandler = WhisperModelHandler(self.config)

        self.systemInteractionHandler = SystemInteractionHandler(self.config)
        self.audioHandler = AudioHandler(self.config, self.stateManager)
        self.realTimeProcessor = RealTimeAudioProcessor(self.config, self.stateManager)
        self.outputHandler = TranscriptionOutputHandler(self.config, self.stateManager,
                                                        self.systemInteractionHandler)
        self.modelLifecycleManager = ModelLifecycleManager(self.config, self.stateManager,
                                                           self.asrModelHandler,
                                                           self.systemInteractionHandler)

        self.transcriptionRequestQueue = queue.Queue()

        self.threads = []  # To keep track of background threads

        self._printInitialInstructions()

    def _printInitialInstructions(self):
        """Prints setup info and user instructions."""
        deviceStr = str(self.config.get('device')) if self.config.get(
            'device') else "CPU/GPU (Auto)"
        handlerType = type(self.asrModelHandler).__name__

        print(f"\n--- Configuration ---")
        print(f"Mode: {self.config.get('transcriptionMode')}")
        print(
            f"ASR Model: {self.config.get('modelName')} (Using Handler: {handlerType}, Target Device: {deviceStr})")
        print(
            f"Audio Device: ID={self.config.get('deviceId') or 'Default'}, Rate={self.config.get('actualSampleRate')}Hz, Channels={self.config.get('actualChannels')}")
        print(f"--- Hotkeys ---")
        print(f"Toggle Recording: '{self.config.get('recordingToggleKey')}'")
        print(f"Toggle Text Output: '{self.config.get('outputToggleKey')}'")
        print(f"--- Timeouts ---")
        print(f"Max Recording Duration: {self.config.get('maxDurationRecording')} s")
        print(f"Stop Recording After Silence: {self.config.get('consecutiveIdleTime')} s")
        print(f"Unload Model After Inactivity: {self.config.get('modelUnloadTimeout')} s")
        print(f"Program Auto-Exit After: {self.config.get('maxDurationProgramActive')} s")
        print(f"------------------\n")

    def _transcriptionWorkerLoop(self):
        """
        Worker loop running in a separate thread.
        Waits for audio data on the queue, transcribes it, and handles output.
        """
        logInfo("Starting Transcription Worker thread.")
        queueTimeoutSeconds = 1.0  # How long to wait for an item before checking program status

        while self.stateManager.shouldProgramContinue():
            try:
                queueItem = self.transcriptionRequestQueue.get(timeout=queueTimeoutSeconds)

                if queueItem is None:  # Use None as a potential sentinel, though timeout is primary exit now
                    self._logDebug("Transcription worker received None sentinel, exiting.")
                    break

                audioDataToTranscribe, sampleRate = queueItem

                if self.asrModelHandler.isModelLoaded():
                    self._logDebug(
                        f"Transcription worker processing {len(audioDataToTranscribe)} samples...")
                    transcriptionResult = self.asrModelHandler.transcribeAudioSegment(
                        audioDataToTranscribe,
                        sampleRate
                    )
                    self.outputHandler.processTranscriptionResult(
                        transcriptionResult,
                        audioDataToTranscribe  # Pass audio data for loudness checks etc.
                    )
                    self._logDebug("Transcription worker finished processing segment.")
                else:
                    logWarning("Transcription worker skipped segment: ASR model is not loaded.")
                    if self.config.get('transcriptionMode') == 'dictationMode':
                        self._logDebug(
                            "Dictation mode state might need reset due to unloaded model during transcription request.")

                self.transcriptionRequestQueue.task_done()  # Signal queue item processed

            except queue.Empty:
                continue
            except Exception as e:
                logError(f"!!! ERROR in Transcription Worker thread: {e}")
                import traceback
                logError(traceback.format_exc())
                time.sleep(1)

        logInfo("Transcription Worker thread stopping.")

    def _startBackgroundThreads(self):
        """Starts threads for hotkeys, model management, and transcription."""
        self._logDebug("Starting background threads...")
        self.threads = []  # Clear list

        keyboardThread = threading.Thread(
            target=self.systemInteractionHandler.monitorKeyboardShortcuts,
            args=(self,),
            name="KeyboardMonitorThread",
            daemon=True
        )
        self.threads.append(keyboardThread)
        keyboardThread.start()

        modelThread = threading.Thread(
            target=self.modelLifecycleManager.manageModelLifecycle,
            name="ModelManagerThread",
            daemon=True
        )
        self.threads.append(modelThread)
        modelThread.start()

        transcriptionThread = threading.Thread(
            target=self._transcriptionWorkerLoop,  # Target the new worker method
            name="TranscriptionWorkerThread",
            daemon=True  # Ensure it exits if main thread exits unexpectedly
        )
        self.threads.append(transcriptionThread)
        transcriptionThread.start()

        self._logDebug(f"Started {len(self.threads)} background threads.")

    def toggleRecording(self):
        """Toggles the recording state. Called by systemInteractionHandler."""
        if self.stateManager.isRecording():
            if self.stateManager.stopRecording():  # If state actually changed
                self.systemInteractionHandler.playNotification("recordingOff")
                logInfo("Recording stopped via hotkey.")
                self.realTimeProcessor.clearBuffer()
                self.audioHandler.clearQueue()
        else:
            if self.stateManager.startRecording():  # If state actually changed
                self.systemInteractionHandler.playNotification("recordingOn")
                logInfo("Recording started via hotkey.")

    def toggleOutput(self):
        """Toggles the text output state. Called by systemInteractionHandler."""
        newState = self.stateManager.toggleOutput()
        notification = "outputEnabled" if newState else "outputDisabled"
        self.systemInteractionHandler.playNotification(notification)

    def _cleanup(self):
        """Cleans up all resources."""
        logInfo("Initiating cleanup...")
        self.stateManager.stopProgram()  # Signal all loops to stop
        self.stateManager.stopRecording()

        logInfo("Stopping audio stream during cleanup...")
        self.audioHandler.stopStream()

        logInfo("Clearing buffers and queues during cleanup...")
        self.realTimeProcessor.clearBuffer()
        self.audioHandler.clearQueue()

        logInfo("Waiting briefly for background threads to stop...")
        time.sleep(0.5)  # Give threads a moment to react to stateManager.shouldProgramContinue()

        if self.asrModelHandler:
            logInfo("Cleaning up ASR model handler...")
            self.asrModelHandler.cleanup()

        logInfo("Cleaning up system interaction handler...")
        self.systemInteractionHandler.cleanup()

        logInfo("Attempting to join background threads...")
        joinTimeout = 1.5  # Increased slightly to allow transcription thread potential work
        for t in self.threads:
            if t is not None and t.is_alive():
                try:
                    t.join(timeout=joinTimeout)
                    if t.is_alive():
                        logWarning(f"Thread {t.name} did not terminate within {joinTimeout}s.")
                except Exception as e:
                    logWarning(f"Error joining thread {t.name}: {e}")
        self.threads = []

        logInfo("Program cleanup complete.")

    def _run_initialSetup(self):
        """Handle initial setup steps including model loading and thread start."""
        if self.stateManager.isRecording() and not self.asrModelHandler.isModelLoaded() and self.asrModelHandler.modelLoaded is not None:
            logInfo("Initial state is recording: ensuring model is loaded...")
            self.asrModelHandler.loadModel()
            if not self.asrModelHandler.isModelLoaded():
                logError(
                    "CRITICAL: Initial model load failed. Recording disabled. Please check logs and model compatibility with the handler.")
                self.stateManager.stopRecording()

        self._startBackgroundThreads()  # Now starts 3 threads

        initialStreamStarted = False
        if self.stateManager.isRecording():
            logInfo("Initial state is recording: attempting to start audio stream...")
            initialStreamStarted = self.audioHandler.startStream()
            if not initialStreamStarted:
                logError(
                    "CRITICAL: Failed to start audio stream initially. Recording disabled.")
                self.stateManager.stopRecording()

    def _run_checkTimeoutsNGlobalState(self):
        """Check program and recording timeouts and manage global state."""
        if self.stateManager.checkProgramTimeout():
            logInfo("Maximum program duration reached.")
            return False

        self.realTimeProcessor.clearBufferIfOutputDisabled()

        if self.stateManager.isRecording():
            if self.stateManager.checkRecordingTimeout():
                logInfo("Recording session timeout reached, stopping recording...")
                self.toggleRecording()
            elif self.stateManager.checkIdleTimeout():
                logInfo("Idle timeout reached, stopping recording...")
                self.toggleRecording()

        return True

    def _run_manageAudioStreamLifecycle(self):
        """Manage audio stream lifecycle based on recording state."""
        shouldStreamBeActive = self.stateManager.isRecording()
        isStreamActuallyActive = self.audioHandler.stream is not None and self.audioHandler.stream.active

        if shouldStreamBeActive and not isStreamActuallyActive:
            self._logDebug("Attempting to start audio stream (recording active)...")
            if not self.audioHandler.startStream():
                logError("Failed to start/restart audio stream. Disabling recording.")
                self.stateManager.stopRecording()
                time.sleep(1)
                return False

        elif not shouldStreamBeActive and isStreamActuallyActive:
            self._logDebug("Stopping audio stream (recording inactive)...")
            self.audioHandler.stopStream()
            self.realTimeProcessor.clearBuffer()
            self.audioHandler.clearQueue()

        return True

    def _run_processAudioChunks(self):
        """Process audio chunks from the queue when recording is active."""
        shouldStreamBeActive = self.stateManager.isRecording()
        isStreamActuallyActive = self.audioHandler.stream is not None and self.audioHandler.stream.active

        if shouldStreamBeActive and isStreamActuallyActive:
            audioProcessedThisLoop = False
            while True:
                chunk = self.audioHandler.getAudioChunk()
                if chunk is None:
                    break
                if self.realTimeProcessor.processIncomingChunk(chunk):
                    audioProcessedThisLoop = True

            return audioProcessedThisLoop
        else:
            return False

    def _run_queueTranscriptionRequest(self, audioProcessedThisLoop):
        """Checks if transcription should be triggered and queues the request for the worker thread."""
        if not self.stateManager.isOutputEnabled():
            return  # Skip queueing if output is disabled

        if audioProcessedThisLoop:  # Check if audio buffer was potentially updated
            audioDataToTranscribe = self.realTimeProcessor.checkTranscriptionTrigger()

            if audioDataToTranscribe is not None:
                try:
                    sampleRate = self.config.get('actualSampleRate')
                    queueItem = (audioDataToTranscribe, sampleRate)
                    self.transcriptionRequestQueue.put(queueItem)
                    self._logDebug(
                        f"Queued transcription request for {len(audioDataToTranscribe)} samples.")
                except Exception as e:
                    logError(f"Failed to queue transcription request: {e}")

    def _run_loopSleep(self):
        """Sleep briefly to prevent high CPU usage, especially if idle."""
        time.sleep(0.01)

    def run(self):
        """Main execution loop orchestrating the real-time transcription process."""
        logInfo("Starting main orchestrator loop...")
        try:
            self._run_initialSetup()  # Includes starting all threads now

            while self.stateManager.shouldProgramContinue():
                if not self._run_checkTimeoutsNGlobalState():
                    break

                if not self._run_manageAudioStreamLifecycle():
                    continue

                audioProcessedThisLoop = self._run_processAudioChunks()

                self._run_queueTranscriptionRequest(audioProcessedThisLoop)

                self._run_loopSleep()

        except KeyboardInterrupt:
            logInfo("\nKeyboardInterrupt received. Stopping...")
            self.stateManager.stopProgram()
        except Exception as e:
            logError(f"\n!!! UNEXPECTED ERROR in main orchestrator loop: {e}")
            import traceback
            logError(traceback.format_exc())
            self.stateManager.stopProgram()
        finally:
            self._cleanup()


if __name__ == "__main__":

    userSettings = {
        "modelName": "nvidia/canary-180m-flash",
        "language": "en",
        "onlyCpu": False,

        "transcriptionMode": "dictationMode",
        "dictationMode_silenceDurationToOutput": 0.6,
        "dictationMode_silenceLoudnessThreshold": 0.0004,
        "constantIntervalMode_transcriptionInterval": 4.0,

        "minLoudDurationForTranscription": 0.3,
        "silenceSkip_threshold": 0.0002,
        "skipSilence_beforeNSecSilence": 0.3,
        "skipSilence_afterNSecSilence": 0.3,
        "commonFalseDetectedWords": ["you", "thank you", "bye", 'amen'],
        "loudnessThresholdOf_commonFalseDetectedWords": 0.00045,

        "removeTrailingDots": True,
        "outputEnabled": False,
        "isRecordingActive": True,
        "enableAudioNotifications": True,
        "playEnableSounds": False,

        "recordingToggleKey": "win+alt+l",
        "outputToggleKey": "ctrl+q",

        "maxDuration_recording": 10000,
        "maxDuration_programActive": 2 * 60 * 60,
        "model_unloadTimeout": 10 * 60,
        "consecutiveIdleTime": 2 * 60,

        "sampleRate": 16000,
        "channels": 1,
        "blockSize": 1024,
        "deviceId": None,

        "debugPrint": True
    }

    try:
        orchestrator = SpeechToTextOrchestrator(**userSettings)
        orchestrator.run()


    except Exception as e:
        logError(f"\n!!! PROGRAM CRITICAL ERROR during setup: {e}")
        import traceback

        logError(traceback.format_exc())
