# modelHandlers.py
import abc  # Abstract Base Classes
import gc
import json

import huggingface_hub
import numpy as np
import requests  # For client-server communication
# Pygame and PyAutoGUI are imported conditionally later where needed
import torch
from transformers import pipeline

from utils import logWarning, logDebug, logInfo, logError


# ==================================
# ASR Model Handling (Abstraction)
# ==================================

class AbstractAsrModelHandler(abc.ABC):
    """
    Abstract Base Class defining the interface for ASR model handlers.
    Implementations handle specific ASR libraries/models (local or remote client).
    """

    def __init__(self, config):
        self.config = config
        self.modelLoaded = False  # Status flag for subclasses
        self._logDebug = lambda msg: logDebug(msg, self.config.get('debugPrint'))

    @abc.abstractmethod
    def loadModel(self):
        """Loads the ASR model into memory (local) or ensures server connection (remote)."""
        pass

    @abc.abstractmethod
    def unloadModel(self):
        """Unloads the ASR model (local) or potentially signals server (remote)."""
        pass

    @abc.abstractmethod
    def transcribeAudioSegment(self, audioData, sampleRate):
        """
        Transcribes a given audio data segment.

        Args:
            audioData (numpy.ndarray): The audio segment (float32, mono expected).
            sampleRate (int): Sample rate of the audio data.

        Returns:
            str: The transcribed text, or an empty string on failure.
        """
        pass

    def isModelLoaded(self):
        """Checks if the model is considered loaded/ready."""
        return self.modelLoaded

    def getDevice(self):
        """Returns the compute device being used ('cuda', 'cpu', 'remote', etc.)."""
        # Subclasses should implement this to report accurately.
        return self.config.get('device', 'unknown')

    def cleanup(self):
        """Default cleanup action is to unload the model if loaded."""
        self._logDebug(f"{type(self).__name__} cleanup initiated.")
        if self.isModelLoaded():
            self.unloadModel()
        self._logDebug(f"{type(self).__name__} cleanup complete.")


# ==================================
# Whisper/Transformers Implementation (Local)
# ==================================
class WhisperModelHandler(AbstractAsrModelHandler):
    """
    Concrete implementation for Whisper models using Hugging Face Transformers.
    Runs the model locally on the machine executing this script.
    """

    def __init__(self, config):
        super().__init__(config)
        self.asrPipeline = None
        self.device = None
        self._determineDevice()
        self.config.set('device', str(self.device))  # Update config with actual device string

    def _determineDevice(self):
        """Determines the compute device (CUDA GPU or CPU) for local execution."""
        if self.config.get('onlyCpu'):
            self.device = torch.device('cpu')
            logInfo("CPU usage forced by 'onlyCpu=True' for Whisper model.")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cuda':
                logInfo("CUDA GPU detected and will be used for Whisper model.")
            else:
                logInfo("CUDA GPU not found or 'onlyCpu=True', using CPU for Whisper model.")

    def _cudaClean(self):
        """Performs garbage collection and attempts to clear PyTorch's CUDA cache."""
        self._logDebug("Cleaning CUDA memory (Whisper Handler)...")
        gc.collect()
        if self.device and self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                # Optional: synchronize and collect ipc handles if issues persist
                # torch.cuda.synchronize()
                # torch.cuda.ipc_collect()
            except Exception as e:
                logWarning(f"CUDA memory cleaning attempt failed partially (Whisper): {e}")
        self._logDebug("CUDA memory cleaning attempt finished (Whisper Handler).")

    def _monitorMemory(self):
        """Monitors and prints current GPU memory usage if debugPrint is enabled."""
        if self.config.get('debugPrint') and self.device and self.device.type == 'cuda':
            try:
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                print(
                    f"GPU Memory (Whisper) - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            except Exception as e:
                logWarning(f"Failed to get GPU memory stats: {e}")

    def loadModel(self):
        """Loads the Whisper ASR model pipeline locally."""
        if self.modelLoaded:
            self._logDebug(f"Whisper model '{self.config.get('modelName')}' already loaded.")
            return

        modelName = self.config.get('modelName')
        if not modelName:
            logError("Cannot load Whisper model: 'modelName' not specified in config.")
            return

        self._logDebug(f"Loading Whisper model '{modelName}' locally to {self.device}...")
        self._monitorMemory()
        self._cudaClean()  # Clean before loading

        # Prepare generation arguments
        genKwargs = {"language": self.config.get('language')}
        # Consider adding task (transcribe/translate) if needed, based on config
        # genKwargs["task"] = self.config.get('whisperTask', 'transcribe')
        # Timestamps can be useful for post-processing or word timings if required
        genKwargs["return_timestamps"] = self.config.get('whisperReturnTimestamps',
                                                         False)  # Default False unless needed
        self._logDebug(f"Pipeline generate_kwargs: {genKwargs}")

        try:
            self.asrPipeline = pipeline(
                "automatic-speech-recognition",
                model=modelName,
                generate_kwargs=genKwargs,
                device=self.device,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
                # Use float16 on GPU if available
            )
            self.modelLoaded = True
            logInfo(f"Whisper model '{modelName}' loaded successfully locally on {self.device}.")
            self._warmUpModel()  # Warm up after loading

        except Exception as e:
            logError(f"Failed loading local Whisper model '{modelName}': {e}")
            logError(
                "Check model name, internet connection (for download), dependencies, and memory.")
            # More specific errors
            if "requires the PyTorch library" in str(e):
                logError("Hint: Ensure PyTorch is installed correctly (`pip install torch`).")
            elif "safetensors_rust" in str(e):
                logError("Hint: Ensure `safetensors` is installed (`pip install safetensors`).")
            self.modelLoaded = False
            self.asrPipeline = None
            self._cudaClean()  # Clean up potential partial load

        self._monitorMemory()  # Monitor after load attempt

    def _warmUpModel(self):
        """Warms up the loaded model with a silent clip to reduce first inference latency."""
        if not self.modelLoaded or not self.asrPipeline:
            return
        try:
            self._logDebug("Warming up the Whisper model...")
            # Whisper expects 16kHz sample rate
            warmupSampleRate = 16000
            dummyAudio = np.zeros(warmupSampleRate, dtype=np.float32)  # 1 second silence
            # Prepare input in the format the pipeline expects
            asrInput = {"raw": dummyAudio, "sampling_rate": warmupSampleRate}
            # Execute inference
            _ = self.asrPipeline(asrInput)
            self._logDebug("Whisper model warm-up complete.")
        except Exception as e:
            logWarning(f"Whisper model warm-up failed: {e}")

    def unloadModel(self):
        """Unloads the local Whisper ASR model and cleans GPU cache."""
        if not self.modelLoaded:
            self._logDebug("Whisper model already unloaded.")
            return

        modelName = self.config.get('modelName')
        self._logDebug(f"Unloading Whisper model '{modelName}' from {self.device}...")
        if self.asrPipeline is not None:
            # Explicitly delete the pipeline object
            try:
                del self.asrPipeline.model  # Try deleting inner model first if possible
            except AttributeError:
                pass  # Ignore if structure is different
            del self.asrPipeline
            self.asrPipeline = None

        self.modelLoaded = False
        self._cudaClean()  # Clean memory *after* deleting reference
        logInfo(f"Whisper model '{modelName}' unloaded.")
        self._monitorMemory()

    def transcribeAudioSegment(self, audioData, sampleRate):
        """Transcribes audio using the loaded local Whisper pipeline."""
        if not self.modelLoaded or self.asrPipeline is None:
            self._logDebug("Whisper transcription skipped: Model not loaded.")
            return ""
        if audioData is None or len(audioData) == 0:
            self._logDebug("Whisper transcription skipped: No audio data provided.")
            return ""

        # Pre-process: Ensure float32 (already done by RealTimeAudioProcessor usually)
        #             Ensure mono (already done by RealTimeAudioProcessor usually)
        #             Check sample rate (Whisper expects 16kHz, but pipeline might handle resampling)
        if sampleRate != 16000:
            logWarning(
                f"Whisper model prefers 16kHz, received {sampleRate}Hz. Pipeline *should* handle resampling.")
            # If issues occur, manual resampling might be needed here using librosa or similar.

        transcription = ""
        try:
            segmentDurationSec = len(audioData) / sampleRate if sampleRate > 0 else 0
            self._logDebug(
                f"Sending {segmentDurationSec:.2f}s audio segment to local Whisper pipeline...")

            # Prepare input dictionary
            asrInput = {"raw": audioData, "sampling_rate": sampleRate}

            # Perform transcription
            result = self.asrPipeline(asrInput)
            # self._logDebug(f"Whisper Raw Result: {result}") # Can be verbose

            # Extract text - structure might vary slightly based on args (e.g., with timestamps)
            if isinstance(result, dict) and "text" in result:
                transcription = result["text"].strip()
            elif isinstance(result, str):  # Sometimes pipeline might return just the string
                transcription = result.strip()
            else:
                logWarning(
                    f"Unexpected Whisper result structure: {type(result)}. Could not extract text.")
                transcription = ""

        except Exception as e:
            logError(f"Error during local Whisper transcription: {e}")
            import traceback
            logError(traceback.format_exc())  # Log full traceback for easier debugging
            transcription = ""
            # Attempt to clean GPU memory if a CUDA error occurred
            if self.device.type == 'cuda' and 'cuda' in str(e).lower():
                self._cudaClean()

        return transcription  # Already stripped in extraction logic

    def getDevice(self):
        """Returns the compute device being used (e.g., 'cuda', 'cpu')."""
        return str(self.device) if self.device else 'unknown'

    @staticmethod
    def listAvailableModels():
        """Static method to retrieve Whisper/ASR models from Hugging Face Hub."""
        try:
            logInfo("Fetching list of available ASR models from Hugging Face Hub...")
            models = huggingface_hub.list_models(filter="automatic-speech-recognition",
                                                 sort="downloads", direction=-1)
            modelIds = [model.id for model in models]
            logInfo(f"Found {len(modelIds)} ASR models on Hub.")
            return modelIds
        except Exception as e:
            logError(f"Could not fetch models from Hugging Face Hub: {e}")
            return []


# ==================================
# Remote NeMo Client Implementation
# ==================================
class RemoteNemoClientHandler(AbstractAsrModelHandler):
    """
    Concrete implementation that acts as a client to a remote ASR server (WSL).
    It sends audio data via HTTP requests and receives transcription results.
    It does *not* load the NeMo model itself; the server does.
    """

    def __init__(self, config):
        super().__init__(config)
        self.serverUrl = config.get('wslServerUrl')
        if not self.serverUrl:
            logError(
                "RemoteNemoClientHandler cannot operate: 'wslServerUrl' not found in configuration.")
            # We cannot function without the URL, but don't raise error here,
            # let methods fail gracefully if called.
            self.serverReachable = False
        else:
            self.serverReachable = None  # Unknown until first request
            logInfo(f"RemoteNemoClientHandler initialized. Target server URL: {self.serverUrl}")
        # The 'modelLoaded' status here reflects reachability/status of the *server's* model
        self.modelLoaded = False  # Assume not loaded initially
        self.config.set('device', 'remote_wsl')  # Indicate execution happens remotely

    def _makeServerRequest(self, method, endpoint, **kwargs):
        """Helper function to make requests to the WSL server."""
        if not self.serverUrl:
            logError("Cannot make server request: Server URL not configured.")
            return None

        url = f"{self.serverUrl.rstrip('/')}/{endpoint.lstrip('/')}"
        self._logDebug(f"Sending {method.upper()} request to server: {url}")
        timeoutSeconds = self.config.get('serverRequestTimeout', 10.0)  # Configurable timeout

        try:
            response = requests.request(method, url, timeout=timeoutSeconds, **kwargs)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            # Mark server as reachable on successful communication
            if self.serverReachable is not True:
                logInfo(f"Successfully connected to WSL server at {self.serverUrl}.")
                self.serverReachable = True

            try:
                return response.json()  # Assume server sends JSON
            except json.JSONDecodeError:
                logError(f"Server response from {url} is not valid JSON: {response.text[:100]}...")
                return None  # Treat non-JSON response as an error

        except requests.exceptions.ConnectionError as e:
            logError(f"Connection Error connecting to WSL server at {url}: {e}")
            if self.serverReachable is not False:
                logError(
                    "Hint: Ensure the wslNemoServer.py script is running in WSL and accessible.")
            self.serverReachable = False
            self.modelLoaded = False  # Assume model unloaded if server unreachable
            return None
        except requests.exceptions.Timeout:
            logError(f"Request Timeout connecting to WSL server at {url} (>{timeoutSeconds}s).")
            self.serverReachable = False  # Potentially reachable but slow
            self.modelLoaded = False
            return None
        except requests.exceptions.RequestException as e:
            logError(f"Error during request to WSL server at {url}: {e}")
            # Log response body if available and useful
            if hasattr(e, 'response') and e.response is not None:
                logError(f"Server Response ({e.response.status_code}): {e.response.text[:200]}...")
            self.serverReachable = False  # General request error
            self.modelLoaded = False
            return None

    def loadModel(self):
        """Attempts to tell the remote server to load the model (if not already loaded)."""
        self._logDebug("Checking remote server model status...")
        # Check status first
        statusResponse = self._makeServerRequest('get', '/status')
        if statusResponse and statusResponse.get('status') == 'loaded':
            self._logDebug("Remote NeMo model is already loaded on server.")
            self.modelLoaded = True
            return

        # If not loaded or status check failed, attempt to trigger load
        self._logDebug("Requesting remote server to load NeMo model...")
        loadResponse = self._makeServerRequest('post', '/load')  # Assuming POST triggers load

        if loadResponse and loadResponse.get('status') == 'loaded':
            logInfo(
                f"Successfully requested remote server to load model '{loadResponse.get('modelName', 'N/A')}'.")
            self.modelLoaded = True
        elif loadResponse:
            logError(
                f"Remote server reported an issue during load: {loadResponse.get('message', 'Unknown error')}")
            self.modelLoaded = False
        else:
            # Error already logged by _makeServerRequest
            logError("Failed to trigger model load on remote server (communication error).")
            self.modelLoaded = False

    def unloadModel(self):
        """Attempts to tell the remote server to unload the model."""
        self._logDebug("Requesting remote server to unload NeMo model...")
        unloadResponse = self._makeServerRequest('post', '/unload')  # Assuming POST triggers unload

        if unloadResponse and unloadResponse.get('status') == 'unloaded':
            logInfo("Successfully requested remote server to unload model.")
            self.modelLoaded = False
        elif unloadResponse:
            logWarning(
                f"Remote server reported an issue during unload: {unloadResponse.get('message', 'Unknown error')}")
            # State might be ambiguous, assume unloaded for safety
            self.modelLoaded = False
        else:
            # Error already logged by _makeServerRequest
            logWarning(
                "Failed to trigger model unload on remote server (communication error). State unknown.")
            # Assume unloaded for safety
            self.modelLoaded = False

    def transcribeAudioSegment(self, audioData, sampleRate):
        """Sends audio data to the remote server for transcription."""
        if not self.serverReachable and self.serverReachable is not None:  # Check reachability if known
            self._logDebug("Remote transcription skipped: Server known to be unreachable.")
            return ""
        if audioData is None or len(audioData) == 0:
            self._logDebug("Remote transcription skipped: No audio data provided.")
            return ""

        segmentDurationSec = len(audioData) / sampleRate if sampleRate > 0 else 0
        self._logDebug(
            f"Sending {segmentDurationSec:.2f}s audio segment to remote NeMo server for transcription...")

        # Ensure audioData is float32 bytes
        if audioData.dtype != np.float32:
            audioData = audioData.astype(np.float32)
        audioBytes = audioData.tobytes()

        # Prepare request data and parameters
        files = {'audio_data': ('audio.bin', audioBytes, 'application/octet-stream')}
        params = {'sample_rate': sampleRate}

        # Make the request
        transcribeResponse = self._makeServerRequest('post', '/transcribe', params=params,
                                                     files=files)

        if transcribeResponse and 'transcription' in transcribeResponse:
            transcription = transcribeResponse['transcription']
            # Update model status based on successful transcription
            if not self.modelLoaded:
                self._logDebug("Received transcription, marking remote model as loaded.")
                self.modelLoaded = True
            self._logDebug(f"Received transcription from server: '{transcription[:100]}...'")
            return transcription.strip()  # Server should ideally strip, but ensure here
        else:
            logError("Failed to get valid transcription from remote server.")
            # Assume model might have unloaded or server failed
            self.modelLoaded = False
            return ""

    def isModelLoaded(self):
        """Returns the last known status of the remote model."""
        # Note: This might be slightly out of sync. A periodic status check could improve accuracy.
        # For simplicity, we rely on the status updated during operations.
        return self.modelLoaded

    def getDevice(self):
        """Returns 'remote_wsl' to indicate where processing occurs."""
        return 'remote_wsl'

    def cleanup(self):
        """Optionally trigger unload on the server during cleanup."""
        self._logDebug("RemoteNemoClientHandler cleanup initiated.")
        # Decide whether to unload the server model on client exit.
        # If the server is persistent, maybe don't unload. If it's tied to the client, unload.
        shouldUnloadOnExit = self.config.get('unloadRemoteModelOnExit', True)
        if shouldUnloadOnExit and self.modelLoaded:
            self.unloadModel()
        self._logDebug("RemoteNemoClientHandler cleanup complete.")
