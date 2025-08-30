# modelHandlers.py
# ==============================================================================
# ASR Model Handlers (Abstract Base Class and Implementations)
# ==============================================================================
#
# Purpose:
# - Defines the `AbstractAsrModelHandler` interface for interacting with different
#   ASR models or services.
# - `WhisperModelHandler`: Implements the interface for local Whisper models using
#   the Hugging Face Transformers library. Handles model loading, unloading,
#   transcription, and CUDA memory management.
# - `RemoteNemoClientHandler`: Implements the interface to communicate with the
#   `wslNemoServer.py` running in WSL via HTTP requests. It sends audio data
#   and receives transcriptions, managing server interaction state.
# ==============================================================================
import abc
import gc
import json
import time
import traceback

import numpy as np
import requests
import torch
# Import the new universal logger instances
from utils.loggerInstance import uniLogger, uniDebugLogger

# Import transformers conditionally if needed by WhisperModelHandler
try:
    from transformers import pipeline

    transformersAvailable = True
except ImportError:
    pipeline = None
    transformersAvailable = False
# Import huggingface_hub conditionally
try:
    import huggingface_hub

    hfHubAvailable = True
except ImportError:
    huggingface_hub = None
    hfHubAvailable = False


# ==================================
# ASR Model Handling (Abstraction)
# ==================================
class AbstractAsrModelHandler(abc.ABC):
    """
    Abstract Base Class defining the interface for ASR model handlers.
    Implementations handle specific ASR libraries/models (local or remote client).
    """

    def __init__(self, config):
        """Initializes the handler with configuration and sets the initial model state."""
        self.config = config
        self.modelLoaded = False  # Status flag indicating readiness
        uniDebugLogger.debug(f"{type(self).__name__} initialized.")

    @abc.abstractmethod
    def loadModel(self) -> bool:
        """
        Loads the ASR model into memory (local) or ensures server connection/readiness (remote).
        Returns:
            bool: True if the model is ready for transcription, False otherwise.
        """
        pass

    @abc.abstractmethod
    def unloadModel(self) -> bool:
        """
        Unloads the ASR model (local) or potentially signals server (remote).
        Returns:
            bool: True if the model was successfully unloaded or was already unloaded, False on error.
        """
        pass

    @abc.abstractmethod
    def transcribeAudioSegment(self, audioData: np.ndarray, sampleRate: int) -> str | None:
        """
        Transcribes a given audio data segment.
        Args:
            audioData (numpy.ndarray): The audio segment (float32 expected).
            sampleRate (int): Sample rate of the audio data.
        Returns:
            str | None: The transcribed text, or None if transcription failed critically.
                        Returns an empty string "" if transcription succeeded but yielded no text.
        """
        pass

    def isModelLoaded(self) -> bool:
        """Checks if the model is considered loaded and ready for transcription."""
        return self.modelLoaded

    @abc.abstractmethod
    def getDevice(self) -> str:
        """Returns the compute device being used ('cuda', 'cpu', 'remote_wsl', 'unknown')."""
        pass

    def cleanup(self):
        """Performs cleanup actions, default is to unload the model if loaded."""
        uniDebugLogger.debug(f"{type(self).__name__} cleanup initiated.")
        if self.isModelLoaded():
            try:
                self.unloadModel()
            except Exception as e:
                uniLogger.error(
                    f"Error during model unload in cleanup for {type(self).__name__}: {e}",
                    excInfo=True)
        uniDebugLogger.debug(f"{type(self).__name__} cleanup complete.")


# ==================================
# Whisper/Transformers Implementation (Local)
# ==================================
class WhisperModelHandler(AbstractAsrModelHandler):
    """
    Concrete implementation for Whisper models using Hugging Face Transformers.
    Runs the model locally on the machine executing this script.
    """

    def __init__(self, config):
        """Initializes the Whisper handler, checks dependencies, and determines the compute device."""
        super().__init__(config)
        # Check for essential libraries
        if not transformersAvailable or not torch:
            uniLogger.critical(
                "WhisperModelHandler requires 'transformers' and 'torch' libraries. Please install them.")
            self.asrPipeline = None
            self.device = None
            self.modelLoaded = False
            return
        self.asrPipeline = None
        self.device = None
        self._determineDevice()
        self.config.set('device', str(self.device))
        uniLogger.info(f"Whisper handler will use device: {self.getDevice()}")

    def _determineDevice(self):
        """Determines the compute device (CUDA GPU or CPU) for local execution."""
        if self.config.get('CPU', False):
            self.device = torch.device('cpu')
            uniLogger.info(
                "CPU usage forced by configuration ('CPU': True) for local Whisper model.")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            uniLogger.info(
                f"CUDA GPU detected ({torch.cuda.get_device_name(self.device)}). Using GPU for local Whisper model.")
        else:
            self.device = torch.device('cpu')
            uniLogger.info("CUDA GPU not found or not selected. Using CPU for local Whisper model.")

    def _cudaClean(self):
        """Performs garbage collection and attempts to clear PyTorch's CUDA cache."""
        uniDebugLogger.debug("Cleaning CUDA memory (Whisper Handler)...")
        collected = gc.collect()
        uniDebugLogger.debug(f"Garbage collector ran, collected {collected} objects.")
        if self.device and self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                uniDebugLogger.debug("torch.cuda.empty_cache() called.")
            except Exception as e:
                uniLogger.warning(f"CUDA memory cleaning attempt failed partially: {e}")
        uniDebugLogger.debug("CUDA memory cleaning attempt finished (Whisper Handler).")

    def _monitorMemory(self):
        """Logs current GPU memory usage if available."""
        if self.device and self.device.type == 'cuda':
            try:
                allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)  # Bytes to GB
                reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)  # Bytes to GB
                uniDebugLogger.debug(
                    f"GPU Memory (Whisper, Device {self.device}) - Allocated: {allocated:.3f} GB, Reserved: {reserved:.3f} GB")
            except Exception as e:
                uniLogger.warning(f"Failed to get GPU memory stats: {e}")

    def loadModel(self) -> bool:
        """Loads the Whisper ASR model pipeline locally using Hugging Face Transformers."""
        if self.modelLoaded:
            uniDebugLogger.debug(
                f"Whisper model '{self.config.get('modelName')}' is already loaded.")
            return True
        # Double-check dependencies before loading
        if not transformersAvailable or not torch or pipeline is None:
            uniLogger.critical(
                "Cannot load Whisper model: Missing required libraries (transformers/torch).")
            return False
        modelName = self.config.get('modelName')
        if not modelName:
            uniLogger.critical("Cannot load Whisper model: 'modelName' not specified in config.")
            return False
        if self.device is None:
            uniLogger.critical("Cannot load Whisper model: Compute device not determined.")
            return False
        uniLogger.info(f"Loading local Whisper model '{modelName}' to device '{self.device}'...")
        self._monitorMemory()
        self._cudaClean()
        # Prepare generation arguments for the pipeline
        language = self.config.get('language')
        genKwargs = {"language": language} if language else {}
        genKwargs["return_timestamps"] = self.config.get('whisperReturnTimestamps', False)
        uniDebugLogger.debug(f"Pipeline generate_kwargs: {genKwargs}")
        # Use float16 for faster inference on CUDA, float32 on CPU
        useFp16 = self.device.type == 'cuda'
        dtype = torch.float16 if useFp16 else torch.float32
        uniDebugLogger.debug(f"Using torch_dtype: {dtype}")
        startTime = time.time()
        try:
            # Allow loading models that require custom code (with security awareness)
            trustRemoteCode = self.config.get('trustRemoteCode', True)
            if trustRemoteCode:
                uniLogger.warning("trust_remote_code=True is enabled for loading the model.")
            # Create the ASR pipeline
            self.asrPipeline = pipeline(
                task="automatic-speech-recognition",
                model=modelName,
                device=self.device,
                torch_dtype=dtype,
                generate_kwargs=genKwargs,
                trust_remote_code=trustRemoteCode
            )
            self.modelLoaded = True
            loadTime = time.time() - startTime
            uniLogger.info(
                f"Whisper model '{modelName}' loaded successfully on {self.device} in {loadTime:.2f}s.")
            self._warmUpModel()  # Warm up the model to reduce latency of the first transcription
        except Exception as e:
            uniLogger.critical(f"Failed loading local Whisper model '{modelName}': {e}",
                               excInfo=True)
            uniLogger.error(
                "Hints: Check model name, internet connection (for first download), dependencies (transformers, torch, maybe accelerate), and available memory (RAM/VRAM).")
            if "trust_remote_code" in str(e).lower():
                uniLogger.error(
                    "Hint: Try setting 'trustRemoteCode': True in userSettings if using a model requiring custom code.")
            if "out of memory" in str(e).lower():
                uniLogger.error(
                    f"Hint: Model '{modelName}' might be too large for your available {self.device.type.upper()} memory.")
            self.modelLoaded = False
            self.asrPipeline = None
            self._cudaClean()
        self._monitorMemory()
        return self.modelLoaded

    def _warmUpModel(self):
        """Warms up the loaded model with a silent audio clip to reduce first inference latency."""
        if not self.modelLoaded or not self.asrPipeline:
            uniDebugLogger.debug("Skipping model warm-up: Model not loaded.")
            return
        try:
            uniLogger.info("Warming up the Whisper model (may take a moment)...")
            # Create a short, silent audio array
            warmupSampleRate = 16000
            dummyAudio = np.zeros(int(warmupSampleRate * 0.5), dtype=np.float32)
            asrInput = {"raw": dummyAudio, "sampling_rate": warmupSampleRate}
            # Run a dummy transcription
            _ = self.asrPipeline(asrInput)
            uniLogger.info("Whisper model warm-up complete.")
        except Exception as e:
            uniLogger.warning(f"Whisper model warm-up failed: {e}", excInfo=True)

    def unloadModel(self) -> bool:
        """Unloads the local Whisper ASR model and cleans GPU cache."""
        if not self.modelLoaded:
            uniDebugLogger.debug("Whisper model already unloaded.")
            return True
        modelName = self.config.get('modelName')
        uniLogger.info(f"Unloading local Whisper model '{modelName}' from {self.device}...")
        try:
            # Delete the pipeline object to free memory
            if self.asrPipeline is not None:
                del self.asrPipeline
                self.asrPipeline = None
            self.modelLoaded = False
            self._cudaClean()
            uniLogger.info(f"Whisper model '{modelName}' unloaded successfully.")
            return True
        except Exception as e:
            uniLogger.error(f"Error during Whisper model unload: {e}", excInfo=True)
            self.modelLoaded = False
            self.asrPipeline = None
            return False
        finally:
            self._monitorMemory()

    def transcribeAudioSegment(self, audioData: np.ndarray, sampleRate: int) -> str | None:
        """Transcribes an audio segment using the loaded local Whisper pipeline."""
        if not self.modelLoaded or self.asrPipeline is None:
            uniLogger.error("Whisper transcription skipped: Model not loaded.")
            return None
        if audioData is None or len(audioData) == 0:
            uniDebugLogger.debug("Whisper transcription skipped: No audio data provided.")
            return ""
        # Ensure audio data is in float32 format
        if audioData.dtype != np.float32:
            uniLogger.warning(
                f"Received audio data with dtype {audioData.dtype}, expected float32. Converting.")
            try:
                if audioData.dtype.kind in ('i', 'u'):  # Handle integer types
                    maxVal = np.iinfo(audioData.dtype).max
                    minVal = np.iinfo(audioData.dtype).min
                    if maxVal > minVal:
                        audioData = (audioData.astype(np.float32) - minVal) / (
                                maxVal - minVal) * 2.0 - 1.0
                else:
                    audioData = audioData.astype(np.float32)
            except Exception as e:
                uniLogger.error(f"Failed to convert audio data to float32: {e}", excInfo=True)
                return None
        # The Transformers pipeline handles resampling automatically
        targetSampleRate = 16000
        if sampleRate != targetSampleRate:
            uniLogger.warning(
                f"Input audio sample rate ({sampleRate}Hz) differs from Whisper's standard ({targetSampleRate}Hz). Transformers pipeline will resample.")
        transcription = ""
        try:
            segmentDurationSec = len(audioData) / sampleRate if sampleRate > 0 else 0
            uniLogger.info(
                f"Starting local Whisper transcription for {segmentDurationSec:.2f}s audio segment...")
            self._monitorMemory()
            startTime = time.time()
            asrInput = {"raw": audioData, "sampling_rate": sampleRate}
            # Run inference without calculating gradients
            with torch.no_grad():
                result = self.asrPipeline(asrInput)
            inferenceTime = time.time() - startTime
            uniLogger.info(f"Local Whisper transcription finished in {inferenceTime:.3f}s.")
            # Extract text from the result dictionary
            if isinstance(result, dict) and "text" in result:
                transcription = result["text"]
            elif isinstance(result, str):
                transcription = result
            else:
                uniLogger.warning(
                    f"Unexpected Whisper result structure: {type(result)}. Could not extract text.")
                transcription = ""
            transcription = transcription.strip() if transcription else ""
            uniDebugLogger.debug(
                f"Whisper transcription result (stripped): '{transcription[:150]}...'")
        except Exception as e:
            uniLogger.error(f"Error during local Whisper transcription: {e}", excInfo=True)
            transcription = None  # Indicate critical failure
            if self.device and self.device.type == 'cuda' and 'cuda' in str(e).lower():
                uniLogger.warning("Attempting CUDA cleanup after transcription error.")
                self._cudaClean()
        finally:
            self._monitorMemory()
        return transcription

    def getDevice(self) -> str:
        """Returns the compute device being used (e.g., 'cuda:0', 'cpu')."""
        return str(self.device) if self.device else 'unknown'

    @staticmethod
    def listAvailableModels():
        """Static method to retrieve a list of Whisper/ASR models from the Hugging Face Hub."""
        if not hfHubAvailable:
            uniLogger.error("Cannot list models: huggingface_hub library not installed.")
            return []
        try:
            uniLogger.info("Fetching list of available ASR models from Hugging Face Hub...")
            modelFilter = huggingface_hub.ModelFilter(task="automatic-speech-recognition")
            models = huggingface_hub.list_models(filter=modelFilter, sort="downloads", direction=-1,
                                                 limit=100)
            # Filter the list to show only models with 'whisper' in their name
            modelIds = [model.id for model in models if 'whisper' in model.id.lower()]
            uniLogger.info(
                f"Found {len(modelIds)} potential Whisper models on Hub (among top ASR downloads).")
            return modelIds
        except Exception as e:
            uniLogger.error(f"Could not fetch models from Hugging Face Hub: {e}", excInfo=True)
            return []


# ==================================
# Remote NeMo Client Implementation
# ==================================
class RemoteNemoClientHandler(AbstractAsrModelHandler):
    """
    Acts as a client to a remote ASR server (wslNemoServer.py).
    It sends audio data via HTTP requests and receives transcription results.
    """

    def __init__(self, config):
        """Initializes the client with the server URL and sets initial state."""
        super().__init__(config)
        self.serverUrl = config.get('wslServerUrl')
        if not self.serverUrl:
            uniLogger.critical(
                "RemoteNemoClientHandler cannot operate: 'wslServerUrl' not found in configuration.")
            self.serverReachable = False
            self.modelLoaded = False
            self.lastStatusCheckTime = 0
        else:
            self.serverReachable = None  # None = Unknown, True = Reachable, False = Unreachable
            self.modelLoaded = False
            self.lastStatusCheckTime = 0
            uniLogger.info(
                f"RemoteNeMo client handler initialized. Target server URL: {self.serverUrl}")
        self.config.set('device', 'remote_wsl')

    def _makeServerRequest(self, method: str, endpoint: str, **kwargs) -> dict | None:
        """Helper function to make requests to the WSL server, handling common errors."""
        if not self.serverUrl:
            uniLogger.error("Cannot make server request: Server URL not configured.")
            return None
        if self.serverReachable is False:
            uniDebugLogger.debug(
                f"Skipping {method.upper()} request to {endpoint}: Server marked as unreachable.")
            return None
        # Construct the full URL for the request
        url = f"{self.serverUrl.rstrip('/')}/{endpoint.lstrip('/')}"
        connectTimeout = self.config.get('serverConnectTimeout', 5.0)
        readTimeout = self.config.get('serverRequestTimeout', 15.0)
        requestTimeout = (connectTimeout, readTimeout)
        uniDebugLogger.debug(
            f"Sending {method.upper()} request to server: {url} (Timeout: {requestTimeout}s)")
        try:
            response = requests.request(method=method, url=url, timeout=requestTimeout, **kwargs)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            if self.serverReachable is not True:
                uniLogger.info(f"Successfully connected to WSL server at {self.serverUrl}.")
                self.serverReachable = True
            try:
                responseData = response.json()
                uniDebugLogger.debug(f"Received JSON response from {url}: {responseData}")
                return responseData
            except json.JSONDecodeError:
                uniLogger.error(
                    f"Server response from {url} is not valid JSON. Status: {response.status_code}. Response text: {response.text[:100]}...")
                return None
        except requests.exceptions.ConnectionError as e:
            uniLogger.error(f"Connection Error connecting to WSL server at {url}: {e}")
            if self.serverReachable is not False:
                uniLogger.error(
                    "Hint: Ensure the wslNemoServer.py script is running in WSL and check firewalls.")
            self.serverReachable = False
            self.modelLoaded = False
            return None
        except requests.exceptions.Timeout as e:
            uniLogger.error(
                f"Request Timeout to WSL server at {url} (Connect >{connectTimeout}s or Read >{readTimeout}s): {e}")
            self.serverReachable = None  # Status is now unknown
            self.modelLoaded = False
            return None
        except requests.exceptions.HTTPError as e:
            uniLogger.error(f"HTTP Error during request to WSL server {url}: {e}")
            if e.response is not None:
                uniLogger.error(
                    f"Server Response ({e.response.status_code}): {e.response.text[:200]}...")
                if e.response.status_code == 503:  # Service Unavailable
                    uniLogger.warning(
                        "Server reported Service Unavailable (503), assuming model not loaded.")
                    self.modelLoaded = False
            return None
        except requests.exceptions.RequestException as e:
            uniLogger.error(f"Unhandled RequestException during request to WSL server {url}: {e}",
                            excInfo=True)
            self.serverReachable = False
            self.modelLoaded = False
            return None

    def checkServerStatus(self, forceCheck=False) -> bool:
        """Checks the status of the remote server's model. Throttles checks to avoid spamming."""
        throttleSeconds = 5.0
        now = time.time()
        if not forceCheck and (now - self.lastStatusCheckTime < throttleSeconds):
            uniDebugLogger.debug("Skipping status check due to throttling.")
            return self.modelLoaded
        self.lastStatusCheckTime = now
        uniDebugLogger.debug("Checking remote server model status...")
        statusResponse = self._makeServerRequest('get', '/status')
        if statusResponse:
            serverStatus = statusResponse.get('status')
            uniLogger.info(
                f"Server Status: Status='{serverStatus}', Model='{statusResponse.get('modelName', 'N/A')}', Device='{statusResponse.get('device', 'N/A')}'")
            if serverStatus == 'loaded':
                self.modelLoaded = True
                return True
            else:  # 'unloaded', 'loading', 'error'
                self.modelLoaded = False
                return False
        else:  # Request failed
            uniLogger.warning("Failed to get server status. Assuming model is not loaded.")
            self.modelLoaded = False
            return False

    def loadModel(self) -> bool:
        """Attempts to tell the remote server to load the model if it's not already loaded."""
        uniLogger.info("Requesting remote NeMo model load/check...")
        # First, check the current status
        if self.checkServerStatus(forceCheck=True):
            uniLogger.info("Remote NeMo model is already loaded on server.")
            return True
        # If not loaded, send a request to the /load endpoint
        uniLogger.info("Model not loaded, sending load request to remote server...")
        loadResponse = self._makeServerRequest('post', '/load')
        if loadResponse and loadResponse.get('status') == 'loaded':
            uniLogger.info(
                f"Remote server confirmed successful model load: '{loadResponse.get('modelName', 'N/A')}'.")
            self.modelLoaded = True
            self.serverReachable = True
            return True
        elif loadResponse:  # Server responded but indicated failure
            uniLogger.error(
                f"Remote server reported failure during load request: {loadResponse.get('message', 'Unknown error')}")
            self.modelLoaded = False
            self.serverReachable = True
            return False
        else:  # Communication error
            uniLogger.error(
                "Failed to trigger model load on remote server due to communication error.")
            self.modelLoaded = False
            return False

    def unloadModel(self) -> bool:
        """Attempts to tell the remote server to unload the model."""
        uniLogger.info("Requesting remote NeMo model unload...")
        unloadResponse = self._makeServerRequest('post', '/unload')
        if unloadResponse and unloadResponse.get('status') in ['unloaded', 'already_unloaded']:
            uniLogger.info(
                f"Remote server confirmed model is unloaded (status: {unloadResponse.get('status')}).")
            self.modelLoaded = False
            self.serverReachable = True
            return True
        elif unloadResponse:
            uniLogger.warning(
                f"Remote server reported an issue during unload: {unloadResponse.get('message', 'Unknown error')}")
            self.modelLoaded = False
            self.serverReachable = True
            return False
        else:
            uniLogger.warning(
                "Failed to trigger model unload on remote server due to communication error. Assuming unloaded.")
            self.modelLoaded = False
            return False

    def transcribeAudioSegment(self, audioData: np.ndarray, sampleRate: int) -> str | None:
        """Sends audio data and target language to the remote server for transcription."""
        if self.serverReachable is False:
            uniLogger.error("Remote transcription skipped: Server marked as unreachable.")
            return None
        # Check model status before sending data
        if not self.modelLoaded:
            uniLogger.warning(
                "Transcription requested but client believes model is not loaded. Checking status...")
            if not self.checkServerStatus(forceCheck=True):
                uniLogger.error(
                    "Remote transcription skipped: Server status check confirmed model is not loaded.")
                return None
        if audioData is None or len(audioData) == 0:
            uniDebugLogger.debug("Remote transcription skipped: No audio data provided.")
            return ""
        targetLang = self.config.get('language', 'en')
        if not targetLang:
            uniLogger.warning(
                "Target language is empty in config, defaulting to 'en' for NeMo request.")
            targetLang = 'en'
        # Ensure audio is float32 before sending
        if audioData.dtype != np.float32:
            uniLogger.warning(
                f"Audio data is {audioData.dtype}, converting to float32 for remote server.")
            try:
                if audioData.dtype.kind in ('i', 'u'):
                    maxValue = np.iinfo(audioData.dtype).max
                    if maxValue > 0: audioData = audioData.astype(np.float32) / maxValue
                else:
                    audioData = audioData.astype(np.float32)
            except Exception as e:
                uniLogger.error(f"Failed to convert audio data to float32 for sending: {e}",
                                excInfo=True)
                return None
        # Convert numpy array to bytes
        audioBytes = audioData.tobytes()
        segmentDurationSec = len(audioData) / sampleRate if sampleRate > 0 else 0
        uniLogger.info(
            f"Sending {segmentDurationSec:.2f}s audio segment ({len(audioBytes)} bytes, Lang: {targetLang}) to remote NeMo server...")
        # Prepare files and parameters for the POST request
        files = {'audio_data': ('audio_segment.raw', audioBytes, 'application/octet-stream')}
        params = {'sample_rate': sampleRate, 'target_lang': targetLang}
        startTime = time.time()
        transcribeResponse = self._makeServerRequest('post', '/transcribe', params=params,
                                                     files=files)
        requestTime = time.time() - startTime
        uniLogger.info(f"Remote transcription request finished in {requestTime:.3f}s.")
        # Process the server's response
        if transcribeResponse and 'transcription' in transcribeResponse:
            transcription = transcribeResponse['transcription']
            uniDebugLogger.debug(f"Received transcription from server: '{transcription[:150]}...'")
            return transcription.strip() if transcription else ""
        else:
            uniLogger.error("Failed to get valid transcription from remote server.")
            return None

    def getDevice(self) -> str:
        """Returns 'remote_wsl' to indicate that processing occurs on the remote server."""
        return 'remote_wsl'

    def cleanup(self):
        """Optionally triggers model unload on the remote server during cleanup."""
        uniDebugLogger.debug("RemoteNemoClientHandler cleanup initiated.")
        shouldUnloadOnExit = self.config.get('unloadRemoteModelOnExit', True)
        if shouldUnloadOnExit:
            if self.modelLoaded or self.serverReachable:
                uniLogger.info("Requesting remote model unload during client cleanup...")
                self.unloadModel()
        else:
            uniLogger.info("Skipping remote unload request during cleanup as per configuration.")
        uniDebugLogger.debug("RemoteNemoClientHandler cleanup complete.")
