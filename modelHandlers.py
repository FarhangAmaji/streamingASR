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
import abc  # Abstract Base Classes
import gc
import json
import \
    logging  # Retained for logging.DEBUG constant in _monitorMemory if a direct check is preferred
import time
import traceback  # For detailed error logging

import numpy as np
import requests  # For client-server communication
import torch

# Import transformers conditionally if needed by WhisperModelHandler
try:
    from transformers import pipeline, AutoConfig

    transformersAvailable = True
except ImportError:
    pipeline = None
    AutoConfig = None
    transformersAvailable = False
# Import huggingface_hub conditionally
try:
    import huggingface_hub

    hfHubAvailable = True
except ImportError:
    huggingface_hub = None
    hfHubAvailable = False

# Import logging helpers from utils
from utils import logWarning, logDebug, logInfo, logError, logCritical  # Added logCritical


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
        self.modelLoaded = False  # Status flag indicating readiness
        logDebug(f"{type(self).__name__} initialized.")

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
                        Returns an empty string "" if transcription succeeded but yielded no text (e.g., silence).
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
        logDebug(f"{type(self).__name__} cleanup initiated.")
        if self.isModelLoaded():
            try:
                self.unloadModel()
            except Exception as e:
                logError(f"Error during model unload in cleanup for {type(self).__name__}: {e}",
                         exc_info=True)
        logDebug(f"{type(self).__name__} cleanup complete.")


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
        # Check for dependencies
        if not transformersAvailable or not torch:
            # Changed to logCritical
            logCritical(
                "WhisperModelHandler requires 'transformers' and 'torch' libraries. Please install them: pip install torch transformers")
            # Set internal state to prevent operations
            self.asrPipeline = None
            self.device = None
            self.modelLoaded = False  # Ensure model is marked as not loaded
            # Optionally raise an error to halt initialization
            # raise ImportError("Missing required libraries for WhisperModelHandler")
            return  # Stop further initialization
        self.asrPipeline = None
        self.device = None
        self._determineDevice()
        # Update config with the actually determined device string (e.g., 'cuda:0' or 'cpu')
        self.config.set('device', str(self.device))
        logInfo(f"Whisper handler will use device: {self.getDevice()}")

    def _determineDevice(self):
        """Determines the compute device (CUDA GPU or CPU) for local execution."""
        if self.config.get('CPU', False):  # Check config setting to force CPU
            self.device = torch.device('cpu')
            logInfo("CPU usage forced by configuration ('CPU': True) for local Whisper model.")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')  # Defaults to cuda:0
            # Optionally, could allow specifying device index in config, e.g., 'cuda:1'
            logInfo(
                f"CUDA GPU detected ({torch.cuda.get_device_name(self.device)}). Using GPU for local Whisper model.")
        else:
            self.device = torch.device('cpu')
            logInfo("CUDA GPU not found or not selected. Using CPU for local Whisper model.")

    def _cudaClean(self):
        """Performs garbage collection and attempts to clear PyTorch's CUDA cache."""
        logDebug("Cleaning CUDA memory (Whisper Handler)...")
        # Run Python's garbage collector first
        collected = gc.collect()
        logDebug(f"Garbage collector ran, collected {collected} objects.")
        # If using CUDA, empty the cache
        if self.device and self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                logDebug("torch.cuda.empty_cache() called.")
                # Optional: synchronize if experiencing persistent memory issues
                # torch.cuda.synchronize()
                # logDebug("torch.cuda.synchronize() called.")
            except Exception as e:
                logWarning(f"CUDA memory cleaning attempt failed partially: {e}")
        logDebug("CUDA memory cleaning attempt finished (Whisper Handler).")

    def _monitorMemory(self):
        """Logs current GPU memory usage. Relies on DynamicLogger's config to show/hide DEBUG logs."""
        # The check `if logging.getLogger("SpeechToTextApp").isEnabledFor(logging.DEBUG):`
        # is removed because `logDebug.levelEquivalent` caused an AttributeError.
        # Now, we simply call logDebug and let DynamicLogger decide if it should be displayed
        # based on the active configuration.
        if self.device and self.device.type == 'cuda':
            try:
                allocated = torch.cuda.memory_allocated(self.device) / (
                        1024 ** 3)  # Convert bytes to GB
                reserved = torch.cuda.memory_reserved(self.device) / (
                        1024 ** 3)  # Convert bytes to GB
                # This logDebug call will be subject to DynamicLogger's filtering.
                logDebug(
                    f"GPU Memory (Whisper, Device {self.device}) - Allocated: {allocated:.3f} GB, Reserved: {reserved:.3f} GB")
            except Exception as e:
                logWarning(f"Failed to get GPU memory stats: {e}")

    def loadModel(self) -> bool:
        """Loads the Whisper ASR model pipeline locally."""
        if self.modelLoaded:
            logDebug(f"Whisper model '{self.config.get('modelName')}' is already loaded.")
            return True  # Already loaded
        # Ensure dependencies are available (checked in __init__, but double-check)
        if not transformersAvailable or not torch or pipeline is None or AutoConfig is None:
            logCritical(
                "Cannot load Whisper model: Missing required libraries (transformers/torch).")
            return False
        modelName = self.config.get('modelName')
        if not modelName:
            logCritical("Cannot load Whisper model: 'modelName' not specified in config.")
            return False
        if self.device is None:
            logCritical("Cannot load Whisper model: Compute device not determined.")
            return False
        logInfo(f"Loading local Whisper model '{modelName}' to device '{self.device}'...")
        self._monitorMemory()  # Memory before load
        self._cudaClean()  # Clean before loading
        # Prepare generation arguments for the pipeline
        language = self.config.get('language')  # Can be None for auto-detect
        genKwargs = {"language": language} if language else {}  # Only add if specified
        # Add other potential Whisper args from config if needed
        # genKwargs["task"] = self.config.get('whisperTask', 'transcribe')
        genKwargs["return_timestamps"] = self.config.get('whisperReturnTimestamps', False)
        logDebug(f"Pipeline generate_kwargs: {genKwargs}")
        # Determine appropriate torch_dtype
        useFp16 = self.device.type == 'cuda'  # Use float16 only on CUDA
        dtype = torch.float16 if useFp16 else torch.float32
        logDebug(f"Using torch_dtype: {dtype}")
        startTime = time.time()
        try:
            # Use trust_remote_code=True if the specific Whisper model requires it (newer or custom versions might)
            # Be aware of the security implications of trusting remote code.
            trustRemoteCode = self.config.get('trustRemoteCode',
                                              True)  # Default to True for newer models, make configurable
            if trustRemoteCode:
                logWarning("trust_remote_code=True is enabled for loading the model.")
            # Create the pipeline
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
            logInfo(
                f"Whisper model '{modelName}' loaded successfully locally on {self.device} in {loadTime:.2f}s.")
            self._warmUpModel()  # Warm up the model after successful load
        except Exception as e:
            logCritical(f"Failed loading local Whisper model '{modelName}': {e}", exc_info=True)
            logError(traceback.format_exc())  # Keep explicit traceback for this critical failure
            logError(
                "Hints: Check model name, internet connection (for first download), dependencies (transformers, torch, maybe accelerate), and available memory (RAM/VRAM).")
            # More specific error hints
            if "trust_remote_code" in str(e).lower():
                logError(
                    "Hint: Try setting 'trustRemoteCode': True in userSettings if using a model requiring custom code.")
            if "out of memory" in str(e).lower():
                logError(
                    f"Hint: Model '{modelName}' might be too large for your available {self.device.type.upper()} memory.")
            self.modelLoaded = False
            self.asrPipeline = None
            self._cudaClean()  # Clean up potential partial load artifacts
        self._monitorMemory()  # Memory after load attempt
        return self.modelLoaded

    def _warmUpModel(self):
        """Warms up the loaded model with a silent clip to reduce first inference latency."""
        if not self.modelLoaded or not self.asrPipeline:
            logDebug("Skipping model warm-up: Model not loaded.")
            return
        try:
            logInfo("Warming up the Whisper model (may take a moment)...")
            # Whisper pipeline expects a dictionary with 'raw' audio and 'sampling_rate'
            # Standard Whisper sample rate is 16kHz
            warmupSampleRate = 16000
            # Create 0.5 seconds of silence as a float32 numpy array
            dummyAudio = np.zeros(int(warmupSampleRate * 0.5), dtype=np.float32)
            # Prepare the input dictionary for the pipeline
            asrInput = {"raw": dummyAudio, "sampling_rate": warmupSampleRate}
            # Execute inference - we don't care about the result ('text')
            _ = self.asrPipeline(asrInput)  # Discard the output
            logInfo("Whisper model warm-up complete.")
        except Exception as e:
            logWarning(f"Whisper model warm-up failed: {e}", exc_info=True)

    def unloadModel(self) -> bool:
        """Unloads the local Whisper ASR model and cleans GPU cache."""
        if not self.modelLoaded:
            logDebug("Whisper model already unloaded.")
            return True  # Consider already unloaded as success
        modelName = self.config.get('modelName')
        logInfo(f"Unloading local Whisper model '{modelName}' from {self.device}...")
        unloadSuccess = False
        try:
            if self.asrPipeline is not None:
                # Explicitly delete components to help garbage collection
                if hasattr(self.asrPipeline, 'model'): del self.asrPipeline.model
                if hasattr(self.asrPipeline,
                           'feature_extractor'): del self.asrPipeline.feature_extractor
                if hasattr(self.asrPipeline, 'tokenizer'): del self.asrPipeline.tokenizer
                del self.asrPipeline
                self.asrPipeline = None
            self.modelLoaded = False
            self._cudaClean()  # Clean memory *after* deleting references
            logInfo(f"Whisper model '{modelName}' unloaded successfully.")
            unloadSuccess = True
        except Exception as e:
            logError(f"Error during Whisper model unload: {e}", exc_info=True)
            # Ensure state reflects reality even if cleanup had issues
            self.modelLoaded = False
            self.asrPipeline = None
            unloadSuccess = False  # Indicate potential issue
        finally:
            self._monitorMemory()  # Check memory after unload attempt
        return unloadSuccess

    def transcribeAudioSegment(self, audioData: np.ndarray, sampleRate: int) -> str | None:
        """Transcribes audio using the loaded local Whisper pipeline."""
        if not self.modelLoaded or self.asrPipeline is None:
            logError("Whisper transcription skipped: Model not loaded.")
            # Return None to indicate critical failure (cannot transcribe)
            return None
        if audioData is None or len(audioData) == 0:
            logDebug("Whisper transcription skipped: No audio data provided.")
            # Return empty string for empty input
            return ""
        # --- Pre-processing ---
        # Ensure float32 - RealTimeAudioProcessor should handle this, but double-check
        if audioData.dtype != np.float32:
            logWarning(
                f"Received audio data with dtype {audioData.dtype}, expected float32. Attempting conversion.")
            try:
                # Attempt robust conversion (e.g., normalize ints)
                if audioData.dtype.kind in ('i', 'u'):
                    maxVal = np.iinfo(audioData.dtype).max
                    minVal = np.iinfo(audioData.dtype).min
                    if maxVal > minVal:
                        audioData = (audioData.astype(np.float32) - minVal) / (
                                maxVal - minVal) * 2.0 - 1.0
                    else:
                        audioData = audioData.astype(np.float32)
                else:
                    audioData = audioData.astype(np.float32)
            except Exception as e:
                logError(f"Failed to convert audio data to float32: {e}",
                         exc_info=True)  # Added exc_info
                return None  # Cannot proceed without correct dtype
        # Check sample rate - Whisper models are typically trained on 16kHz.
        # The Transformers pipeline *should* handle resampling automatically.
        targetSampleRate = 16000
        if sampleRate != targetSampleRate:
            logWarning(
                f"Input audio sample rate ({sampleRate}Hz) differs from Whisper's standard ({targetSampleRate}Hz). Transformers pipeline will attempt resampling.")
            # If automatic resampling fails, manual resampling using librosa would be needed here.
        transcription = ""
        try:
            segmentDurationSec = len(audioData) / sampleRate if sampleRate > 0 else 0
            logInfo(
                f"Starting local Whisper transcription for {segmentDurationSec:.2f}s audio segment...")
            self._monitorMemory()  # Optional: Check memory before inference
            startTime = time.time()
            # Prepare input dictionary for the pipeline
            asrInput = {"raw": audioData, "sampling_rate": sampleRate}
            # --- Perform Transcription ---
            # Use torch.no_grad() for inference if not done internally by pipeline
            with torch.no_grad():
                result = self.asrPipeline(asrInput)
            inferenceTime = time.time() - startTime
            logInfo(f"Local Whisper transcription finished in {inferenceTime:.3f}s.")
            # logDebug(f"Whisper Raw Result: {result}") # Can be very verbose
            # Extract text - structure might vary slightly based on args (e.g., with timestamps)
            if isinstance(result, dict) and "text" in result:
                transcription = result["text"]  # Keep potential leading/trailing spaces for now
            elif isinstance(result, str):  # Sometimes pipeline might return just the string
                transcription = result
            else:
                logWarning(
                    f"Unexpected Whisper result structure: {type(result)}. Could not extract text.")
                transcription = ""  # Default to empty string
            # Basic cleanup - further filtering happens in TranscriptionOutputHandler
            transcription = transcription.strip() if transcription else ""
            logDebug(f"Whisper transcription result (stripped): '{transcription[:150]}...'")
        except Exception as e:
            logError(f"Error during local Whisper transcription: {e}", exc_info=True)
            logError(traceback.format_exc())  # Keep explicit traceback for this error
            transcription = None  # Indicate critical failure
            # Attempt to clean GPU memory if a CUDA error occurred
            if self.device and self.device.type == 'cuda' and 'cuda' in str(e).lower():
                logWarning("Attempting CUDA cleanup after transcription error.")
                self._cudaClean()
        finally:
            self._monitorMemory()  # Optional: Check memory after inference
        return transcription  # Return string (even empty) or None on critical error

    def getDevice(self) -> str:
        """Returns the compute device being used (e.g., 'cuda:0', 'cpu')."""
        return str(self.device) if self.device else 'unknown'

    @staticmethod
    def listAvailableModels():
        """Static method to retrieve Whisper/ASR models from Hugging Face Hub."""
        if not hfHubAvailable:
            logError("Cannot list models: huggingface_hub library not installed.")
            return []
        try:
            logInfo("Fetching list of available ASR models from Hugging Face Hub...")
            # Filter specifically for Whisper models if desired, or general ASR
            # modelFilter = huggingface_hub.ModelFilter(task="automatic-speech-recognition", library="transformers", model_name="whisper")
            modelFilter = huggingface_hub.ModelFilter(task="automatic-speech-recognition")
            models = huggingface_hub.list_models(filter=modelFilter, sort="downloads",
                                                 direction=-1, limit=100)  # Limit results
            modelIds = [model.id for model in models if
                        'whisper' in model.id.lower()]  # Extra filter by name
            logInfo(
                f"Found {len(modelIds)} potential Whisper models on Hub (among top ASR downloads).")
            # Example: ['openai/whisper-large-v3', 'openai/whisper-medium.en', ...]
            return modelIds
        except Exception as e:
            logError(f"Could not fetch models from Hugging Face Hub: {e}", exc_info=True)
            return []


# ==================================
# Remote NeMo Client Implementation
# ==================================
class RemoteNemoClientHandler(AbstractAsrModelHandler):
    """
    Concrete implementation that acts as a client to a remote ASR server (WSL).
    It sends audio data via HTTP requests and receives transcription results.
    """

    def __init__(self, config):
        super().__init__(config)
        self.serverUrl = config.get('wslServerUrl')
        if not self.serverUrl:
            logCritical(
                "RemoteNemoClientHandler cannot operate: 'wslServerUrl' not found in configuration.")
            # Mark as not functional
            self.serverReachable = False
            self.modelLoaded = False
            self.lastStatusCheckTime = 0
            # Optionally raise error here to halt application if URL is mandatory
            # raise ValueError("wslServerUrl is required for RemoteNemoClientHandler")
        else:
            self.serverReachable = None  # None = Unknown, True = Reachable, False = Unreachable
            self.modelLoaded = False  # Reflects server's model state, updated by requests
            self.lastStatusCheckTime = 0  # Throttle status checks
            logInfo(f"RemoteNeMo client handler initialized. Target server URL: {self.serverUrl}")
        # Set device identifier for this handler
        self.config.set('device', 'remote_wsl')

    def _makeServerRequest(self, method: str, endpoint: str, **kwargs) -> dict | None:
        """
        Helper function to make requests to the WSL server, handling common errors.
        Args:
            method (str): HTTP method ('get', 'post', etc.).
            endpoint (str): Server endpoint (e.g., '/status').
            **kwargs: Additional arguments for requests.request (e.g., json, files, params).
        Returns:
            dict | None: Parsed JSON response from the server, or None if the request failed.
        """
        if not self.serverUrl:
            logError("Cannot make server request: Server URL not configured.")
            return None
        if self.serverReachable is False:
            # If we know it's unreachable, don't even try (reduces error spam)
            logDebug(
                f"Skipping {method.upper()} request to {endpoint}: Server marked as unreachable.")
            return None
        # Construct full URL, ensuring no double slashes
        url = f"{self.serverUrl.rstrip('/')}/{endpoint.lstrip('/')}"
        # Get configured timeouts, provide defaults using camelCase keys
        connectTimeout = self.config.get('serverConnectTimeout', 5.0)
        readTimeout = self.config.get('serverRequestTimeout', 15.0)
        requestTimeout = (connectTimeout, readTimeout)
        logDebug(f"Sending {method.upper()} request to server: {url} (Timeout: {requestTimeout}s)")
        try:
            response = requests.request(
                method=method,
                url=url,
                timeout=requestTimeout,
                **kwargs  # Pass other arguments like files, params, json
            )
            # Raise HTTPError for bad responses (4xx client error, 5xx server error)
            response.raise_for_status()
            # Mark server as reachable on successful communication
            if self.serverReachable is not True:
                logInfo(f"Successfully connected to WSL server at {self.serverUrl}.")
                self.serverReachable = True
            # Attempt to parse JSON response
            try:
                responseData = response.json()
                logDebug(f"Received JSON response from {url}: {responseData}")
                return responseData
            except json.JSONDecodeError:
                logError(
                    f"Server response from {url} is not valid JSON. Status: {response.status_code}. Response text (start): {response.text[:100]}...")
                # Treat non-JSON response as an error for consistency
                return None
        except requests.exceptions.ConnectionError as e:
            logError(f"Connection Error connecting to WSL server at {url}: {e}")
            if self.serverReachable is not False:  # Log hint only on first failure or if status was unknown
                logError("Hint: Ensure the wslNemoServer.py script is running in WSL.")
                logError("Hint: Check Windows Firewall (allow inbound TCP port 5001).")
                logError(
                    "Hint: Check WSL Firewall if active (`sudo ufw status`, `sudo ufw allow 5001/tcp`).")
                logError(
                    f"Hint: Try connecting directly to WSL IP (e.g., via `curl http://<WSL-IP>:5001/status` from Windows CMD). If that works, try changing wslServerUrl in config (but IP may change).")
                logError(
                    "Hint: Verify WSL localhost forwarding is working if using http://localhost:5001.")
            self.serverReachable = False
            self.modelLoaded = False  # Assume model unloaded if server unreachable
            return None
        except requests.exceptions.Timeout as e:
            logError(
                f"Request Timeout connecting to or reading from WSL server at {url} (Connect >{connectTimeout}s or Read >{readTimeout}s): {e}")
            # Server might be reachable but slow/overloaded
            self.serverReachable = None  # Mark as unknown, could recover
            self.modelLoaded = False  # Assume potentially unloaded state
            return None
        except requests.exceptions.HTTPError as e:
            # Handles 4xx/5xx errors after raise_for_status()
            logError(f"HTTP Error during request to WSL server {url}: {e}")
            # Log response body if available and potentially informative
            if e.response is not None:
                logError(
                    f"Server Response ({e.response.status_code}): {e.response.text[:200]}...")  # Log start of error response text
            # Don't necessarily mark as unreachable for HTTP errors, could be bad request or server internal error
            self.serverReachable = True  # We reached it, but got an error response
            # Server might still have model loaded depending on error, but safer to assume not?
            # Let's keep modelLoaded state as is unless error clearly indicates otherwise (e.g., 503)
            if e.response is not None and e.response.status_code == 503:  # Service Unavailable
                logWarning("Server reported Service Unavailable (503), assuming model not loaded.")
                self.modelLoaded = False
            return None  # Indicate request failure
        except requests.exceptions.RequestException as e:
            # Catch other potential requests library errors
            logError(f"Unhandled RequestException during request to WSL server {url}: {e}",
                     exc_info=True)
            self.serverReachable = False  # Assume unreachable on unknown request errors
            self.modelLoaded = False
            return None

    def checkServerStatus(self, forceCheck=False) -> bool:
        """
        Checks the status of the remote server's model. Throttles checks unless forced.
        Updates self.modelLoaded and self.serverReachable based on the response.
        Args:
            forceCheck (bool): If True, bypasses throttling and checks status immediately.
        Returns:
            bool: True if the server reports the model is 'loaded', False otherwise (or if status check fails).
        """
        # Throttle status checks to avoid spamming the server
        throttleSeconds = 5.0  # Check status at most every 5 seconds unless forced
        now = time.time()
        if not forceCheck and (now - self.lastStatusCheckTime < throttleSeconds):
            logDebug("Skipping status check due to throttling.")
            return self.modelLoaded  # Return last known status
        self.lastStatusCheckTime = now
        logDebug("Checking remote server model status...")
        statusResponse = self._makeServerRequest('get', '/status')
        if statusResponse:
            serverStatus = statusResponse.get('status')
            modelName = statusResponse.get('modelName', 'N/A')
            device = statusResponse.get('device', 'N/A')
            logInfo(
                f"Server Status: Status='{serverStatus}', Model='{modelName}', Device='{device}'")
            if serverStatus == 'loaded':
                self.modelLoaded = True
                return True
            else:
                # Includes 'unloaded', 'loading', 'error' states
                self.modelLoaded = False
                return False
        else:
            # Request failed (error already logged by _makeServerRequest)
            logWarning("Failed to get server status. Assuming model is not loaded.")
            self.modelLoaded = False
            # serverReachable should have been updated by _makeServerRequest
            return False

    def loadModel(self) -> bool:
        """Attempts to tell the remote server to load the model (if not already loaded)."""
        logInfo("Requesting remote NeMo model load/check...")
        # First, check current status directly (bypassing throttle for explicit load request)
        if self.checkServerStatus(forceCheck=True):
            logInfo("Remote NeMo model is already loaded on server.")
            return True  # Already loaded
        # If not loaded or status check failed, attempt to trigger load
        logInfo("Model not loaded or status unknown, sending load request to remote server...")
        loadResponse = self._makeServerRequest('post', '/load')
        if loadResponse and loadResponse.get('status') == 'loaded':
            modelName = loadResponse.get('modelName', 'N/A')
            logInfo(f"Remote server confirmed successful model load: '{modelName}'.")
            self.modelLoaded = True
            self.serverReachable = True  # Mark as reachable after successful load command
            return True
        elif loadResponse:
            # Server responded but indicated failure
            errMsg = loadResponse.get('message', 'Unknown error from server')
            logError(f"Remote server reported failure during load request: {errMsg}")
            self.modelLoaded = False
            self.serverReachable = True  # Reachable, but failed to load
            return False
        else:
            # Communication error (already logged by _makeServerRequest)
            logError("Failed to trigger model load on remote server due to communication error.")
            self.modelLoaded = False
            # serverReachable updated by _makeServerRequest
            return False

    def unloadModel(self) -> bool:
        """Attempts to tell the remote server to unload the model."""
        logInfo("Requesting remote NeMo model unload...")
        # Check status first? Optional, unload is usually safe to send even if already unloaded.
        # if not self.modelLoaded:
        #     logInfo("Client believes model is already unloaded, sending unload request anyway.")
        unloadResponse = self._makeServerRequest('post', '/unload')
        if unloadResponse and unloadResponse.get('status') == 'unloaded':
            logInfo("Remote server confirmed successful model unload.")
            self.modelLoaded = False
            self.serverReachable = True  # Mark reachable after success
            return True
        elif unloadResponse and unloadResponse.get('status') == 'already_unloaded':
            logInfo("Remote server reported model was already unloaded.")
            self.modelLoaded = False  # Ensure client state is consistent
            self.serverReachable = True
            return True  # Consider success
        elif unloadResponse:
            # Server responded but indicated failure during unload
            errMsg = unloadResponse.get('message', 'Unknown error from server')
            logWarning(f"Remote server reported an issue during unload request: {errMsg}")
            # State might be ambiguous, but assume unloaded from client perspective for safety
            self.modelLoaded = False
            self.serverReachable = True  # Reachable, but failed unload logic
            return False  # Indicate potential issue
        else:
            # Communication error (already logged by _makeServerRequest)
            logWarning(
                "Failed to trigger model unload on remote server due to communication error. Assuming unloaded.")
            # Assume unloaded for safety, serverReachable updated by _makeServerRequest
            self.modelLoaded = False
            return False  # Indicate communication failure

    def transcribeAudioSegment(self, audioData: np.ndarray, sampleRate: int) -> str | None:
        """Sends audio data and target language to the remote server for transcription."""
        # --- Pre-checks ---
        if self.serverReachable is False:
            logError("Remote transcription skipped: Server marked as unreachable.")
            return None  # Indicate critical failure (cannot transcribe)
        if not self.modelLoaded:
            # Maybe try a quick status check if model isn't marked loaded?
            logWarning(
                "Transcription requested but client believes model is not loaded. Attempting status check...")
            if not self.checkServerStatus(
                    forceCheck=True):  # Force check before transcription attempt
                logError(
                    "Remote transcription skipped: Server status check confirmed model is not loaded.")
                return None
            # If status check now shows loaded, proceed
        if audioData is None or len(audioData) == 0:
            logDebug("Remote transcription skipped: No audio data provided.")
            return ""  # Return empty string for empty input
        # --- Get Language from Config ---
        targetLang = self.config.get('language', 'en')  # Default to 'en' if not specified
        if not targetLang:
            logWarning("Target language is empty in config, defaulting to 'en' for NeMo request.")
            targetLang = 'en'
        # --- Prepare Audio Data ---
        # Ensure audioData is float32 bytes for sending
        if audioData.dtype != np.float32:
            logWarning(
                f"Received audio data with dtype {audioData.dtype}, converting to float32 for remote server.")
            try:
                # Perform robust conversion (handle int types)
                if audioData.dtype.kind in ('i', 'u'):
                    maxValue = np.iinfo(audioData.dtype).max
                    minValue = np.iinfo(audioData.dtype).min
                    if maxValue > minValue:
                        audioData = (audioData.astype(np.float32) - minValue) / (
                                maxValue - minValue) * 2.0 - 1.0
                    else:
                        audioData = audioData.astype(np.float32)
                else:
                    audioData = audioData.astype(np.float32)
            except Exception as e:
                logError(f"Failed to convert audio data to float32 for sending: {e}",
                         exc_info=True)  # Added exc_info
                return None  # Cannot proceed
        audioBytes = audioData.tobytes()
        if not audioBytes:
            logWarning("Audio data became empty after converting to bytes.")
            return ""
        segmentDurationSec = len(audioData) / sampleRate if sampleRate > 0 else 0
        logInfo(
            f"Sending {segmentDurationSec:.2f}s audio segment ({len(audioBytes)} bytes, Lang: {targetLang}) to remote NeMo server...")
        # --- Prepare Request ---
        # Server expects multipart/form-data with 'audio_data' file
        # and query parameters 'sample_rate' and 'target_lang'
        files = {'audio_data': ('audio_segment.raw', audioBytes, 'application/octet-stream')}
        params = {'sample_rate': sampleRate, 'target_lang': targetLang}  # Add target_lang here
        startTime = time.time()
        # --- Make Request ---
        transcribeResponse = self._makeServerRequest('post', '/transcribe', params=params,
                                                     files=files)
        requestTime = time.time() - startTime
        logInfo(f"Remote transcription request finished in {requestTime:.3f}s.")
        # --- Process Response ---
        if transcribeResponse and 'transcription' in transcribeResponse:
            # Server successfully returned a transcription
            transcription = transcribeResponse['transcription']
            logDebug(f"Received transcription from server: '{transcription[:150]}...'")
            # Return the text, potentially empty if server transcribed silence
            return transcription.strip() if transcription else ""
        else:
            # Request failed or server returned an error (errors logged by _makeServerRequest)
            logError("Failed to get valid transcription from remote server.")
            # If connection failed, serverReachable/modelLoaded updated in _makeServerRequest
            # If HTTP error (like 503), modelLoaded might have been updated too
            return None  # Indicate critical failure

    def isModelLoaded(self) -> bool:
        """Returns the last known status of the remote model. May not be perfectly up-to-date."""
        # Consider adding a periodic status check if near-real-time accuracy is crucial,
        # but for now, rely on status updated during load/unload/transcribe/check calls.
        logDebug(f"Checking isModelLoaded status: {self.modelLoaded}")
        return self.modelLoaded

    def getDevice(self) -> str:
        """Returns 'remote_wsl' to indicate where processing occurs."""
        return 'remote_wsl'

    def cleanup(self):
        """Optionally trigger unload on the server during cleanup, based on config."""
        logDebug("RemoteNemoClientHandler cleanup initiated.")
        # Decide whether to ask the server to unload the model when this client exits.
        shouldUnloadOnExit = self.config.get('unloadRemoteModelOnExit', True)
        if shouldUnloadOnExit:
            if self.modelLoaded or self.serverReachable:  # Attempt unload if model might be loaded or server might be reachable
                logInfo("Requesting remote model unload during client cleanup...")
                self.unloadModel()  # Attempt to send unload request
            else:
                logDebug(
                    "Skipping remote unload request during cleanup: Client believes server is unreachable or model already unloaded.")
        else:
            logInfo(
                "Skipping remote unload request during cleanup as per configuration ('unloadRemoteModelOnExit': False).")
        logDebug("RemoteNemoClientHandler cleanup complete.")
