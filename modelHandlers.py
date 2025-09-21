# modelHandlers.py
# ==============================================================================
# ASR Model Handlers (Abstract Base Class and Local Implementations)
# ==============================================================================
#
# Purpose:
# - Defines the `AbstractAsrModelHandler` interface for all model handlers.
# - Contains the `WhisperModelHandler` for local Whisper models.
# - The remote client handler has been moved to wsl_server/client_handler.py
# ==============================================================================
import abc
import gc
import time
import traceback

import numpy as np
import torch
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
    Implementations handle specific ASR libraries/models.
    """

    def __init__(self, config):
        """
        Initializes the handler with configuration and sets the initial model state.
        Args:
            config (ConfigurationManager): The application's configuration object.
        """
        self.config = config
        self.modelLoaded = False  # Status flag indicating readiness
        uniDebugLogger.debug(f"{type(self).__name__} initialized.")

    @abc.abstractmethod
    def loadModel(self) -> bool:
        """
        Loads the ASR model into memory.
        Returns:
            bool: True if the model is ready for transcription, False otherwise.
        """
        pass

    @abc.abstractmethod
    def unloadModel(self) -> bool:
        """
        Unloads the ASR model from memory.
        Returns:
            bool: True if the model was successfully unloaded, False on error.
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
        """Returns the compute device being used (e.g., 'cuda', 'cpu')."""
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
    Concrete implementation for local Whisper models using Hugging Face Transformers.
    """

    def __init__(self, config):
        """
        Initializes the Whisper handler, checks dependencies, and determines the compute device.
        Args:
            config (ConfigurationManager): The application's configuration object.
        """
        super().__init__(config)
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
        gc.collect()
        if self.device and self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                uniDebugLogger.debug("torch.cuda.empty_cache() called.")
            except Exception as e:
                uniLogger.warning(f"CUDA memory cleaning attempt failed partially: {e}")

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
        if not transformersAvailable or not torch or pipeline is None:
            uniLogger.critical("Cannot load Whisper model: Missing required libraries.")
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

        language = self.config.get('language')
        genKwargs = {"language": language} if language else {}
        genKwargs["return_timestamps"] = self.config.get('whisperReturnTimestamps', False)
        uniDebugLogger.debug(f"Pipeline generate_kwargs: {genKwargs}")

        dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        uniDebugLogger.debug(f"Using torch_dtype: {dtype}")

        startTime = time.time()
        try:
            trustRemoteCode = self.config.get('trustRemoteCode', True)
            if trustRemoteCode:
                uniLogger.warning("trust_remote_code=True is enabled for loading the model.")

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
            self._warmUpModel()
        except Exception as e:
            uniLogger.critical(f"Failed loading local Whisper model '{modelName}': {e}",
                               excInfo=True)
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
            uniLogger.info("Warming up the Whisper model...")
            warmupSampleRate = 16000
            dummyAudio = np.zeros(int(warmupSampleRate * 0.5), dtype=np.float32)
            asrInput = {"raw": dummyAudio, "sampling_rate": warmupSampleRate}
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

        if audioData.dtype != np.float32:
            audioData = audioData.astype(np.float32)

        transcription = ""
        try:
            segmentDurationSec = len(audioData) / sampleRate if sampleRate > 0 else 0
            uniLogger.info(
                f"Starting local Whisper transcription for {segmentDurationSec:.2f}s audio segment...")
            self._monitorMemory()
            startTime = time.time()
            asrInput = {"raw": audioData, "sampling_rate": sampleRate}

            with torch.no_grad():
                result = self.asrPipeline(asrInput)

            inferenceTime = time.time() - startTime
            uniLogger.info(f"Local Whisper transcription finished in {inferenceTime:.3f}s.")

            if isinstance(result, dict) and "text" in result:
                transcription = result["text"]
            elif isinstance(result, str):
                transcription = result
            else:
                uniLogger.warning(f"Unexpected Whisper result structure: {type(result)}.")
                transcription = ""

            transcription = transcription.strip() if transcription else ""
            uniDebugLogger.debug(
                f"Whisper transcription result (stripped): '{transcription[:150]}...'")
        except Exception as e:
            uniLogger.error(f"Error during local Whisper transcription: {e}", excInfo=True)
            transcription = None
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
            modelIds = [model.id for model in models if 'whisper' in model.id.lower()]
            uniLogger.info(f"Found {len(modelIds)} potential Whisper models on Hub.")
            return modelIds
        except Exception as e:
            uniLogger.error(f"Could not fetch models from Hugging Face Hub: {e}", excInfo=True)
            return []
