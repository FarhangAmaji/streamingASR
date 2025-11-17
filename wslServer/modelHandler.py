# wsl_server/model_handler.py
# ==============================================================================
# NeMo ASR Model Handler for WSL Server (UPDATED & FIXED)
# ==============================================================================
import gc
import time
import numpy as np
import torch
from utils.loggerInstance import uniLogger, uniDebugLogger


class NemoServerModelHandler:
    def __init__(self, targetModelName):
        self.targetModelName = targetModelName
        self.model = None
        self.modelLoaded = False
        self.loadInProgress = False
        self.loadError = None
        self.device = None
        self._determineDevice()
        uniLogger.info(
            f"NemoServerModelHandler initialized for model: {self.targetModelName} on device {self.device}")

    def _determineDevice(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            gpuName = torch.cuda.get_device_name(self.device)
            uniLogger.info(f"CUDA available ({gpuName}). NeMo server will use GPU.")
        else:
            self.device = torch.device('cpu')
            uniLogger.info("CUDA not available. NeMo server will use CPU.")
            uniLogger.warning("Running NeMo models on CPU can be very slow.")

    def _cudaClean(self):
        uniDebugLogger.debug("Cleaning CUDA memory (NeMo Server)...")
        gc.collect()
        if self.device and self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                uniDebugLogger.debug("torch.cuda.empty_cache() called.")
            except Exception as e:
                uniLogger.warning(f"CUDA memory cleaning failed: {e}")

    def loadModel(self) -> bool:
        if self.loadInProgress:
            uniLogger.warning("Load request ignored: load already in progress.")
            return False
        self.loadInProgress = True
        self.loadError = None
        try:
            uniLogger.info("Attempting NeMo import...")
            from nemo.collections.asr.models import ASRModel
            uniLogger.info("NeMo imported successfully.")
        except Exception as importError:
            msg = f"Cannot load NeMo model - toolkit not found or import failed: {importError}"
            uniLogger.critical(msg, excInfo=True)
            self.loadError = msg
            self.loadInProgress = False
            return False

        if self.modelLoaded:
            uniLogger.info(f"NeMo model '{self.targetModelName}' already loaded.")
            self.loadInProgress = False
            return True

        uniLogger.info(f"Loading NeMo model '{self.targetModelName}' to {self.device}...")
        self._cudaClean()
        startTime = time.time()
        try:
            # Load the model from Hugging Face or local cache
            newModel = ASRModel.from_pretrained(self.targetModelName)
            # Move the model to the determined device (GPU/CPU)
            newModel = newModel.to(self.device)
            # Set the model to evaluation mode
            newModel.eval()
            self.model = newModel
            self.modelLoaded = True
            loadTime = time.time() - startTime
            uniLogger.info(f"NeMo model loaded successfully in {loadTime:.2f}s.")
            return True
        except Exception as loadErrorDetail:
            msg = f"CRITICAL FAILURE loading NeMo model '{self.targetModelName}': {loadErrorDetail}"
            uniLogger.critical(msg, excInfo=True)
            self.modelLoaded = False
            self.model = None
            self.loadError = str(loadErrorDetail)
            self._cudaClean()
            return False
        finally:
            self.loadInProgress = False

    def unloadModel(self) -> bool:
        if not self.modelLoaded:
            uniLogger.info("NeMo model already unloaded.")
            return True
        uniLogger.info(f"Unloading NeMo model '{self.targetModelName}'...")
        try:
            del self.model
            self.model = None
            self.modelLoaded = False
            self.loadError = None
            self._cudaClean()
            uniLogger.info(f"NeMo model unloaded.")
            return True
        except Exception as e:
            uniLogger.error(f"Error during NeMo model unload: {e}", excInfo=True)
            self.modelLoaded = False
            return False

    def transcribeAudioData(self, audioDataBytes, sampleRate, targetLang) -> str | None:
        if not self.modelLoaded or self.model is None:
            uniLogger.error("Transcription skipped - Model not loaded.")
            return None
        try:
            # Convert raw bytes back to a NumPy float32 array
            audioNp = np.frombuffer(audioDataBytes, dtype=np.float32)
            if len(audioNp) == 0:
                uniLogger.warning("Received zero-length audio data.")
                return ""

            duration = len(audioNp) / sampleRate if sampleRate > 0 else 0
            uniLogger.info(
                f"Received {len(audioDataBytes)} bytes ({duration:.2f}s, Lang: {targetLang}) for transcription.")

            # Run inference without calculating gradients to save memory and computation
            with torch.no_grad():
                kwargs = {'audio': [audioNp], 'batch_size': 1}
                # For Canary models, specific parameters are required
                if 'canary' in self.targetModelName.lower():
                    kwargs['target_lang'] = targetLang
                    kwargs['task'] = 'asr'

                transcriptionResults = self.model.transcribe(**kwargs)

            # --- LOGGING FOR DEBUGGING ---
            # These logs prove that the new code is running
            uniLogger.info(f"Raw NeMo Result Type: {type(transcriptionResults)}")

            # --- SMART TEXT EXTRACTION ---
            final_text = ""
            if transcriptionResults:
                first_item = transcriptionResults[0]

                # Case 1: Output is a simple string (Older models)
                if isinstance(first_item, str):
                    final_text = first_item

                # Case 2: Output is a list/Hypothesis object (Newer models like Canary)
                # We check if it has a 'text' attribute
                elif hasattr(first_item, 'text'):
                    final_text = first_item.text

                # Case 3: Fallback - try converting to string directly
                else:
                    uniLogger.warning(
                        f"Unknown result item type: {type(first_item)}. Converting to str.")
                    final_text = str(first_item)

            final_text = final_text.strip()
            uniLogger.info(f"Extracted Final Text: '{final_text}'")
            return final_text

        except Exception as e:
            uniLogger.error(f"Error during transcription: {e}", excInfo=True)
            return None