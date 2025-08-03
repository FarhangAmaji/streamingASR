# wslNemoServer.py
# ==============================================================================
# NeMo ASR Model Server (for WSL Environment) - Starts Flask FIRST
# ==============================================================================
#
# Purpose:
# - Runs a simple web server (Flask) inside a WSL environment (e.g., Ubuntu).
# - **Starts Flask immediately** to be responsive to client connections.
# - If --load_on_start is passed, attempts to load the NeMo model in a
#   background thread after Flask starts.
# - Logs activities to wslNemoServer.log.
# - Listens for HTTP requests from the main application (running on Windows).
# - Provides endpoints: /transcribe, /load, /unload, /status.
#
# Usage:
# - Run this script from within your WSL distribution.
# - Example: python wslNemoServer.py --model_name "nvidia/parakeet-rnnt-1.1b" --port 5001 --load_on_start
# ==============================================================================
import argparse
import gc
import logging
import sys
import threading  # Needed for background loading
import time
import numpy as np
import torch

# --- Flask Setup ---
# Import Flask early, before potentially problematic libraries if possible
try:
    from flask import Flask, request, jsonify

    flaskAvailable = True
except ImportError:
    # Use print for critical early errors as logging might not be fully set up
    print("CRITICAL ERROR: Flask library not found. pip install Flask", file=sys.stderr)
    sys.exit(1)  # Cannot run without Flask
# --- Configure Logging ---
# Configure logging after Flask import but before NeMo
logFileName = 'wslNemoServer.log'
logLevel = logging.DEBUG  # Use DEBUG for detailed logs during troubleshooting
logFormat = '%(asctime)s - %(levelname)-8s - [%(threadName)s] - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
dateFormat = '%Y-%m-%d %H:%M:%S,%f'[:-3]  # Milliseconds format
# Use filemode 'a' (append) which is generally better for server logs
# Use filemode 'w' (overwrite) only if specifically needed for clean debug runs
logging.basicConfig(level=logging.DEBUG,  # Set root logger level
                    format=logFormat,
                    datefmt=dateFormat,
                    filename=logFileName,
                    filemode='w')  # Write mode
# Console Handler (optional, but helpful for seeing logs directly in WSL console)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logging.Formatter(logFormat, datefmt=dateFormat))
consoleHandler.setLevel(logLevel)  # Match console level for debugging
logging.getLogger().addHandler(consoleHandler)  # Add handler to the root logger
# Silence noisy libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(
    logging.WARNING)  # Also silence requests if it's used implicitly
logging.getLogger("huggingface_hub").setLevel(logging.INFO)  # Hub downloads can be verbose
logging.getLogger("nemo_toolkit").setLevel(logging.INFO)  # NeMo INFO can be useful
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.INFO)  # Flask's internal server logs requests
logger = logging.getLogger(__name__)  # Get a logger for this module
logger.info(f"--- WSL NeMo Server Script Started ---")
logger.info(f"Logging configured. Level: DEBUG. Log file: {logFileName}")
app = Flask(__name__)


# --- Global Server State ---
class NemoServerModelHandler:
    """
    Manages the NeMo ASR model lifecycle and transcription within the server.
    Delays NeMo import until loadModel is called.
    """

    def __init__(self, targetModelName):
        # Store the name of the model we *intend* to load
        self.targetModelName = targetModelName
        self.model = None
        self.modelLoaded = False
        self.loadInProgress = False  # Flag to prevent concurrent loads
        self.loadError = None  # Store last load error message
        self.device = None
        # NOTE: NeMo is NOT imported here. Import happens in loadModel.
        self._determineDevice()
        logger.info(
            f"NemoServerModelHandler initialized for target model: {self.targetModelName} on device {self.device}")
        logger.info(f"Model '{self.targetModelName}' is initially NOT loaded.")

    def _determineDevice(self):
        """Determines the compute device (CUDA GPU or CPU)."""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            try:
                gpuName = torch.cuda.get_device_name(self.device)
            except Exception as gpuError:
                gpuName = f"Error getting name: {gpuError}"
            logger.info(f"CUDA available ({gpuName}). NeMo server will use GPU ({self.device}).")
        else:
            self.device = torch.device('cpu')
            logger.info("CUDA not available. NeMo server will use CPU.")
            logger.warning("Running NeMo models on CPU can be very slow and memory-intensive.")

    def _cudaClean(self):
        """Performs garbage collection and attempts to clear PyTorch's CUDA cache."""
        logger.debug("Attempting to clean CUDA memory (NeMo Server)...")
        collected = gc.collect()
        logger.debug(f"Garbage collector collected {collected} objects.")
        if self.device and self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                logger.debug("torch.cuda.empty_cache() called.")
            except Exception as e:
                logger.warning(f"CUDA memory cleaning attempt failed partially: {e}")
        logger.debug("CUDA memory cleaning attempt finished (NeMo Server).")

    def loadModel(self):
        """
        Imports NeMo and loads the ASR model.
        Returns True on success, False on failure.
        """
        # Prevent concurrent load attempts
        if self.loadInProgress:
            logger.warning("Load request ignored: Another load operation is already in progress.")
            return False  # Indicate busy/failure
        # Set loading flag immediately
        self.loadInProgress = True
        self.loadError = None  # Clear previous error
        # --- Import NeMo Here ---
        ASRModel = None
        try:
            logger.info("Attempting NeMo import inside loadModel...")
            from nemo.collections.asr.models import ASRModel
            logger.info("NeMo imported successfully.")
        except ImportError:
            errorMessage = "Cannot load NeMo model - NeMo toolkit not found or import failed during load request."
            logger.critical(errorMessage)
            logger.critical(
                "Please ensure 'nemo_toolkit[asr]' is installed in the WSL environment.")
            self.loadError = errorMessage
            self.loadInProgress = False
            return False
        except Exception as importError:
            errorMessage = f"Unexpected error importing NeMo during load request: {importError}"
            logger.critical(errorMessage, exc_info=True)
            self.loadError = errorMessage
            self.loadInProgress = False
            return False
        # --- Proceed with load if import succeeded ---
        if self.modelLoaded:
            logger.info(
                f"NeMo model '{self.targetModelName}' already loaded (checked after import).")
            self.loadInProgress = False  # Ensure flag is reset
            return True
        logger.info(f"Attempting to load NeMo model '{self.targetModelName}' to {self.device}...")
        self._cudaClean()
        startTime = time.time()
        success = False
        try:
            logger.debug(f"Calling ASRModel.from_pretrained('{self.targetModelName}')...")
            # Use the imported ASRModel class
            newModel = ASRModel.from_pretrained(self.targetModelName)
            logger.debug(f"Model loaded from pretrained, moving to device: {self.device}")
            newModel = newModel.to(self.device)
            logger.debug("Setting model to evaluation mode...")
            newModel.eval()
            # Success - update state
            self.model = newModel
            self.modelLoaded = True  # Mark as loaded *before* resetting loadInProgress
            success = True
            loadTime = time.time() - startTime
            logger.info(
                f"NeMo model '{self.targetModelName}' loaded successfully to {self.device} in {loadTime:.2f}s.")
        except Exception as loadErrorDetail:
            errorMessage = f"CRITICAL FAILURE loading NeMo model '{self.targetModelName}': {loadErrorDetail}"
            logger.error(errorMessage, exc_info=True)
            logger.error(
                "Check model name, internet connection (for download), NeMo installation, dependencies, and available memory (RAM/VRAM).")
            self.modelLoaded = False
            self.model = None
            self.loadError = str(loadErrorDetail)  # Store the error message
            self._cudaClean()
            success = False
        finally:
            # Mark loading as finished AFTER updating modelLoaded/loadError
            self.loadInProgress = False
            logger.info(f"Load attempt finished. Success: {success}")
        return success

    def unloadModel(self):
        """Unloads the NeMo ASR model."""
        if not self.modelLoaded:
            logger.info("NeMo model already unloaded.")
            return True
        # Use targetModelName for logging consistency
        logger.info(f"Unloading NeMo model '{self.targetModelName}' from {self.device}...")
        unloadSuccess = False
        try:
            if self.model is not None:
                try:
                    logger.debug("Moving model to CPU before unloading...")
                    self.model.to('cpu')
                except Exception as cpuMoveError:
                    logger.warning(
                        f"Ignoring error while moving model to CPU during unload: {cpuMoveError}")
                logger.debug("Deleting model object reference...")
                del self.model
                self.model = None
            self.modelLoaded = False
            self.loadError = None  # Clear error on successful unload
            self._cudaClean()
            logger.info(f"NeMo model '{self.targetModelName}' unloaded.")
            unloadSuccess = True
        except Exception as e:
            logger.error(f"Error during NeMo model unload: {e}", exc_info=True)
            # Ensure state reflects failure
            self.modelLoaded = False
            unloadSuccess = False
        return unloadSuccess

    def transcribeAudioData(self, audioDataBytes, sampleRate, targetLang):
        """
        Transcribes audio data received as bytes. Converts bytes to numpy array first.
        Uses the provided targetLang for multilingual models. Handles Hypothesis objects.
        """
        if not self.modelLoaded or self.model is None:
            logger.error("NeMo transcription skipped - Model not loaded.")
            return None  # Indicate critical failure
        transcriptionText = None
        try:
            try:
                audioNp = np.frombuffer(audioDataBytes, dtype=np.float32)
                if len(audioNp) == 0:
                    logger.warning("Received zero-length audio data after numpy conversion.")
                    return ""
            except ValueError as e:
                logger.error(f"Could not convert received bytes to float32 numpy array: {e}")
                return None
            audioDurationSeconds = len(audioNp) / sampleRate if sampleRate > 0 else 0
            logger.info(
                f"Received {len(audioDataBytes)} bytes ({len(audioNp)} samples, {audioDurationSeconds:.2f}s, Lang: {targetLang}) for transcription.")
            logger.debug(
                f"Starting NeMo transcription (Lang: {targetLang}) on device {self.device}...")
            startTime = time.time()
            # --- Ensure NeMo is imported if transcribe is called before load ---
            # This is a safety check, normally loadModel should have been called first.
            if 'ASRModel' not in locals() and 'ASRModel' not in globals():
                logger.warning(
                    "NeMo ASRModel not imported yet. Attempting import before transcription.")
                try:
                    from nemo.collections.asr.models import ASRModel
                    logger.info("NeMo imported successfully during transcription request.")
                except Exception as importError:
                    logger.error(
                        f"Failed to import NeMo during transcription request: {importError}")
                    return None  # Cannot transcribe
            with torch.no_grad():
                transcriptionKwargs = {
                    'audio': [audioNp],
                    'batch_size': 1
                }
                # Pass target_lang if model name suggests it's multilingual (e.g., Canary)
                if 'canary' in self.targetModelName.lower():
                    transcriptionKwargs['target_lang'] = targetLang
                    transcriptionKwargs['task'] = 'asr'
                    logger.debug(
                        f"Added target_lang='{targetLang}' and task='asr' for Canary model.")
                elif targetLang:
                    # Check if model supports inference kwargs before passing potentially unknown ones
                    # For simplicity, we log a warning if language is provided but model isn't known multilingual
                    logger.warning(
                        f"Target language '{targetLang}' provided, but unsure if model '{self.targetModelName}' uses it. Not passing to transcribe().")
                # Ensure self.model exists before calling transcribe
                if self.model is None:
                    logger.error("Transcription cannot proceed: self.model is None.")
                    return None
                transcriptionResults = self.model.transcribe(**transcriptionKwargs)
            transcribeTime = time.time() - startTime
            logger.info(f"NeMo transcription finished in {transcribeTime:.2f}s.")
            # Extract text from Hypothesis or other potential formats
            if transcriptionResults and isinstance(transcriptionResults, list):
                firstResult = transcriptionResults[0]
                # Handle NeMo's Hypothesis object or simple string
                if hasattr(firstResult, 'text') and isinstance(firstResult.text, str):
                    transcriptionText = firstResult.text
                elif isinstance(firstResult, str):
                    transcriptionText = firstResult
                # Handle nested list case seen sometimes [[result]]
                elif isinstance(firstResult, list) and firstResult and isinstance(firstResult[0],
                                                                                  str):
                    transcriptionText = firstResult[0]
                else:
                    logger.warning(
                        f"Unexpected result format inside list: {type(firstResult)}. Full: {transcriptionResults}")
                    transcriptionText = ""
            elif isinstance(transcriptionResults,
                            str):  # Handle case where transcribe returns just a string
                transcriptionText = transcriptionResults
            else:
                logger.warning(
                    f"Empty or unexpected result format: {type(transcriptionResults)}. Full: {transcriptionResults}")
                transcriptionText = ""
            transcriptionText = transcriptionText.strip() if transcriptionText else ""
        except Exception as e:
            logger.error(f"Error during NeMo transcription/processing: {e}", exc_info=True)
            transcriptionText = None
        return transcriptionText


# Global handler instance (initialized later based on args)
nemoHandler = None


# --- Flask Routes ---
@app.route('/status', methods=['GET'])
def getStatus():
    """Endpoint to check if the NeMo model is loaded or loading."""
    global nemoHandler
    if not nemoHandler:
        logger.error("Status request failed: Server handler not initialized.")
        return jsonify({"status": "error", "message": "Server handler not initialized"}), 500
    currentStatus = "unloaded"
    message = "Model is not loaded."
    if nemoHandler.loadInProgress:
        currentStatus = "loading"
        message = "Model load is in progress."
    elif nemoHandler.modelLoaded:
        currentStatus = "loaded"
        message = "Model is loaded and ready."
    elif nemoHandler.loadError:
        currentStatus = "error"
        message = f"Model loading failed: {nemoHandler.loadError}"
    # Log at DEBUG level to reduce noise during normal operation, but keep for troubleshooting
    logger.debug(
        f"Status request received. Reporting status: '{currentStatus}' for model '{nemoHandler.targetModelName}'")
    return jsonify({
        "status": currentStatus,
        "message": message,
        "modelName": nemoHandler.targetModelName,  # Always report the target model
        "device": str(nemoHandler.device)
    }), 200


@app.route('/load', methods=['POST'])
def loadModelEndpoint():
    """Endpoint to explicitly trigger loading the NeMo model."""
    global nemoHandler
    if not nemoHandler:
        logger.error("Load request failed: Server handler not initialized.")
        return jsonify({"status": "error", "message": "Server handler not initialized"}), 500
    if nemoHandler.loadInProgress:
        logger.warning("Load request received, but load already in progress.")
        return jsonify({"status": "loading", "message": "Model load already in progress",
                        "modelName": nemoHandler.targetModelName}), 409  # Conflict
    if nemoHandler.modelLoaded:
        logger.warning(
            f"Load request received but model '{nemoHandler.targetModelName}' is already loaded.")
        return jsonify({"status": "loaded", "message": "Model already loaded",
                        "modelName": nemoHandler.targetModelName}), 200
    logger.info(
        f"Received /load request. Starting load in background thread for '{nemoHandler.targetModelName}'...")
    # Run load in background to prevent blocking Flask request for too long
    # The loadModel method itself handles the loadInProgress flag.
    thread = threading.Thread(target=nemoHandler.loadModel, name="ModelLoaderThread", daemon=True)
    thread.start()
    logger.info("Model loading initiated in background thread.")
    # Return immediately, indicating loading has started
    # Client should poll /status to know when it finishes/fails.
    return jsonify({"status": "loading", "message": "Model loading initiated in background",
                    "modelName": nemoHandler.targetModelName}), 202  # Accepted


@app.route('/unload', methods=['POST'])
def unloadModelEndpoint():
    """Endpoint to explicitly trigger unloading the NeMo model."""
    global nemoHandler
    if not nemoHandler:
        logger.error("Unload request failed: Server handler not initialized.")
        return jsonify({"status": "error", "message": "Server handler not initialized"}), 500
    if nemoHandler.loadInProgress:
        logger.warning("Unload request received, but model loading is currently in progress.")
        return jsonify({"status": "loading", "message": "Cannot unload while model is loading",
                        "modelName": nemoHandler.targetModelName}), 409  # Conflict
    if not nemoHandler.modelLoaded:
        logger.warning("Unload request received but model is already unloaded.")
        # Return success even if already unloaded, it achieves the desired state
        return jsonify({"status": "unloaded", "message": "Model already unloaded",
                        "modelName": nemoHandler.targetModelName}), 200
    logger.info(f"Received request to unload model '{nemoHandler.targetModelName}'...")
    success = nemoHandler.unloadModel()
    if success:
        return jsonify({"status": "unloaded", "message": "Model unloaded successfully",
                        "modelName": nemoHandler.targetModelName}), 200
    else:
        logger.error("An issue occurred during model unload (check server logs).")
        return jsonify({"status": "error", "message": "An issue occurred during unload",
                        "modelName": nemoHandler.targetModelName}), 500


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Endpoint to receive audio data and target language, returns transcription."""
    global nemoHandler
    endpointStartTime = time.time()
    if not nemoHandler:
        logger.error("Transcribe request failed: Server handler not initialized.")
        return jsonify({"status": "error", "message": "Server handler not initialized"}), 500
    # Check if loading is in progress
    if nemoHandler.loadInProgress:
        logger.warning("Transcription request received, but model loading is in progress.")
        return jsonify({"status": "loading",
                        "message": "Model is currently loading, please wait."}), 503  # Service Unavailable
    if not nemoHandler.modelLoaded:
        errorMessage = "Model is not loaded."
        if nemoHandler.loadError:
            errorMessage = f"Model failed to load ({nemoHandler.loadError}). Use /load to retry."
        logger.error(f"Transcription request received, but {errorMessage}")
        return jsonify({"status": "error", "message": errorMessage}), 503  # Service Unavailable
    if 'audio_data' not in request.files:
        logger.error("Transcribe request failed: Missing 'audio_data' in request files.")
        return jsonify({"status": "error", "message": "Missing 'audio_data' in request files"}), 400
    audioFile = request.files['audio_data']
    audioBytes = audioFile.read()
    sampleRateString = request.args.get('sample_rate')
    if not sampleRateString:
        logger.error("Transcribe request failed: Missing 'sample_rate' query parameter.")
        return jsonify({"status": "error", "message": "Missing 'sample_rate' query parameter"}), 400
    try:
        sampleRate = int(sampleRateString)
    except (TypeError, ValueError):
        logger.error(
            f"Invalid 'sample_rate' query parameter '{sampleRateString}'. Must be integer.")
        return jsonify(
            {"status": "error", "message": "Invalid 'sample_rate', must be an integer"}), 400
    targetLang = request.args.get('target_lang')
    if not targetLang:
        logger.error("Transcribe request failed: Missing 'target_lang' query parameter.")
        return jsonify({"status": "error", "message": "Missing 'target_lang' query parameter"}), 400
    if not audioBytes:
        logger.warning("Received empty audio data in transcribe request.")
        return jsonify({"transcription": ""}), 200
    logger.info(
        f"Processing transcription request: {len(audioBytes)} bytes, Rate: {sampleRate}Hz, Lang: {targetLang}")
    resultText = nemoHandler.transcribeAudioData(audioBytes, sampleRate, targetLang)
    if resultText is not None:
        logger.info(f"Sending transcription result: '{resultText[:100]}...'")
        endpointDuration = time.time() - endpointStartTime
        logger.info(f"/transcribe endpoint processed in {endpointDuration:.3f} seconds")
        return jsonify({"transcription": resultText}), 200
    else:
        logger.error("Transcription failed on server during processing.")
        return jsonify({"status": "error",
                        "message": "Transcription failed on server (check server logs)"}), 500


# --- Background Load Function ---
def initialModelLoadTask(handler):
    """Target function for the background thread to perform initial model load."""
    logger.info("[InitialModelLoader] Background thread started...")
    if handler:
        logger.info("[InitialModelLoader] Calling handler.loadModel()...")
        loadOk = handler.loadModel()  # Attempt load, result logged internally
        if loadOk:
            logger.info("[InitialModelLoader] Background initial model load successful.")
        else:
            logger.error(
                "[InitialModelLoader] Background initial model load failed. Check logs above for details.")
    else:
        logger.error("[InitialModelLoader] Background load task cannot run: Nemo handler is None.")
    logger.info("[InitialModelLoader] Background initial model load thread finished.")


# --- Main Execution ---
if __name__ == "__main__":
    # Removed the check for top-level nemoAvailable flag here,
    # as import is now deferred to loadModel.
    # The script will start Flask regardless, and loadModel will handle import errors later.
    parser = argparse.ArgumentParser(description="NeMo ASR Model Server for WSL")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the NeMo ASR model to load (e.g., 'nvidia/parakeet-rnnt-1.1b')")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host address to bind the server to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5001,
                        help="Port number to run the server on (default: 5001)")
    parser.add_argument("--load_on_start", action='store_true',
                        help="Attempt to load the model in the background after server starts")
    args = parser.parse_args()
    # Initialize the global handler (model is NOT loaded here)
    try:
        logger.info(f"Initializing NemoServerModelHandler for target model '{args.model_name}'...")
        # nemoHandler is global
        nemoHandler = NemoServerModelHandler(args.model_name)
    except Exception as handlerInitError:
        logger.critical(f"Failed to initialize NemoServerModelHandler: {handlerInitError}",
                        exc_info=True)
        # Use basic print as logging might not be fully flushed on critical exit
        print(f"CRITICAL: Failed to initialize NemoServerModelHandler: {handlerInitError}",
              file=sys.stderr)
        exit(1)
    # Start initial load in background thread *if* requested, *before* starting Flask app run
    loadThread = None  # Initialize thread variable
    if args.load_on_start:
        logger.info("Starting initial model load in background thread (--load_on_start)...")
        # Pass the global nemoHandler to the task function
        loadThread = threading.Thread(target=initialModelLoadTask, args=(nemoHandler,),
                                      name="InitialModelLoader", daemon=True)
        loadThread.start()
    else:
        logger.info("Model will NOT be loaded automatically on start. Use the /load endpoint.")
    # Start Flask server AFTER initializing handler and potentially starting load thread
    logger.info(
        f"Preparing to start Flask server for model '{args.model_name}' on {args.host}:{args.port}")
    logger.info("Endpoints available at /status, /load, /unload, /transcribe")
    logger.info("Use Ctrl+C in this terminal to stop the server.")
    logger.info("Calling app.run()...")
    # Run the Flask app in the main thread (blocking)
    try:
        # use_reloader=False is important when running scripts non-interactively
        # or when using background threads, as reloader can cause issues.
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    except Exception as flaskError:
        logger.critical(f"Flask server failed to run: {flaskError}", exc_info=True)
    finally:
        logger.info("Flask server run() finished or was interrupted.")
        logger.info("Server shutting down...")
        if nemoHandler:
            logger.info("Attempting final model unload during shutdown...")
            nemoHandler.unloadModel()  # Attempt to unload model on shutdown
        logger.info("Server stopped.")
        logging.shutdown()  # Flush and close log handlers
