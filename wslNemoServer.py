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
# - Logs activities to wslNemoServer.log (now via DynamicLogger).
# - Listens for HTTP requests from the main application (running on Windows).
# - Provides endpoints: /transcribe, /load, /unload, /status.
#
# Usage:
# - Run this script from within your WSL distribution.
# - Ensure 'dynamicLogger.py' is in the same directory or Python path.
# - Example: python wslNemoServer.py --model_name "nvidia/parakeet-rnnt-1.1b" --port 5001 --load_on_start
# ==============================================================================
import argparse
import gc
import \
    logging  # Retain for logging constants (e.g., logging.DEBUG) and managing third-party loggers
import sys
import threading
import time
import numpy as np
import torch
import os  # For log file deletion
from pathlib import Path  # For log file path handling

# --- DynamicLogger Import ---
# Assume dynamicLogger.py is in the same directory or accessible via PYTHONPATH
try:
    from dynamicLogger import DynamicLogger

    dynamicLoggerAvailable = True
except ImportError as e_dl:
    # Use print for critical early errors as logging might not be fully set up
    print(
        f"CRITICAL ERROR: DynamicLogger library not found. Ensure dynamicLogger.py is accessible: {e_dl}",
        file=sys.stderr)


    # Fallback to basic print logger if DynamicLogger is unavailable
    class PrintLoggerFallback:
        def _log(self, level, msg, exc_info=False, **_):
            print(f"{level}: {msg}",
                  file=sys.stderr if level in ["ERROR", "CRITICAL"] else sys.stdout)
            if exc_info:
                import traceback
                traceback.print_exc(file=sys.stderr)

        def debug(self, msg, **kwargs): self._log("DEBUG", msg, **kwargs)

        def info(self, msg, **kwargs): self._log("INFO", msg, **kwargs)

        def warning(self, msg, **kwargs): self._log("WARNING", msg, **kwargs)

        def error(self, msg, exc_info=False, **kwargs): self._log("ERROR", msg, exc_info=exc_info,
                                                                  **kwargs)

        def critical(self, msg, exc_info=False, **kwargs): self._log("CRITICAL", msg,
                                                                     exc_info=exc_info, **kwargs)


    serverLogger = PrintLoggerFallback()
    print("WARNING: DynamicLogger not found, falling back to basic print statements for logging.",
          file=sys.stderr)
    dynamicLoggerAvailable = False

# --- Flask Setup ---
try:
    from flask import Flask, request, jsonify

    flaskAvailable = True
except ImportError:
    # This print remains critical as Flask is essential
    print("CRITICAL ERROR: Flask library not found. pip install Flask", file=sys.stderr)
    if dynamicLoggerAvailable:  # Try to log with DynamicLogger if it was found, otherwise already using PrintLoggerFallback
        serverLogger.critical("Flask library not found. Server cannot start.")
    sys.exit(1)

# --- WSL Server Log Configuration Sets (for DynamicLogger) ---
wslServerLogConfigSets = {
    'default': {
        'logLevel': logging.DEBUG,
        'handlers': [
            {
                'handlerType': 'console',
                'logLevel': logging.DEBUG,  # Keep console verbose for server debugging
                'logFormat': '%(asctime)s | %(levelname)-8s | [%(threadName)s] | %(callerInfo)s:%(lineno)d | %(message)s',
                'timestampFormat': '%Y-%m-%d %H:%M:%S,%f'[:-3]
            },
            {
                'handlerType': 'file',
                'logLevel': logging.DEBUG,
                'filePath': 'wslNemoServer.log',  # DynamicLogger handles this path
                'logFormat': '%(asctime)s | %(levelname)-8s | [%(threadName)s] | %(callerInfo)s:%(lineno)d | %(message)s',
                'timestampFormat': '%Y-%m-%d %H:%M:%S,%f'[:-3],
            }
        ]
    },
    'werkzeug_quiet': {  # For Flask's internal request logs if needed
        'logLevel': logging.INFO,
        'handlers': [
            {  # Only log werkzeug to file, not console, to reduce noise
                'handlerType': 'file',
                'logLevel': logging.INFO,
                'filePath': 'wslNemoServer_requests.log',
                'logFormat': '%(asctime)s | WERKZEUG | %(message)s',
                'timestampFormat': '%Y-%m-%d %H:%M:%S',
            }
        ]
    }
}

# --- Initialize DynamicLogger for the Server ---
if dynamicLoggerAvailable:
    try:
        # Simulate filemode='w' by deleting the log file before DynamicLogger initializes
        # DynamicLogger's default file mode is 'a' (append).
        logFilePath = Path('wslNemoServer.log')  # Assuming relative to script location
        if logFilePath.exists():
            try:
                logFilePath.unlink()
                # Initial print, as logger isn't fully up yet for this specific action
                print(
                    f"INFO: Deleted existing log file '{logFilePath}' to simulate 'w' mode for DynamicLogger.",
                    file=sys.stdout)
            except OSError as e_del:
                print(f"WARNING: Could not delete existing log file '{logFilePath}': {e_del}",
                      file=sys.stderr)

        serverLogger = DynamicLogger(
            name="WSLNemoServer",  # Unique name for this logger instance
            logConfigSets=wslServerLogConfigSets
            # No highOrderOptions needed for the server at this time
        )
    except Exception as e_dyn_logger_init:
        print(
            f"CRITICAL ERROR: Failed to initialize DynamicLogger for WSLNemoServer: {e_dyn_logger_init}",
            file=sys.stderr)


        # Fallback again if init fails after successful import
        class PrintLoggerFallbackOnError:  # Slightly different name to avoid conflict
            def _log(self, level, msg, exc_info=False, **_):
                print(f"{level}: {msg}",
                      file=sys.stderr if level in ["ERROR", "CRITICAL"] else sys.stdout)
                if exc_info:
                    import traceback
                    traceback.print_exc(file=sys.stderr)

            def debug(self, msg, **kwargs): self._log("DEBUG", msg, **kwargs)

            def info(self, msg, **kwargs): self._log("INFO", msg, **kwargs)

            def warning(self, msg, **kwargs): self._log("WARNING", msg, **kwargs)

            def error(self, msg, exc_info=False, **kwargs): self._log("ERROR", msg,
                                                                      exc_info=exc_info, **kwargs)

            def critical(self, msg, exc_info=False, **kwargs): self._log("CRITICAL", msg,
                                                                         exc_info=exc_info,
                                                                         **kwargs)


        serverLogger = PrintLoggerFallbackOnError()
        serverLogger.warning(
            "DynamicLogger initialization failed, falling back to basic print statements.")

# Silence noisy third-party libraries (using standard Python logging for them)
libraries_to_silence = {
    "urllib3": logging.WARNING,
    "requests": logging.WARNING,
    "huggingface_hub": logging.INFO,
    "nemo_toolkit": logging.INFO,  # NeMo's own INFO logs can be useful for its operations
    "torch": logging.WARNING,
    "werkzeug": logging.INFO  # Flask's internal server logs requests. Can be noisy.
    # Consider creating a specific DynamicLogger config set for werkzeug if needed
    # and routing its logs, or set to WARNING to quiet it more.
}
for lib_name, level in libraries_to_silence.items():
    try:
        logging.getLogger(lib_name).setLevel(level)
    except Exception as e_lib_silence:
        serverLogger.warning(f"Could not set log level for library '{lib_name}': {e_lib_silence}")

serverLogger.info(f"--- WSL NeMo Server Script Started (using DynamicLogger if available) ---")
serverLogger.info(
    f"DynamicLogger for WSLNemoServer initialized. Main log file: wslNemoServer.log (or as configured)")

app = Flask(__name__)


# --- Global Server State ---
class NemoServerModelHandler:
    """
    Manages the NeMo ASR model lifecycle and transcription within the server.
    Delays NeMo import until loadModel is called.
    """

    def __init__(self, targetModelName):
        self.targetModelName = targetModelName
        self.model = None
        self.modelLoaded = False
        self.loadInProgress = False
        self.loadError = None
        self.device = None
        self._determineDevice()
        serverLogger.info(
            f"NemoServerModelHandler initialized for target model: {self.targetModelName} on device {self.device}")
        serverLogger.info(f"Model '{self.targetModelName}' is initially NOT loaded.")

    def _determineDevice(self):
        """Determines the compute device (CUDA GPU or CPU)."""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            try:
                gpuName = torch.cuda.get_device_name(self.device)
            except Exception as gpuError:
                gpuName = f"Error getting name: {gpuError}"
            serverLogger.info(
                f"CUDA available ({gpuName}). NeMo server will use GPU ({self.device}).")
        else:
            self.device = torch.device('cpu')
            serverLogger.info("CUDA not available. NeMo server will use CPU.")
            serverLogger.warning(
                "Running NeMo models on CPU can be very slow and memory-intensive.")

    def _cudaClean(self):
        """Performs garbage collection and attempts to clear PyTorch's CUDA cache."""
        serverLogger.debug("Attempting to clean CUDA memory (NeMo Server)...")
        collected = gc.collect()
        serverLogger.debug(f"Garbage collector collected {collected} objects.")
        if self.device and self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                serverLogger.debug("torch.cuda.empty_cache() called.")
            except Exception as e:
                serverLogger.warning(f"CUDA memory cleaning attempt failed partially: {e}")
        serverLogger.debug("CUDA memory cleaning attempt finished (NeMo Server).")

    def loadModel(self):
        """
        Imports NeMo and loads the ASR model.
        Returns True on success, False on failure.
        """
        if self.loadInProgress:
            serverLogger.warning(
                "Load request ignored: Another load operation is already in progress.")
            return False
        self.loadInProgress = True
        self.loadError = None
        ASRModel = None
        try:
            serverLogger.info("Attempting NeMo import inside loadModel...")
            from nemo.collections.asr.models import ASRModel
            serverLogger.info("NeMo imported successfully.")
        except ImportError:
            errorMessage = "Cannot load NeMo model - NeMo toolkit not found or import failed during load request."
            serverLogger.critical(errorMessage)
            serverLogger.critical(
                "Please ensure 'nemo_toolkit[asr]' is installed in the WSL environment.")
            self.loadError = errorMessage
            self.loadInProgress = False
            return False
        except Exception as importError:
            errorMessage = f"Unexpected error importing NeMo during load request: {importError}"
            serverLogger.critical(errorMessage, exc_info=True)
            self.loadError = errorMessage
            self.loadInProgress = False
            return False

        if self.modelLoaded:
            serverLogger.info(
                f"NeMo model '{self.targetModelName}' already loaded (checked after import).")
            self.loadInProgress = False
            return True

        serverLogger.info(
            f"Attempting to load NeMo model '{self.targetModelName}' to {self.device}...")
        self._cudaClean()
        startTime = time.time()
        success = False
        try:
            serverLogger.debug(f"Calling ASRModel.from_pretrained('{self.targetModelName}')...")
            newModel = ASRModel.from_pretrained(self.targetModelName)
            serverLogger.debug(f"Model loaded from pretrained, moving to device: {self.device}")
            newModel = newModel.to(self.device)
            serverLogger.debug("Setting model to evaluation mode...")
            newModel.eval()
            self.model = newModel
            self.modelLoaded = True
            success = True
            loadTime = time.time() - startTime
            serverLogger.info(
                f"NeMo model '{self.targetModelName}' loaded successfully to {self.device} in {loadTime:.2f}s.")
        except Exception as loadErrorDetail:
            errorMessage = f"CRITICAL FAILURE loading NeMo model '{self.targetModelName}': {loadErrorDetail}"
            serverLogger.critical(errorMessage, exc_info=True)  # Changed from error to critical
            serverLogger.error(  # Keep additional hints as error
                "Check model name, internet connection (for download), NeMo installation, dependencies, and available memory (RAM/VRAM).")
            self.modelLoaded = False
            self.model = None
            self.loadError = str(loadErrorDetail)
            self._cudaClean()
            success = False
        finally:
            self.loadInProgress = False
            serverLogger.info(f"Load attempt finished. Success: {success}")
        return success

    def unloadModel(self):
        """Unloads the NeMo ASR model."""
        if not self.modelLoaded:
            serverLogger.info("NeMo model already unloaded.")
            return True
        serverLogger.info(f"Unloading NeMo model '{self.targetModelName}' from {self.device}...")
        unloadSuccess = False
        try:
            if self.model is not None:
                try:
                    serverLogger.debug("Moving model to CPU before unloading...")
                    self.model.to('cpu')
                except Exception as cpuMoveError:
                    serverLogger.warning(
                        f"Ignoring error while moving model to CPU during unload: {cpuMoveError}")
                serverLogger.debug("Deleting model object reference...")
                del self.model
                self.model = None
            self.modelLoaded = False
            self.loadError = None
            self._cudaClean()
            serverLogger.info(f"NeMo model '{self.targetModelName}' unloaded.")
            unloadSuccess = True
        except Exception as e:
            serverLogger.error(f"Error during NeMo model unload: {e}", exc_info=True)
            self.modelLoaded = False
            unloadSuccess = False
        return unloadSuccess

    def transcribeAudioData(self, audioDataBytes, sampleRate, targetLang):
        """
        Transcribes audio data received as bytes. Converts bytes to numpy array first.
        Uses the provided targetLang for multilingual models. Handles Hypothesis objects.
        """
        if not self.modelLoaded or self.model is None:
            serverLogger.error("NeMo transcription skipped - Model not loaded.")
            return None
        transcriptionText = None
        try:
            try:
                audioNp = np.frombuffer(audioDataBytes, dtype=np.float32)
                if len(audioNp) == 0:
                    serverLogger.warning("Received zero-length audio data after numpy conversion.")
                    return ""
            except ValueError as e:
                serverLogger.error(f"Could not convert received bytes to float32 numpy array: {e}")
                return None
            audioDurationSeconds = len(audioNp) / sampleRate if sampleRate > 0 else 0
            serverLogger.info(
                f"Received {len(audioDataBytes)} bytes ({len(audioNp)} samples, {audioDurationSeconds:.2f}s, Lang: {targetLang}) for transcription.")
            serverLogger.debug(
                f"Starting NeMo transcription (Lang: {targetLang}) on device {self.device}...")
            startTime = time.time()

            if 'ASRModel' not in locals() and 'ASRModel' not in globals():  # Safety check
                serverLogger.warning(
                    "NeMo ASRModel not imported yet. Attempting import before transcription.")
                try:
                    from nemo.collections.asr.models import ASRModel
                    serverLogger.info("NeMo imported successfully during transcription request.")
                except Exception as importError:
                    serverLogger.error(
                        f"Failed to import NeMo during transcription request: {importError}",
                        exc_info=True)
                    return None

            with torch.no_grad():
                transcriptionKwargs = {
                    'audio': [audioNp],
                    'batch_size': 1
                }
                if 'canary' in self.targetModelName.lower():
                    transcriptionKwargs['target_lang'] = targetLang
                    transcriptionKwargs['task'] = 'asr'
                    serverLogger.debug(
                        f"Added target_lang='{targetLang}' and task='asr' for Canary model.")
                elif targetLang and targetLang != 'en':  # Only warn if a non-English lang is given for non-Canary
                    serverLogger.warning(
                        f"Target language '{targetLang}' provided, but model '{self.targetModelName}' might not be multilingual or use this parameter. Transcription will proceed without it.")

                if self.model is None:  # Should be caught by earlier check but good to be defensive
                    serverLogger.error("Transcription cannot proceed: self.model is None.")
                    return None
                transcriptionResults = self.model.transcribe(**transcriptionKwargs)

            transcribeTime = time.time() - startTime
            serverLogger.info(f"NeMo transcription finished in {transcribeTime:.2f}s.")

            if transcriptionResults and isinstance(transcriptionResults, list):
                firstResult = transcriptionResults[0]
                if hasattr(firstResult, 'text') and isinstance(firstResult.text, str):
                    transcriptionText = firstResult.text
                elif isinstance(firstResult, str):
                    transcriptionText = firstResult
                elif isinstance(firstResult, list) and firstResult and isinstance(firstResult[0],
                                                                                  str):
                    transcriptionText = firstResult[0]
                else:
                    serverLogger.warning(
                        f"Unexpected result format inside list: {type(firstResult)}. Full: {transcriptionResults}")
                    transcriptionText = ""
            elif isinstance(transcriptionResults, str):
                transcriptionText = transcriptionResults
            else:
                serverLogger.warning(
                    f"Empty or unexpected result format: {type(transcriptionResults)}. Full: {transcriptionResults}")
                transcriptionText = ""
            transcriptionText = transcriptionText.strip() if transcriptionText else ""
        except Exception as e:
            serverLogger.error(f"Error during NeMo transcription/processing: {e}", exc_info=True)
            transcriptionText = None
        return transcriptionText


# Global handler instance (initialized later based on args)
nemoHandler = None
app = Flask(__name__)  # Flask app instance


# --- Flask Routes ---
@app.route('/status', methods=['GET'])
def getStatus():
    global nemoHandler
    if not nemoHandler:
        serverLogger.error("Status request failed: Server handler not initialized.")
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
    serverLogger.debug(
        f"Status request received. Reporting status: '{currentStatus}' for model '{nemoHandler.targetModelName}'")
    return jsonify({
        "status": currentStatus,
        "message": message,
        "modelName": nemoHandler.targetModelName,
        "device": str(nemoHandler.device)
    }), 200


@app.route('/load', methods=['POST'])
def loadModelEndpoint():
    global nemoHandler
    if not nemoHandler:
        serverLogger.error("Load request failed: Server handler not initialized.")
        return jsonify({"status": "error", "message": "Server handler not initialized"}), 500
    if nemoHandler.loadInProgress:
        serverLogger.warning("Load request received, but load already in progress.")
        return jsonify({"status": "loading", "message": "Model load already in progress",
                        "modelName": nemoHandler.targetModelName}), 409
    if nemoHandler.modelLoaded:
        serverLogger.warning(
            f"Load request received but model '{nemoHandler.targetModelName}' is already loaded.")
        return jsonify({"status": "loaded", "message": "Model already loaded",
                        "modelName": nemoHandler.targetModelName}), 200
    serverLogger.info(
        f"Received /load request. Starting load in background thread for '{nemoHandler.targetModelName}'...")
    thread = threading.Thread(target=nemoHandler.loadModel, name="ModelLoaderThread", daemon=True)
    thread.start()
    serverLogger.info("Model loading initiated in background thread.")
    return jsonify({"status": "loading", "message": "Model loading initiated in background",
                    "modelName": nemoHandler.targetModelName}), 202


@app.route('/unload', methods=['POST'])
def unloadModelEndpoint():
    global nemoHandler
    if not nemoHandler:
        serverLogger.error("Unload request failed: Server handler not initialized.")
        return jsonify({"status": "error", "message": "Server handler not initialized"}), 500
    if nemoHandler.loadInProgress:
        serverLogger.warning("Unload request received, but model loading is currently in progress.")
        return jsonify({"status": "loading", "message": "Cannot unload while model is loading",
                        "modelName": nemoHandler.targetModelName}), 409
    if not nemoHandler.modelLoaded:
        serverLogger.warning("Unload request received but model is already unloaded.")
        return jsonify({"status": "unloaded", "message": "Model already unloaded",
                        "modelName": nemoHandler.targetModelName}), 200
    serverLogger.info(f"Received request to unload model '{nemoHandler.targetModelName}'...")
    success = nemoHandler.unloadModel()
    if success:
        return jsonify({"status": "unloaded", "message": "Model unloaded successfully",
                        "modelName": nemoHandler.targetModelName}), 200
    else:
        serverLogger.error("An issue occurred during model unload (check server logs).")
        return jsonify({"status": "error", "message": "An issue occurred during unload",
                        "modelName": nemoHandler.targetModelName}), 500


@app.route('/transcribe', methods=['POST'])
def transcribe():
    global nemoHandler
    endpointStartTime = time.time()
    if not nemoHandler:
        serverLogger.error("Transcribe request failed: Server handler not initialized.")
        return jsonify({"status": "error", "message": "Server handler not initialized"}), 500
    if nemoHandler.loadInProgress:
        serverLogger.warning("Transcription request received, but model loading is in progress.")
        return jsonify({"status": "loading",
                        "message": "Model is currently loading, please wait."}), 503
    if not nemoHandler.modelLoaded:
        errorMessage = "Model is not loaded."
        if nemoHandler.loadError:
            errorMessage = f"Model failed to load ({nemoHandler.loadError}). Use /load to retry."
        serverLogger.error(f"Transcription request received, but {errorMessage}")
        return jsonify({"status": "error", "message": errorMessage}), 503
    if 'audio_data' not in request.files:
        serverLogger.error("Transcribe request failed: Missing 'audio_data' in request files.")
        return jsonify({"status": "error", "message": "Missing 'audio_data' in request files"}), 400
    audioFile = request.files['audio_data']
    audioBytes = audioFile.read()
    sampleRateString = request.args.get('sample_rate')
    if not sampleRateString:
        serverLogger.error("Transcribe request failed: Missing 'sample_rate' query parameter.")
        return jsonify({"status": "error", "message": "Missing 'sample_rate' query parameter"}), 400
    try:
        sampleRate = int(sampleRateString)
    except (TypeError, ValueError):
        serverLogger.error(
            f"Invalid 'sample_rate' query parameter '{sampleRateString}'. Must be integer.")
        return jsonify(
            {"status": "error", "message": "Invalid 'sample_rate', must be an integer"}), 400
    targetLang = request.args.get('target_lang')
    if not targetLang:
        serverLogger.error("Transcribe request failed: Missing 'target_lang' query parameter.")
        return jsonify({"status": "error", "message": "Missing 'target_lang' query parameter"}), 400
    if not audioBytes:
        serverLogger.warning("Received empty audio data in transcribe request.")
        return jsonify({"transcription": ""}), 200
    serverLogger.info(
        f"Processing transcription request: {len(audioBytes)} bytes, Rate: {sampleRate}Hz, Lang: {targetLang}")
    resultText = nemoHandler.transcribeAudioData(audioBytes, sampleRate, targetLang)
    if resultText is not None:
        serverLogger.info(f"Sending transcription result: '{resultText[:100]}...'")
        endpointDuration = time.time() - endpointStartTime
        serverLogger.info(f"/transcribe endpoint processed in {endpointDuration:.3f} seconds")
        return jsonify({"transcription": resultText}), 200
    else:
        serverLogger.error("Transcription failed on server during processing.")
        return jsonify({"status": "error",
                        "message": "Transcription failed on server (check server logs)"}), 500


# --- Background Load Function ---
def initialModelLoadTask(handler):
    serverLogger.info("[InitialModelLoader] Background thread started...")
    if handler:
        serverLogger.info("[InitialModelLoader] Calling handler.loadModel()...")
        loadOk = handler.loadModel()
        if loadOk:
            serverLogger.info("[InitialModelLoader] Background initial model load successful.")
        else:
            serverLogger.error(
                "[InitialModelLoader] Background initial model load failed. Check logs above for details.")
    else:
        serverLogger.error(
            "[InitialModelLoader] Background load task cannot run: Nemo handler is None.")
    serverLogger.info("[InitialModelLoader] Background initial model load thread finished.")


# --- Main Execution ---
if __name__ == "__main__":
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

    try:
        serverLogger.info(
            f"Initializing NemoServerModelHandler for target model '{args.model_name}'...")
        nemoHandler = NemoServerModelHandler(args.model_name)
    except Exception as handlerInitError:
        # This will use the fallback PrintLogger if DynamicLogger itself failed during its own init
        serverLogger.critical(f"Failed to initialize NemoServerModelHandler: {handlerInitError}",
                              exc_info=True)
        sys.exit(1)

    loadThread = None
    if args.load_on_start:
        serverLogger.info("Starting initial model load in background thread (--load_on_start)...")
        loadThread = threading.Thread(target=initialModelLoadTask, args=(nemoHandler,),
                                      name="InitialModelLoader", daemon=True)
        loadThread.start()
    else:
        serverLogger.info(
            "Model will NOT be loaded automatically on start. Use the /load endpoint.")

    serverLogger.info(
        f"Preparing to start Flask server for model '{args.model_name}' on {args.host}:{args.port}")
    serverLogger.info("Endpoints available at /status, /load, /unload, /transcribe")
    serverLogger.info("Use Ctrl+C in this terminal to stop the server.")
    serverLogger.info("Calling app.run()...")

    try:
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    except Exception as flaskError:
        serverLogger.critical(f"Flask server failed to run: {flaskError}", exc_info=True)
    finally:
        serverLogger.info("Flask server run() finished or was interrupted.")
        serverLogger.info("Server shutting down...")
        if nemoHandler:
            serverLogger.info("Attempting final model unload during shutdown...")
            nemoHandler.unloadModel()
        serverLogger.info("Server stopped.")
        # Standard logging shutdown; DynamicLogger handles its own resources or relies on GC.
        logging.shutdown()
