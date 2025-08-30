# wslNemoServer.py
# ==============================================================================
# NeMo ASR Model Server (for WSL Environment)
# ==============================================================================
#
# Purpose:
# - Runs a Flask web server inside a WSL environment.
# - Loads a NeMo ASR model and exposes it via HTTP endpoints.
# - Listens for requests from the main application on Windows.
# - Provides endpoints: /transcribe, /load, /unload, /status.
#
# Usage:
# - Run from within WSL:
#   python wslNemoServer.py --model_name "nvidia/parakeet-rnnt-1.1b" --port 5001 --load_on_start
# ==============================================================================
import argparse
import gc
import sys
import threading
import time
import numpy as np
import torch

# Import the new universal logger instances
from utils.loggerInstance import uniLogger, uniDebugLogger

# --- Flask Setup ---
try:
    # Import Flask components for creating the web server
    from flask import Flask, request, jsonify
except ImportError:
    # If Flask is not installed, print a critical error and exit, as the server cannot run.
    print("CRITICAL ERROR: Flask library not found. pip install Flask", file=sys.stderr)
    sys.exit(1)

# Log that the server script has started.
uniLogger.info("--- WSL NeMo Server Script Started ---")


# --- Global Server State ---
class NemoServerModelHandler:
    """
    Manages the NeMo ASR model lifecycle (loading, unloading, transcription)
    and state within the server.
    """

    def __init__(self, targetModelName):
        """Initializes the model handler with the target model name and determines the compute device."""
        self.targetModelName = targetModelName
        self.model = None  # Holds the loaded NeMo model object
        self.modelLoaded = False  # Flag indicating if the model is ready
        self.loadInProgress = False  # Flag to prevent concurrent load operations
        self.loadError = None  # Stores error message if loading fails
        self.device = None  # Will be 'cuda' or 'cpu'
        self._determineDevice()
        uniLogger.info(
            f"NemoServerModelHandler initialized for model: {self.targetModelName} on device {self.device}")

    def _determineDevice(self):
        """Determines the compute device (CUDA GPU or CPU) for NeMo."""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            gpuName = torch.cuda.get_device_name(self.device)
            uniLogger.info(f"CUDA available ({gpuName}). NeMo server will use GPU.")
        else:
            self.device = torch.device('cpu')
            uniLogger.info("CUDA not available. NeMo server will use CPU.")
            uniLogger.warning("Running NeMo models on CPU can be very slow.")

    def _cudaClean(self):
        """Performs garbage collection and attempts to clear PyTorch's CUDA cache."""
        uniDebugLogger.debug("Cleaning CUDA memory (NeMo Server)...")
        gc.collect()  # Run Python's garbage collector
        if self.device and self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                uniDebugLogger.debug("torch.cuda.empty_cache() called.")
            except Exception as e:
                uniLogger.warning(f"CUDA memory cleaning failed: {e}")

    def loadModel(self):
        """Imports NeMo dynamically and loads the ASR model from pretrained weights."""
        if self.loadInProgress:
            uniLogger.warning("Load request ignored: load already in progress.")
            return False
        self.loadInProgress = True
        self.loadError = None
        try:
            # Import NeMo here to keep initial startup fast and avoid dependency errors if not used.
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
        self._cudaClean()  # Clean memory before loading a new model
        startTime = time.time()
        try:
            # Load the model from Hugging Face or local cache
            newModel = ASRModel.from_pretrained(self.targetModelName)
            # Move the model to the determined device (GPU/CPU)
            newModel = newModel.to(self.device)
            # Set the model to evaluation mode (disables dropout, etc.)
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
            self._cudaClean()  # Attempt to clean up if loading failed
            return False
        finally:
            self.loadInProgress = False

    def unloadModel(self):
        """Unloads the NeMo ASR model and frees up memory."""
        if not self.modelLoaded:
            uniLogger.info("NeMo model already unloaded.")
            return True
        uniLogger.info(f"Unloading NeMo model '{self.targetModelName}'...")
        try:
            # Delete the model object to release its memory
            del self.model
            self.model = None
            self.modelLoaded = False
            self.loadError = None
            self._cudaClean()  # Clean GPU memory after unloading
            uniLogger.info(f"NeMo model unloaded.")
            return True
        except Exception as e:
            uniLogger.error(f"Error during NeMo model unload: {e}", excInfo=True)
            self.modelLoaded = False
            return False

    def transcribeAudioData(self, audioDataBytes, sampleRate, targetLang):
        """Transcribes audio data received as bytes using the loaded NeMo model."""
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

            # Process the model's output to get the final text
            if transcriptionResults and isinstance(transcriptionResults[0], (str, list)):
                result_text = transcriptionResults[0]
                if isinstance(result_text, list):  # Handle batched output format
                    result_text = result_text[0]
                return result_text.strip()
            return ""  # Return empty string if transcription is empty
        except Exception as e:
            uniLogger.error(f"Error during transcription: {e}", excInfo=True)
            return None


# Global handler instance, initialized when the script starts
nemoHandler = None
# Flask app instance
app = Flask(__name__)


@app.route('/status', methods=['GET'])
def getStatus():
    """Endpoint to check the current status of the model (loaded, unloaded, error)."""
    if not nemoHandler:
        return jsonify({"status": "error", "message": "Server handler not initialized"}), 500
    status = "unloaded"
    message = "Model is not loaded."
    if nemoHandler.loadInProgress:
        status, message = "loading", "Model load is in progress."
    elif nemoHandler.modelLoaded:
        status, message = "loaded", "Model is loaded and ready."
    elif nemoHandler.loadError:
        status, message = "error", f"Model loading failed: {nemoHandler.loadError}"
    return jsonify({
        "status": status, "message": message,
        "modelName": nemoHandler.targetModelName, "device": str(nemoHandler.device)
    }), 200


@app.route('/load', methods=['POST'])
def loadModelEndpoint():
    """Endpoint to trigger loading the model in a background thread."""
    if not nemoHandler:
        return jsonify({"status": "error", "message": "Server handler not initialized"}), 500
    if nemoHandler.loadInProgress:
        return jsonify({"status": "loading", "message": "Model load already in progress"}), 409
    if nemoHandler.modelLoaded:
        return jsonify({"status": "loaded", "message": "Model already loaded"}), 200

    uniLogger.info("Received /load request. Starting load in background...")
    # Start loading in a separate thread to avoid blocking the server
    thread = threading.Thread(target=nemoHandler.loadModel, name="ModelLoaderThread", daemon=True)
    thread.start()
    return jsonify({"status": "loading", "message": "Model loading initiated"}), 202


@app.route('/unload', methods=['POST'])
def unloadModelEndpoint():
    """Endpoint to trigger unloading the model."""
    if not nemoHandler: return jsonify(
        {"status": "error", "message": "Server handler not initialized"}), 500
    if nemoHandler.loadInProgress: return jsonify(
        {"status": "loading", "message": "Cannot unload while model is loading"}), 409
    if not nemoHandler.modelLoaded: return jsonify(
        {"status": "unloaded", "message": "Model already unloaded"}), 200

    uniLogger.info(f"Received request to unload model...")
    if nemoHandler.unloadModel():
        return jsonify({"status": "unloaded", "message": "Model unloaded successfully"}), 200
    else:
        return jsonify({"status": "error", "message": "An issue occurred during unload"}), 500


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Endpoint to receive audio data and return the transcription."""
    # --- Pre-checks for server and model state ---
    if not nemoHandler: return jsonify(
        {"status": "error", "message": "Server handler not initialized"}), 500
    if not nemoHandler.modelLoaded: return jsonify(
        {"status": "error", "message": "Model is not loaded"}), 503
    if 'audio_data' not in request.files: return jsonify(
        {"status": "error", "message": "Missing 'audio_data'"}), 400

    try:
        # --- Extract data from the request ---
        audioBytes = request.files['audio_data'].read()
        sampleRate = int(request.args.get('sample_rate'))
        targetLang = request.args.get('target_lang')
        if not audioBytes: return jsonify({"transcription": ""}), 200

        # --- Perform transcription ---
        resultText = nemoHandler.transcribeAudioData(audioBytes, sampleRate, targetLang)
        if resultText is not None:
            return jsonify({"transcription": resultText}), 200
        else:
            return jsonify({"status": "error", "message": "Transcription failed on server"}), 500
    except Exception as e:
        # Catch errors from invalid request parameters (e.g., non-integer sample_rate)
        uniLogger.error(f"Error in /transcribe endpoint: {e}", excInfo=True)
        return jsonify({"status": "error", "message": "Invalid request parameters"}), 400


def initialModelLoadTask(handler):
    """A target function for a background thread to load the model on startup."""
    uniLogger.info("[InitialLoad] Background thread started for initial model load.")
    if handler and not handler.loadModel():
        uniLogger.error("[InitialLoad] Background initial model load FAILED.")


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="NeMo ASR Model Server for WSL")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the NeMo ASR model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=5001, help="Port to run on")
    parser.add_argument("--load_on_start", action='store_true',
                        help="Load model in background on start")
    args = parser.parse_args()

    try:
        # --- Initialization ---
        nemoHandler = NemoServerModelHandler(args.model_name)
        if args.load_on_start:
            # If specified, start loading the model immediately in the background
            threading.Thread(target=initialModelLoadTask, args=(nemoHandler,), daemon=True).start()

        # --- Start Server ---
        uniLogger.info(f"Starting Flask server for '{args.model_name}' on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    except Exception as e:
        uniLogger.critical(f"Server failed to run: {e}", excInfo=True)
    finally:
        # --- Cleanup ---
        uniLogger.info("Server shutting down...")
        if nemoHandler: nemoHandler.unloadModel()  # Attempt to unload the model on exit
        uniLogger.info("Server stopped.")
