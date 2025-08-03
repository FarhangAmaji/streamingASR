# wslNemoServer.py

# ==============================================================================
# NeMo ASR Model Server (for WSL Environment) - Updated with Logging
# ==============================================================================
#
# Purpose:
# - Runs a simple web server (Flask) inside a WSL environment (e.g., Ubuntu).
# - Loads a specified Nvidia NeMo ASR model using the nemo_toolkit.
# - Logs activities to wslNemoServer.log.
# - Listens for HTTP requests from the main application (running on Windows).
# - Provides endpoints: /transcribe, /load, /unload, /status.
#
# Usage:
# - Run this script from within your WSL distribution.
# - Example: python wslNemoServer.py --model_name "nvidia/parakeet-rnnt-1.1b" --port 5001
# ==============================================================================

import argparse
import gc
import os
import tempfile
import time
import logging  # Import standard logging
import sys  # For stdout handler
import traceback  # For logging exceptions

import numpy as np
import soundfile as sf
import torch
# --- Flask Setup ---
from flask import Flask, request, jsonify

# --- Configure Logging ---
logFileName = 'wslNemoServer.log'
logLevel = logging.INFO  # Default level, can be changed
logFormat = '%(asctime)s - %(levelname)-8s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
dateFormat = '%Y-%m-%d %H:%M:%S,%f'[:-3]  # Milliseconds format

logging.basicConfig(level=logging.DEBUG,  # Set root logger level
                    format=logFormat,
                    datefmt=dateFormat,
                    filename=logFileName,
                    filemode='a')  # Append mode

# Console Handler (optional, but helpful for seeing logs directly in WSL console)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logging.Formatter(logFormat, datefmt=dateFormat))
consoleHandler.setLevel(logLevel)  # Console level can be INFO or DEBUG
logging.getLogger().addHandler(consoleHandler)  # Add handler to the root logger

# Silence noisy libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("nemo_toolkit").setLevel(logging.WARNING)  # Nemo can be very verbose
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.INFO)  # Flask's internal server

logger = logging.getLogger(__name__)  # Get a logger for this module
logger.info(f"WSL NeMo Server logging configured. Log file: {logFileName}")

# --- NeMo Setup ---
try:
    # Import NeMo ASRModel - adjust import based on specific NeMo versions/models if needed
    from nemo.collections.asr.models import ASRModel

    nemoAvailable = True
    logger.info("NVIDIA NeMo toolkit found and imported.")
except ImportError:
    logger.critical("NeMo toolkit not found. This server cannot function.")
    logger.critical("Please install it in your WSL environment: pip install nemo_toolkit[asr]")
    ASRModel = None
    nemoAvailable = False
except Exception as e:
    logger.critical(f"Unexpected error importing NeMo: {e}", exc_info=True)
    ASRModel = None
    nemoAvailable = False

app = Flask(__name__)


# --- Global Server State ---
class NemoServerModelHandler:
    """Manages the NeMo ASR model lifecycle and transcription within the server."""

    def __init__(self, modelName):
        self.modelName = modelName
        self.model = None
        self.modelLoaded = False
        self.device = None
        self._determineDevice()
        logger.info(f"NemoServerModelHandler initialized for model: {self.modelName}")

    # Inside NemoServerModelHandler class
    def _determineDevice(self):
        """Determines the compute device (CUDA GPU or CPU)."""
        # FOR TESTING: Force CPU
        self.device = torch.device('cpu')
        logger.info("Forcing CPU usage for NeMo server testing.")
        # Original logic commented out:
        # if torch.cuda.is_available():
        #      self.device = torch.device('cuda')
        #      logger.info("CUDA available. NeMo server will use GPU.")
        # else:
        #      self.device = torch.device('cpu')
        #      logger.info("CUDA not available. NeMo server will use CPU.")

    def _cudaClean(self):
        """Performs garbage collection and attempts to clear PyTorch's CUDA cache."""
        logger.debug("Attempting to clean CUDA memory (NeMo Server)...")
        gc.collect()
        if self.device and self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                logger.debug("torch.cuda.empty_cache() called.")
            except Exception as e:
                logger.warning(f"CUDA memory cleaning attempt failed partially: {e}")
        logger.debug("CUDA memory cleaning attempt finished (NeMo Server).")

    def loadModel(self):
        """Loads the NeMo ASR model."""
        if not nemoAvailable:
            logger.error("Cannot load NeMo model - NeMo toolkit is not available.")
            return False
        if self.modelLoaded:
            logger.info(f"NeMo model '{self.modelName}' already loaded.")
            return True

        logger.info(f"Attempting to load NeMo model '{self.modelName}' to {self.device}...")
        self._cudaClean()  # Clean before loading
        startTime = time.time()
        try:
            # Use the generic ASRModel.from_pretrained - may need adjustment for specific model types
            self.model = ASRModel.from_pretrained(self.modelName).to(self.device)
            self.model.eval()  # Set to evaluation mode
            self.modelLoaded = True
            loadTime = time.time() - startTime
            logger.info(f"NeMo model '{self.modelName}' loaded successfully in {loadTime:.2f}s.")
            return True
        except Exception as e:
            logger.error(f"Failed loading NeMo model '{self.modelName}': {e}", exc_info=True)
            logger.error("Check model name, internet connection, NeMo installation, and memory.")
            self.modelLoaded = False
            self.model = None
            self._cudaClean()  # Clean up potential partial load
            return False

    def unloadModel(self):
        """Unloads the NeMo ASR model."""
        if not self.modelLoaded:
            logger.info("NeMo model already unloaded.")
            return True

        logger.info(f"Unloading NeMo model '{self.modelName}' from {self.device}...")
        if self.model is not None:
            # Move model to CPU before deleting? Helps sometimes.
            try:
                self.model.to('cpu')
            except Exception:
                pass  # Ignore errors moving unloaded model
            del self.model
            self.model = None

        self.modelLoaded = False
        self._cudaClean()  # Clean memory AFTER deleting reference
        logger.info(f"NeMo model '{self.modelName}' unloaded.")
        return True

    def transcribeAudioData(self, audioDataBytes, sampleRate,
                            targetLang):  # Add targetLang parameter
        """
        Transcribes audio data received as bytes. Converts bytes to numpy array first.
        Uses the provided targetLang for multilingual models. Handles Hypothesis objects.
        """
        if not self.modelLoaded or self.model is None:
            logger.error("NeMo transcription skipped - Model not loaded.")
            return None  # Indicate failure
        transcriptionText = None  # Use a different variable name for the final text
        try:
            # Convert bytes back to numpy array (assuming float32)
            try:
                audioNp = np.frombuffer(audioDataBytes, dtype=np.float32)
                if len(audioNp) == 0:
                    logger.warning("Received zero-length audio data after numpy conversion.")
                    return ""
            except ValueError as e:
                logger.error(f"Could not convert received bytes to float32 numpy array: {e}")
                return None
            audioDurationSec = len(audioNp) / sampleRate if sampleRate > 0 else 0
            logger.info(
                f"Received {len(audioDataBytes)} bytes ({len(audioNp)} samples, {audioDurationSec:.2f}s, Lang: {targetLang}).")

            # --- Perform NeMo Transcription ---
            logger.info(f"Starting NeMo transcription (Lang: {targetLang})...")
            startTime = time.time()
            with torch.no_grad():
                transcriptionResults = self.model.transcribe(
                    # Use a different name for the raw result
                    audio=[audioNp],
                    batch_size=1,
                    target_lang=targetLang,
                    task='asr'
                )
            transcribeTime = time.time() - startTime
            logger.info(f"NeMo transcription finished in {transcribeTime:.2f}s.")
            # logger.debug(f"NeMo Raw Result Type: {type(transcriptionResults)}")
            # logger.debug(f"NeMo Raw Result: {transcriptionResults}") # Can be verbose

            # --- Extract Text from Hypothesis (or other potential formats) ---
            # Check if the result is a list and not empty
            if transcriptionResults and isinstance(transcriptionResults, list):
                firstResult = transcriptionResults[0]  # Get the first element

                # Check if the first element is a Hypothesis object (common in newer NeMo)
                # Need to import Hypothesis if we want explicit type checking,
                # but checking for a '.text' attribute is safer (duck typing)
                if hasattr(firstResult, 'text') and isinstance(firstResult.text, str):
                    logger.debug("Raw result is likely a Hypothesis object, extracting .text")
                    transcriptionText = firstResult.text
                # Check if it's already a string (some models might return list[str])
                elif isinstance(firstResult, str):
                    logger.debug("Raw result is a list containing a string.")
                    transcriptionText = firstResult
                # Check for nested list like [['text']]
                elif isinstance(firstResult, list) and firstResult and isinstance(firstResult[0],
                                                                                  str):
                    logger.debug("Raw result is a nested list containing a string.")
                    transcriptionText = firstResult[0]
                else:
                    logger.warning(
                        f"NeMo transcription result item has unexpected format: {type(firstResult)}. Full result: {transcriptionResults}")
                    transcriptionText = ""  # Default to empty if extraction fails

            # Handle case where transcribe might return a single string directly
            elif isinstance(transcriptionResults, str):
                logger.debug("Raw result is a single string.")
                transcriptionText = transcriptionResults
            else:
                logger.warning(
                    f"NeMo transcription result was empty or in unexpected format: {type(transcriptionResults)}. Full result: {transcriptionResults}")
                transcriptionText = ""  # Default to empty

            # Strip whitespace from the extracted text
            transcriptionText = transcriptionText.strip() if transcriptionText else ""

        except Exception as e:
            # Catch errors during transcription OR text extraction
            logger.error(f"Error during NeMo transcription/processing: {e}", exc_info=True)
            transcriptionText = None  # Indicate failure

        return transcriptionText  # Return the extracted text string or None on error


# Global handler instance (initialized later based on args)
nemoHandler = None


# --- Flask Routes ---

@app.route('/status', methods=['GET'])
def getStatus():
    """Endpoint to check if the NeMo model is loaded."""
    if not nemoHandler:
        logger.error("Status request failed: Server handler not initialized.")
        return jsonify({"status": "error", "message": "Server handler not initialized"}), 500

    isLoaded = nemoHandler.modelLoaded
    status_str = "loaded" if isLoaded else "unloaded"
    logger.info(f"Status request: Model='{nemoHandler.modelName}', Status='{status_str}'")
    return jsonify({
        "status": status_str,
        "modelName": nemoHandler.modelName,
        "device": str(nemoHandler.device)
    }), 200


@app.route('/load', methods=['POST'])
def loadModel():
    """Endpoint to explicitly trigger loading the NeMo model."""
    if not nemoHandler:
        logger.error("Load request failed: Server handler not initialized.")
        return jsonify({"status": "error", "message": "Server handler not initialized"}), 500

    if nemoHandler.modelLoaded:
        logger.warning(
            f"Load request received but model '{nemoHandler.modelName}' is already loaded.")
        return jsonify({"status": "loaded", "message": "Model already loaded",
                        "modelName": nemoHandler.modelName}), 200

    logger.info(f"Received request to load model '{nemoHandler.modelName}'...")
    success = nemoHandler.loadModel()
    if success:
        return jsonify({"status": "loaded", "message": "Model loaded successfully",
                        "modelName": nemoHandler.modelName}), 200
    else:
        # Error logged by loadModel method
        return jsonify({"status": "error", "message": "Failed to load model",
                        "modelName": nemoHandler.modelName}), 500  # Internal Server Error


@app.route('/unload', methods=['POST'])
def unloadModel():
    """Endpoint to explicitly trigger unloading the NeMo model."""
    if not nemoHandler:
        logger.error("Unload request failed: Server handler not initialized.")
        return jsonify({"status": "error", "message": "Server handler not initialized"}), 500

    if not nemoHandler.modelLoaded:
        logger.warning("Unload request received but model is already unloaded.")
        return jsonify({"status": "unloaded", "message": "Model already unloaded",
                        "modelName": nemoHandler.modelName}), 200

    logger.info(f"Received request to unload model '{nemoHandler.modelName}'...")
    success = nemoHandler.unloadModel()
    if success:
        return jsonify({"status": "unloaded", "message": "Model unloaded successfully",
                        "modelName": nemoHandler.modelName}), 200
    else:
        logger.error("An issue occurred during model unload.")
        return jsonify({"status": "error", "message": "An issue occurred during unload",
                        "modelName": nemoHandler.modelName}), 500  # Internal Server Error


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Endpoint to receive audio data and target language, returns transcription."""
    global nemoHandler  # Ensure we are using the global handler
    endpointStartTime = time.time()  # camelCase variable name
    if not nemoHandler:
        logger.error("Transcribe request failed: Server handler not initialized.")
        return jsonify({"status": "error", "message": "Server handler not initialized"}), 500

    # --- Ensure Model is Loaded ---
    if not nemoHandler.modelLoaded:
        logger.error(
            "Transcription request received, but model is not loaded. Please use /load first.")
        return jsonify({"status": "error",
                        "message": "Model is not loaded. Use /load endpoint."}), 503  # Service Unavailable

    # --- Get Data from Request ---
    if 'audio_data' not in request.files:
        logger.error("Transcribe request failed: Missing 'audio_data' in request files.")
        return jsonify({"status": "error",
                        "message": "Missing 'audio_data' in request files"}), 400  # Bad Request

    audioFile = request.files['audio_data']
    audioBytes = audioFile.read()

    # Get sample rate
    sampleRateStr = request.args.get('sample_rate')
    if not sampleRateStr:
        logger.error("Transcribe request failed: Missing 'sample_rate' query parameter.")
        return jsonify({"status": "error",
                        "message": "Missing 'sample_rate' query parameter"}), 400  # Bad Request
    try:
        sampleRate = int(sampleRateStr)
    except (TypeError, ValueError):
        logger.error(
            f"Transcribe request failed: Invalid 'sample_rate' query parameter '{sampleRateStr}'. Must be integer.")
        return jsonify({"status": "error",
                        "message": "Invalid 'sample_rate', must be an integer"}), 400  # Bad Request

    # Get target language (NEW)
    targetLang = request.args.get('target_lang')
    if not targetLang:
        # Return 400 Bad Request if target_lang is missing, as Canary requires it
        logger.error("Transcribe request failed: Missing 'target_lang' query parameter.")
        return jsonify({"status": "error",
                        "message": "Missing 'target_lang' query parameter"}), 400  # Bad Request

    # Check if audio data is empty after reading
    if not audioBytes:
        logger.warning("Received empty audio data in transcribe request.")
        # Return empty transcription for empty audio, which is a valid scenario
        return jsonify({"transcription": ""}), 200

    logger.info(
        f"Received transcription request: {len(audioBytes)} bytes, Rate: {sampleRate}Hz, Lang: {targetLang}")

    # --- Perform Transcription ---
    # Call the updated handler function which now accepts targetLang
    # and handles the Hypothesis object internally
    resultText = nemoHandler.transcribeAudioData(audioBytes, sampleRate, targetLang)

    # --- Process Result ---
    # The handler function returns None only on critical errors.
    # It returns "" if transcription was successful but yielded no text (e.g., silence).
    if resultText is not None:
        logger.info(f"Sending transcription result: '{resultText[:100]}...'")
        endpointDuration = time.time() - endpointStartTime  # camelCase variable name
        logger.info(f"/transcribe endpoint processed in {endpointDuration:.3f} seconds")
        # Return the transcription (which might be an empty string)
        return jsonify({"transcription": resultText}), 200
    else:
        # A critical error occurred during transcription (already logged by handler)
        logger.error("Transcription failed on server during processing.")
        # Return 500 Internal Server Error
        return jsonify({"status": "error",
                        "message": "Transcription failed on server"}), 500  # Internal Server Error


# --- Main Execution ---
if __name__ == "__main__":
    if not nemoAvailable:
        # Critical log already happened during import attempt
        print("Exiting: NeMo toolkit is required but not available.", file=sys.stderr)
        exit(1)

    parser = argparse.ArgumentParser(description="NeMo ASR Model Server for WSL")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the NeMo ASR model to load (e.g., 'nvidia/parakeet-rnnt-1.1b')")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host address to bind the server to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5001,
                        help="Port number to run the server on (default: 5001)")
    parser.add_argument("--load_on_start", action='store_true',
                        help="Attempt to load the model immediately on server start")

    args = parser.parse_args()

    # Initialize the global handler
    try:
        nemoHandler = NemoServerModelHandler(args.model_name)
    except Exception as handler_init_e:
        logger.critical(f"Failed to initialize NemoServerModelHandler: {handler_init_e}",
                        exc_info=True)
        exit(1)

    # Optionally load model on startup
    if args.load_on_start:
        logger.info("Attempting to load model on server startup as requested...")
        nemoHandler.loadModel()  # Attempt load, logs errors internally

    logger.info(
        f"Starting NeMo ASR server for model '{args.model_name}' on {args.host}:{args.port}")
    logger.info("Endpoints available at /status, /load, /unload, /transcribe")

    # Run the Flask app
    try:
        # Use Flask's development server (suitable for this internal use case)
        app.run(host=args.host, port=args.port, debug=False)  # Turn Flask debug off for stability
    except Exception as flask_e:
        logger.critical(f"Flask server failed to run: {flask_e}", exc_info=True)
    finally:
        # Cleanup attempt (though usually interrupted by Ctrl+C)
        logger.info("Server shutting down...")
        if nemoHandler:
            nemoHandler.unloadModel()  # Attempt to unload model on shutdown
        logger.info("Server stopped.")
        logging.shutdown()  # Flush and close log handlers
