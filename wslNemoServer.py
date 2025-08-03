# wslNemoServer.py

# ==============================================================================
# NeMo ASR Model Server (for WSL Environment)
# ==============================================================================
#
# Purpose:
# - Runs a simple web server (Flask) inside a WSL environment (e.g., Ubuntu).
# - Loads a specified Nvidia NeMo ASR model using the nemo_toolkit.
# - Listens for HTTP requests from the main application (running on Windows).
# - Provides endpoints to:
#   - /transcribe: Receive audio data, transcribe it using the loaded NeMo model,
#                  and return the text result.
#   - /load: Trigger loading of the NeMo model.
#   - /unload: Trigger unloading of the NeMo model.
#   - /status: Report whether the model is currently loaded.
#
# Usage:
# - Run this script from within your WSL distribution.
# - Pass the required NeMo model name as a command-line argument.
# - Example: `python wslNemoServer.py --model_name "nvidia/parakeet-rnnt-1.1b" --port 5001`
#
# Dependencies (Ensure installed *in WSL*):
# - Python standard libraries (os, gc, tempfile, argparse, json, logging)
# - Flask: For the web server (`pip install Flask`).
# - nemo_toolkit[asr]: Nvidia NeMo toolkit for ASR models (`pip install nemo_toolkit[asr]`).
# - soundfile: For reading/writing temporary audio files (`pip install soundfile`).
# - numpy: For audio data manipulation (`pip install numpy`).
# - torch: Required by NeMo (`pip install torch`).
# ==============================================================================

import os
import gc
import tempfile
import argparse
import json
import logging
import time
from pathlib import Path

import soundfile as sf
import numpy as np
import torch

# --- Flask Setup ---
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- NeMo Setup ---
try:
    from nemo.collections.asr.models import ASRModel

    nemoAvailable = True
except ImportError:
    print("ERROR: NeMo toolkit not found. This server cannot function.")
    print("Please install it in your WSL environment: pip install nemo_toolkit[asr]")
    ASRModel = None
    nemoAvailable = False
except Exception as e:
    print(f"ERROR: Unexpected error importing NeMo: {e}")
    ASRModel = None
    nemoAvailable = False


# --- Global Server State ---
# Encapsulate model handling logic
class NemoServerModelHandler:
    """Manages the NeMo ASR model lifecycle and transcription within the server."""

    def __init__(self, modelName):
        self.modelName = modelName
        self.model = None
        self.modelLoaded = False
        self.device = None
        self._determineDevice()
        print(f"NemoServerModelHandler initialized for model: {self.modelName}")

    def _determineDevice(self):
        """Determines the compute device (CUDA GPU or CPU)."""
        # Server doesn't need 'onlyCpu' flag, it uses GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"NeMo server will use device: {self.device}")

    def _cudaClean(self):
        """Performs garbage collection and attempts to clear PyTorch's CUDA cache."""
        print("Cleaning CUDA memory (NeMo Server)...")
        gc.collect()
        if self.device and self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"WARNING: CUDA memory cleaning attempt failed partially: {e}")
        print("CUDA memory cleaning attempt finished (NeMo Server).")

    def loadModel(self):
        """Loads the NeMo ASR model."""
        if not nemoAvailable:
            print("ERROR: Cannot load NeMo model - NeMo toolkit is not available.")
            return False
        if self.modelLoaded:
            print(f"NeMo model '{self.modelName}' already loaded.")
            return True

        print(f"Loading NeMo model '{self.modelName}' to {self.device}...")
        self._cudaClean()
        startTime = time.time()
        try:
            self.model = ASRModel.from_pretrained(self.modelName).to(self.device)
            self.model.eval()
            self.modelLoaded = True
            loadTime = time.time() - startTime
            print(f"NeMo model '{self.modelName}' loaded successfully in {loadTime:.2f}s.")
            # Optional: Warm-up (usually less critical for NeMo server unless first request latency matters)
            # self._warmUpModel()
            return True
        except Exception as e:
            print(f"ERROR: Failed loading NeMo model '{self.modelName}': {e}")
            print("Check model name, internet connection, NeMo installation, and memory.")
            self.modelLoaded = False
            self.model = None
            self._cudaClean()  # Clean up potential partial load
            return False

    def unloadModel(self):
        """Unloads the NeMo ASR model."""
        if not self.modelLoaded:
            print("NeMo model already unloaded.")
            return True

        print(f"Unloading NeMo model '{self.modelName}' from {self.device}...")
        if self.model is not None:
            del self.model
            self.model = None
        self.modelLoaded = False
        self._cudaClean()
        print(f"NeMo model '{self.modelName}' unloaded.")
        return True

    def transcribeAudioData(self, audioDataBytes, sampleRate):
        """
        Transcribes audio data received as bytes. Saves to a temporary file first.
        """
        if not self.modelLoaded or self.model is None:
            print("ERROR: NeMo transcription skipped - Model not loaded.")
            return None  # Indicate failure

        tempFilePath = None
        transcription = None
        try:
            # --- Create temporary file ---
            # Suffix needs dot prefix on Linux/macOS with tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpFile:
                tempFilePath = tmpFile.name
                # Convert bytes back to numpy array (assuming float32)
                try:
                    audioNp = np.frombuffer(audioDataBytes, dtype=np.float32)
                except ValueError as e:
                    print(f"ERROR: Could not convert received bytes to float32 numpy array: {e}")
                    return None

                # Write numpy array to the temporary WAV file
                sf.write(tmpFile.name, audioNp, sampleRate, format='WAV', subtype='FLOAT')
                print(
                    f"Saved received audio ({len(audioNp) / sampleRate:.2f}s) to temp file: {tempFilePath}")

            # --- Perform NeMo Transcription ---
            print(f"Starting NeMo transcription for temp file...")
            startTime = time.time()
            with torch.no_grad():
                transcriptions = self.model.transcribe(
                    paths2audio_files=[tempFilePath],
                    batch_size=1,
                    num_workers=0  # Use 0 for simplicity/compatibility in server context
                )
            transcribeTime = time.time() - startTime
            print(f"NeMo transcription finished in {transcribeTime:.2f}s.")
            print(f"NeMo Raw Result: {transcriptions}")  # Log raw result

            # --- Extract Text ---
            if transcriptions and isinstance(transcriptions, list) and transcriptions[0]:
                if isinstance(transcriptions[0], list):  # Handle potential list of hypotheses
                    transcription = transcriptions[0][0]
                elif isinstance(transcriptions[0], str):
                    transcription = transcriptions[0]
            elif isinstance(transcriptions, str):  # Handle direct string return
                transcription = transcriptions

            if transcription is None:
                print("WARNING: NeMo transcription result was empty or in unexpected format.")
                transcription = ""  # Return empty string for consistency

        except Exception as e:
            print(f"ERROR during NeMo transcription process: {e}")
            import traceback
            print(traceback.format_exc())
            transcription = None  # Indicate failure
        finally:
            # --- Clean up temporary file ---
            if tempFilePath and os.path.exists(tempFilePath):
                try:
                    os.remove(tempFilePath)
                    print(f"Removed temporary file: {tempFilePath}")
                except Exception as e:
                    print(f"WARNING: Failed to remove temporary file {tempFilePath}: {e}")

        return transcription  # Return text string or None on error


# Global handler instance (initialized later based on args)
nemoHandler = None


# --- Flask Routes ---

@app.route('/status', methods=['GET'])
def getStatus():
    """Endpoint to check if the NeMo model is loaded."""
    if not nemoHandler:
        return jsonify({"status": "error", "message": "Server handler not initialized"}), 500
    isLoaded = nemoHandler.modelLoaded
    return jsonify({
        "status": "loaded" if isLoaded else "unloaded",
        "modelName": nemoHandler.modelName,
        "device": str(nemoHandler.device)
    }), 200


@app.route('/load', methods=['POST'])
def loadModel():
    """Endpoint to explicitly trigger loading the NeMo model."""
    if not nemoHandler:
        return jsonify({"status": "error", "message": "Server handler not initialized"}), 500
    if nemoHandler.modelLoaded:
        return jsonify({"status": "loaded", "message": "Model already loaded",
                        "modelName": nemoHandler.modelName}), 200

    print("Received request to load model...")
    success = nemoHandler.loadModel()
    if success:
        return jsonify({"status": "loaded", "message": "Model loaded successfully",
                        "modelName": nemoHandler.modelName}), 200
    else:
        return jsonify({"status": "error", "message": "Failed to load model",
                        "modelName": nemoHandler.modelName}), 500


@app.route('/unload', methods=['POST'])
def unloadModel():
    """Endpoint to explicitly trigger unloading the NeMo model."""
    if not nemoHandler:
        return jsonify({"status": "error", "message": "Server handler not initialized"}), 500
    if not nemoHandler.modelLoaded:
        return jsonify({"status": "unloaded", "message": "Model already unloaded",
                        "modelName": nemoHandler.modelName}), 200

    print("Received request to unload model...")
    success = nemoHandler.unloadModel()
    if success:
        return jsonify({"status": "unloaded", "message": "Model unloaded successfully",
                        "modelName": nemoHandler.modelName}), 200
    else:
        # Unload shouldn't usually fail unless mid-operation?
        return jsonify({"status": "error", "message": "An issue occurred during unload",
                        "modelName": nemoHandler.modelName}), 500


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Endpoint to receive audio data and return transcription."""
    global nemoHandler  # Ensure we are using the global handler

    if not nemoHandler:
        return jsonify({"status": "error", "message": "Server handler not initialized"}), 500

    # --- Ensure Model is Loaded ---
    if not nemoHandler.modelLoaded:
        print("Transcription request received, but model not loaded. Attempting to load...")
        loadSuccess = nemoHandler.loadModel()
        if not loadSuccess:
            return jsonify({"status": "error",
                            "message": "Model is not loaded and failed to load"}), 503  # Service Unavailable

    # --- Get Data from Request ---
    if 'audio_data' not in request.files:
        return jsonify({"status": "error", "message": "Missing 'audio_data' in request files"}), 400

    audioFile = request.files['audio_data']
    audioBytes = audioFile.read()

    try:
        sampleRate = int(request.args.get('sample_rate'))
        if not sampleRate:
            raise ValueError("Missing or invalid 'sample_rate' query parameter")
    except (TypeError, ValueError) as e:
        return jsonify({"status": "error", "message": f"Invalid sample rate: {e}"}), 400

    if not audioBytes:
        return jsonify({"status": "error", "message": "Received empty audio data"}), 400

    print(f"Received transcription request: {len(audioBytes)} bytes, Sample Rate: {sampleRate}Hz")

    # --- Perform Transcription ---
    resultText = nemoHandler.transcribeAudioData(audioBytes, sampleRate)

    if resultText is not None:
        print(f"Sending transcription result: '{resultText[:100]}...'")
        return jsonify({"transcription": resultText}), 200
    else:
        # Error occurred during transcription (already logged by handler)
        return jsonify({"status": "error", "message": "Transcription failed on server"}), 500


# --- Main Execution ---
if __name__ == "__main__":
    if not nemoAvailable:
        print("Exiting: NeMo toolkit is required but not available.")
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
    nemoHandler = NemoServerModelHandler(args.model_name)

    # Optionally load model on startup
    if args.load_on_start:
        print("Attempting to load model on server startup...")
        nemoHandler.loadModel()  # Attempt load, ignore failure for now, endpoint can retry

    print(f"Starting NeMo ASR server for model '{args.model_name}' on {args.host}:{args.port}")
    print("Endpoints:")
    print("  GET /status")
    print("  POST /load")
    print("  POST /unload")
    print("  POST /transcribe (expects 'audio_data' file and 'sample_rate' query param)")

    # Run the Flask app
    # Use Flask's development server (suitable for this use case)
    # Consider using a more robust server like gunicorn for production if needed
    app.run(host=args.host, port=args.port, debug=False)  # Turn debug off for default

    # Cleanup (though usually interrupted)
    print("Server shutting down...")
    if nemoHandler:
        nemoHandler.unloadModel()  # Attempt to unload model on shutdown
    print("Server stopped.")
