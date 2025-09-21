# wsl_server/server_app.py
# ==============================================================================
# NeMo ASR Model Server Application (Flask)
# ==============================================================================
#
# Purpose:
# - This is the main entry point for the WSL ASR server.
# - It runs a Flask web server to expose the NeMo model's functionality.
# - It handles command-line arguments for configuration.
# - It uses the NemoServerModelHandler to manage the AI model.
# ==============================================================================
import argparse
import sys
import threading
from utils.loggerInstance import uniLogger
from modelHandler import NemoServerModelHandler

# --- Flask Setup ---
try:
    from flask import Flask, request, jsonify
except ImportError:
    uniLogger.critical(
        "CRITICAL ERROR: Flask library not found. Please run 'pip install Flask' in your WSL environment.")
    sys.exit(1)

# --- Flask App Initialization ---
app = Flask(__name__)
nemoHandler = None  # Global handler instance, initialized in __main__


# --- Flask Routes ---
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
    if not nemoHandler:
        return jsonify({"status": "error", "message": "Server handler not initialized"}), 500
    if nemoHandler.loadInProgress:
        return jsonify(
            {"status": "loading", "message": "Cannot unload while model is loading"}), 409
    if not nemoHandler.modelLoaded:
        return jsonify({"status": "unloaded", "message": "Model already unloaded"}), 200

    uniLogger.info(f"Received request to unload model...")
    if nemoHandler.unloadModel():
        return jsonify({"status": "unloaded", "message": "Model unloaded successfully"}), 200
    else:
        return jsonify({"status": "error", "message": "An issue occurred during unload"}), 500


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Endpoint to receive audio data and return the transcription."""
    if not nemoHandler or not nemoHandler.modelLoaded:
        return jsonify({"status": "error", "message": "Model is not loaded"}), 503
    if 'audio_data' not in request.files:
        return jsonify({"status": "error", "message": "Missing 'audio_data'"}), 400

    try:
        audioBytes = request.files['audio_data'].read()
        sampleRate = int(request.args.get('sample_rate'))
        targetLang = request.args.get('target_lang')

        if not audioBytes:
            return jsonify({"transcription": ""}), 200

        resultText = nemoHandler.transcribeAudioData(audioBytes, sampleRate, targetLang)
        if resultText is not None:
            return jsonify({"transcription": resultText}), 200
        else:
            return jsonify({"status": "error", "message": "Transcription failed on server"}), 500
    except Exception as e:
        uniLogger.error(f"Error in /transcribe endpoint: {e}", excInfo=True)
        return jsonify({"status": "error", "message": "Invalid request parameters"}), 400


def initialModelLoadTask(handler):
    """
    A target function for a background thread to load the model on startup.
    Args:
        handler (NemoServerModelHandler): The instance of the model handler.
    """
    uniLogger.info("[InitialLoad] Background thread started for initial model load.")
    if handler and not handler.loadModel():
        uniLogger.error("[InitialLoad] Background initial model load FAILED.")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="NeMo ASR Model Server for WSL")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the NeMo ASR model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=5001, help="Port to run on")
    parser.add_argument("--load_on_start", action='store_true',
                        help="Load model in background on start")
    args = parser.parse_args()

    try:
        # Initialize the global model handler instance
        nemoHandler = NemoServerModelHandler(args.model_name)

        # If specified, start loading the model immediately in the background
        if args.load_on_start:
            threading.Thread(target=initialModelLoadTask, args=(nemoHandler,), daemon=True).start()

        # Start the Flask web server
        uniLogger.info(f"Starting Flask server for '{args.model_name}' on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)

    except Exception as e:
        uniLogger.critical(f"Server failed to run: {e}", excInfo=True)
    finally:
        uniLogger.info("Server shutting down...")
        if nemoHandler:
            nemoHandler.unloadModel()
        uniLogger.info("Server stopped.")
