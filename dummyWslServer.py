# dummyWslServer.py
# ==============================================================================
# Minimal NeMo ASR Model Server for Debugging Launch/Load
# ==============================================================================
#
# Purpose:
# - Minimal Flask server to test automatic launch and NeMo model loading.
# - Starts Flask immediately.
# - Attempts NeMo model load in background if --load_on_start is passed.
# - Provides /status, /load, /unload endpoints.
# - Logs to dummyWslServer.log
# ==============================================================================
import argparse
import gc
import os
import threading
import time
import logging
import sys
import traceback
import torch

# --- Flask Setup ---
try:
    from flask import Flask, jsonify, request
    flaskAvailable = True
except ImportError:
    print("CRITICAL ERROR: Flask library not found. pip install Flask", file=sys.stderr)
    sys.exit(1)

# --- NeMo Setup ---
nemoAvailable = False
ASRModel = None
try:
    from nemo.collections.asr.models import ASRModel
    nemoAvailable = True
except ImportError:
    print("WARNING: NeMo toolkit not found or import failed.", file=sys.stderr)
except Exception as e:
    print(f"ERROR importing NeMo: {e}", file=sys.stderr)

# --- Configure Logging ---
logFileName = 'dummyWslServer.log'
logLevel = logging.DEBUG # Use DEBUG for detailed logs
logFormat = '%(asctime)s - %(levelname)-8s - [%(threadName)s] - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
dateFormat = '%Y-%m-%d %H:%M:%S,%f'[:-3]
logging.basicConfig(level=logging.DEBUG,
                    format=logFormat,
                    datefmt=dateFormat,
                    filename=logFileName,
                    filemode='w') # Overwrite log each time for clean debugging
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logging.Formatter(logFormat, datefmt=dateFormat))
consoleHandler.setLevel(logLevel)
logging.getLogger().addHandler(consoleHandler)
logging.getLogger('werkzeug').setLevel(logging.INFO) # Reduce Flask request noise if needed later
logger = logging.getLogger(__name__)
logger.info(f"--- Dummy WSL Server Script Started ---")
logger.info(f"Logging configured. Level: DEBUG. Log file: {logFileName}")
if not nemoAvailable: logger.warning("NeMo toolkit not available. Model operations will fail.")

app = Flask(__name__)

# --- Minimal Model Handler Logic ---
class DummyNemoHandler:
    def __init__(self, modelName):
        self.targetModelName = modelName
        self.model = None
        self.modelLoaded = False
        self.loadInProgress = False
        self.loadError = None
        self.device = None
        self._determineDevice()
        logger.info(f"DummyNemoHandler initialized for target model: {self.targetModelName} on device {self.device}")
        logger.info(f"Model '{self.targetModelName}' is initially NOT loaded.")

    def _determineDevice(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            try: gpuName = torch.cuda.get_device_name(self.device)
            except Exception as e: gpuName = f"Error: {e}"
            logger.info(f"CUDA available ({gpuName}). Using GPU ({self.device}).")
        else:
            self.device = torch.device('cpu')
            logger.info("CUDA not available. Using CPU.")

    def loadModel(self):
        if self.loadInProgress: logger.warning("Load already in progress."); return False
        if self.modelLoaded: logger.info("Model already loaded."); return True
        if not nemoAvailable:
            self.loadError = "NeMo toolkit not available."; logger.error(self.loadError); return False

        self.loadInProgress = True
        self.loadError = None
        logger.info(f"Attempting NeMo model load: '{self.targetModelName}' to {self.device}...")
        startTime = time.time()
        success = False
        try:
            logger.debug("Calling ASRModel.from_pretrained...")
            newModel = ASRModel.from_pretrained(self.targetModelName)
            logger.debug("Moving model to device...")
            newModel = newModel.to(self.device)
            logger.debug("Setting model to eval mode...")
            newModel.eval()
            self.model = newModel
            self.modelLoaded = True
            success = True
            loadTime = time.time() - startTime
            logger.info(f"Model '{self.targetModelName}' loaded successfully in {loadTime:.2f}s.")
        except Exception as e:
            self.loadError = f"Failed loading model: {e}"
            logger.error(self.loadError, exc_info=True)
            self.modelLoaded = False
            self.model = None
            success = False
        finally:
            self.loadInProgress = False
            logger.info(f"Load attempt finished. Success: {success}")
        return success

    def unloadModel(self):
        if not self.modelLoaded: logger.info("Model already unloaded."); return True
        logger.info(f"Unloading model '{self.targetModelName}'...")
        try:
            if self.model: del self.model; self.model = None
            self.modelLoaded = False; self.loadError = None
            gc.collect(); torch.cuda.empty_cache()
            logger.info("Model unloaded.")
            return True
        except Exception as e:
            logger.error(f"Error unloading model: {e}", exc_info=True)
            return False

# Global handler instance
dummyHandler = None

# --- Background Load Task ---
def backgroundLoadTask(handler):
    logger.info("[BackgroundLoad] Thread started...")
    if handler:
        handler.loadModel() # Attempt load
    logger.info("[BackgroundLoad] Thread finished.")

# --- Flask Routes ---
@app.route('/status', methods=['GET'])
def getStatus():
    global dummyHandler
    if not dummyHandler: return jsonify({"status": "error", "message": "Handler not initialized"}), 500
    status = "loaded" if dummyHandler.modelLoaded else ("loading" if dummyHandler.loadInProgress else ("error" if dummyHandler.loadError else "unloaded"))
    message = dummyHandler.loadError if status == "error" else f"Model status is {status}"
    logger.debug(f"/status requested. Reporting: {status}")
    return jsonify({"status": status, "message": message, "modelName": dummyHandler.targetModelName, "device": str(dummyHandler.device)}), 200

@app.route('/load', methods=['POST'])
def loadModelEndpoint():
    global dummyHandler
    if not dummyHandler: return jsonify({"status": "error", "message": "Handler not initialized"}), 500
    if dummyHandler.loadInProgress: return jsonify({"status": "loading", "message": "Load already in progress"}), 409
    if dummyHandler.modelLoaded: return jsonify({"status": "loaded", "message": "Model already loaded"}), 200

    logger.info("Received /load request. Starting load in background thread.")
    thread = threading.Thread(target=dummyHandler.loadModel, name="APILoaderThread", daemon=True)
    thread.start()
    return jsonify({"status": "loading", "message": "Model load initiated"}), 202

@app.route('/unload', methods=['POST'])
def unloadModelEndpoint():
    global dummyHandler
    if not dummyHandler: return jsonify({"status": "error", "message": "Handler not initialized"}), 500
    if dummyHandler.loadInProgress: return jsonify({"status": "loading", "message": "Cannot unload while loading"}), 409
    if not dummyHandler.modelLoaded: return jsonify({"status": "unloaded", "message": "Model already unloaded"}), 200

    logger.info("Received /unload request.")
    success = dummyHandler.unloadModel()
    if success: return jsonify({"status": "unloaded", "message": "Model unloaded"}), 200
    else: return jsonify({"status": "error", "message": "Unload failed"}), 500

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dummy WSL NeMo Server")
    parser.add_argument("--model_name", type=str, required=True, help="Target NeMo model name")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=5001, help="Port number")
    parser.add_argument("--load_on_start", action='store_true', help="Load model immediately in background")
    args = parser.parse_args()

    # Initialize handler first
    try:
        dummyHandler = DummyNemoHandler(args.model_name)
    except Exception as e:
        logger.critical(f"Failed to initialize handler: {e}", exc_info=True); exit(1)

    # Start background load if requested
    if args.load_on_start:
        logger.info("Starting initial model load in background (--load_on_start)...")
        loadThread = threading.Thread(target=backgroundLoadTask, args=(dummyHandler,), name="InitialModelLoader", daemon=True)
        loadThread.start()
    else:
        logger.info("Server started. Model load not initiated automatically.")

    # Start Flask AFTER initializing handler and potentially starting load thread
    logger.info(f"Starting Flask server on {args.host}:{args.port}...")
    try:
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    except Exception as e:
        logger.critical(f"Flask server failed: {e}", exc_info=True)
    finally:
        logger.info("Flask server stopped.")
        if dummyHandler: dummyHandler.unloadModel() # Attempt cleanup
        logging.shutdown()