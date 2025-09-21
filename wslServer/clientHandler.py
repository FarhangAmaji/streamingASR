# wsl_server/client_handler.py
# ==============================================================================
# Remote NeMo ASR Client Handler
# ==============================================================================
#
# Purpose:
# - Contains the RemoteNemoClientHandler class.
# - This class acts as a client to the remote ASR server running in WSL.
# - It handles all HTTP communication for status checks, model loading, and transcription.
# ==============================================================================
import json
import time
import requests
import numpy as np
from utils.loggerInstance import uniLogger, uniDebugLogger
from modelHandlers import AbstractAsrModelHandler


class RemoteNemoClientHandler(AbstractAsrModelHandler):
    """
    Acts as a client to a remote ASR server (wsl_server/server_app.py).
    It sends audio data via HTTP requests and receives transcription results.
    """

    def __init__(self, config):
        """
        Initializes the client with the server URL and sets initial state.
        Args:
            config (ConfigurationManager): The application's configuration object.
        """
        super().__init__(config)
        self.serverUrl = config.get('wslServerUrl')
        if not self.serverUrl:
            uniLogger.critical(
                "RemoteNemoClientHandler cannot operate: 'wslServerUrl' not found in configuration.")
            self.serverReachable = False
        else:
            self.serverReachable = None  # None = Unknown, True = Reachable, False = Unreachable

        self.modelLoaded = False
        self.lastStatusCheckTime = 0
        uniLogger.info(
            f"RemoteNeMo client handler initialized. Target server URL: {self.serverUrl}")
        self.config.set('device', 'remote_wsl')

    def _makeServerRequest(self, method: str, endpoint: str, **kwargs) -> dict | None:
        """
        Helper function to make requests to the WSL server, handling common errors.
        Args:
            method (str): HTTP method ('get', 'post', etc.).
            endpoint (str): Server endpoint (e.g., '/status').
            **kwargs: Additional arguments for requests.request.
        Returns:
            dict | None: Parsed JSON response from the server, or None if the request failed.
        """
        if not self.serverUrl:
            uniLogger.error("Cannot make server request: Server URL not configured.")
            return None
        if self.serverReachable is False:
            uniDebugLogger.debug(
                f"Skipping {method.upper()} request to {endpoint}: Server marked as unreachable.")
            return None

        url = f"{self.serverUrl.rstrip('/')}/{endpoint.lstrip('/')}"
        connectTimeout = self.config.get('serverConnectTimeout', 5.0)
        readTimeout = self.config.get('serverRequestTimeout', 15.0)
        requestTimeout = (connectTimeout, readTimeout)
        uniDebugLogger.debug(
            f"Sending {method.upper()} request to server: {url} (Timeout: {requestTimeout}s)")
        try:
            response = requests.request(method=method, url=url, timeout=requestTimeout, **kwargs)
            response.raise_for_status()
            if self.serverReachable is not True:
                uniLogger.info(f"Successfully connected to WSL server at {self.serverUrl}.")
                self.serverReachable = True
            try:
                responseData = response.json()
                uniDebugLogger.debug(f"Received JSON response from {url}: {responseData}")
                return responseData
            except json.JSONDecodeError:
                uniLogger.error(
                    f"Server response from {url} is not valid JSON. Status: {response.status_code}. Response text: {response.text[:100]}...")
                return None
        except requests.exceptions.ConnectionError as e:
            uniLogger.error(f"Connection Error connecting to WSL server at {url}: {e}")
            if self.serverReachable is not False:
                uniLogger.error(
                    "Hint: Ensure the server script is running in WSL and check firewalls.")
            self.serverReachable = False
            self.modelLoaded = False
            return None
        except requests.exceptions.Timeout as e:
            uniLogger.error(
                f"Request Timeout to WSL server at {url}: {e}")
            self.serverReachable = None
            self.modelLoaded = False
            return None
        except requests.exceptions.HTTPError as e:
            uniLogger.error(f"HTTP Error during request to WSL server {url}: {e}")
            if e.response is not None:
                uniLogger.error(
                    f"Server Response ({e.response.status_code}): {e.response.text[:200]}...")
                if e.response.status_code == 503:
                    uniLogger.warning(
                        "Server reported Service Unavailable (503), assuming model not loaded.")
                    self.modelLoaded = False
            return None
        except requests.exceptions.RequestException as e:
            uniLogger.error(f"Unhandled RequestException during request to WSL server {url}: {e}",
                            excInfo=True)
            self.serverReachable = False
            self.modelLoaded = False
            return None

    def checkServerStatus(self, forceCheck=False) -> bool:
        """
        Checks the status of the remote server's model, throttling checks unless forced.
        Args:
            forceCheck (bool): If True, bypasses throttling.
        Returns:
            bool: True if the server reports the model is 'loaded', False otherwise.
        """
        throttleSeconds = 5.0
        now = time.time()
        if not forceCheck and (now - self.lastStatusCheckTime < throttleSeconds):
            uniDebugLogger.debug("Skipping status check due to throttling.")
            return self.modelLoaded
        self.lastStatusCheckTime = now
        uniDebugLogger.debug("Checking remote server model status...")
        statusResponse = self._makeServerRequest('get', '/status')
        if statusResponse:
            serverStatus = statusResponse.get('status')
            uniLogger.info(
                f"Server Status: Status='{serverStatus}', Model='{statusResponse.get('modelName', 'N/A')}', Device='{statusResponse.get('device', 'N/A')}'")
            if serverStatus == 'loaded':
                self.modelLoaded = True
                return True
            else:
                self.modelLoaded = False
                return False
        else:
            uniLogger.warning("Failed to get server status. Assuming model is not loaded.")
            self.modelLoaded = False
            return False

    def loadModel(self) -> bool:
        """Attempts to tell the remote server to load the model if it's not already loaded."""
        uniLogger.info("Requesting remote NeMo model load/check...")
        if self.checkServerStatus(forceCheck=True):
            uniLogger.info("Remote NeMo model is already loaded on server.")
            return True
        uniLogger.info("Model not loaded, sending load request to remote server...")
        loadResponse = self._makeServerRequest('post', '/load')
        if loadResponse and loadResponse.get('status') == 'loaded':
            uniLogger.info(
                f"Remote server confirmed successful model load: '{loadResponse.get('modelName', 'N/A')}'.")
            self.modelLoaded = True
            self.serverReachable = True
            return True
        elif loadResponse:
            uniLogger.error(
                f"Remote server reported failure during load request: {loadResponse.get('message', 'Unknown error')}")
            self.modelLoaded = False
            self.serverReachable = True
            return False
        else:
            uniLogger.error(
                "Failed to trigger model load on remote server due to communication error.")
            self.modelLoaded = False
            return False

    def unloadModel(self) -> bool:
        """Attempts to tell the remote server to unload the model."""
        uniLogger.info("Requesting remote NeMo model unload...")
        unloadResponse = self._makeServerRequest('post', '/unload')
        if unloadResponse and unloadResponse.get('status') in ['unloaded', 'already_unloaded']:
            uniLogger.info(
                f"Remote server confirmed model is unloaded (status: {unloadResponse.get('status')}).")
            self.modelLoaded = False
            self.serverReachable = True
            return True
        elif unloadResponse:
            uniLogger.warning(
                f"Remote server reported an issue during unload: {unloadResponse.get('message', 'Unknown error')}")
            self.modelLoaded = False
            self.serverReachable = True
            return False
        else:
            uniLogger.warning(
                "Failed to trigger model unload on remote server due to communication error. Assuming unloaded.")
            self.modelLoaded = False
            return False

    def transcribeAudioSegment(self, audioData: np.ndarray, sampleRate: int) -> str | None:
        """Sends audio data to the remote server for transcription."""
        if self.serverReachable is False:
            uniLogger.error("Remote transcription skipped: Server marked as unreachable.")
            return None
        if not self.modelLoaded:
            uniLogger.warning(
                "Transcription requested but client believes model is not loaded. Checking status...")
            if not self.checkServerStatus(forceCheck=True):
                uniLogger.error(
                    "Remote transcription skipped: Server status check confirmed model is not loaded.")
                return None
        if audioData is None or len(audioData) == 0:
            uniDebugLogger.debug("Remote transcription skipped: No audio data provided.")
            return ""

        targetLang = self.config.get('language', 'en')
        if not targetLang:
            uniLogger.warning(
                "Target language is empty in config, defaulting to 'en' for NeMo request.")
            targetLang = 'en'

        if audioData.dtype != np.float32:
            audioData = audioData.astype(np.float32)

        audioBytes = audioData.tobytes()
        segmentDurationSec = len(audioData) / sampleRate if sampleRate > 0 else 0
        uniLogger.info(
            f"Sending {segmentDurationSec:.2f}s audio segment ({len(audioBytes)} bytes, Lang: {targetLang}) to remote NeMo server...")

        files = {'audio_data': ('audio_segment.raw', audioBytes, 'application/octet-stream')}
        params = {'sample_rate': sampleRate, 'target_lang': targetLang}
        startTime = time.time()
        transcribeResponse = self._makeServerRequest('post', '/transcribe', params=params,
                                                     files=files)
        requestTime = time.time() - startTime
        uniLogger.info(f"Remote transcription request finished in {requestTime:.3f}s.")

        if transcribeResponse and 'transcription' in transcribeResponse:
            transcription = transcribeResponse['transcription']
            uniDebugLogger.debug(f"Received transcription from server: '{transcription[:150]}...'")
            return transcription.strip() if transcription else ""
        else:
            uniLogger.error("Failed to get valid transcription from remote server.")
            return None

    def getDevice(self) -> str:
        """Returns 'remote_wsl' to indicate where processing occurs."""
        return 'remote_wsl'

    def cleanup(self):
        """Optionally triggers model unload on the remote server during cleanup."""
        uniDebugLogger.debug("RemoteNemoClientHandler cleanup initiated.")
        if self.config.get('unloadRemoteModelOnExit', True):
            if self.modelLoaded or self.serverReachable:
                uniLogger.info("Requesting remote model unload during client cleanup...")
                self.unloadModel()
        else:
            uniLogger.info("Skipping remote unload request during cleanup as per configuration.")
        uniDebugLogger.debug("RemoteNemoClientHandler cleanup complete.")
