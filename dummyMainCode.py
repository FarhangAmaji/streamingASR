# dummyMainCode.py
# ==============================================================================
# Minimal Client to Test WSL Server Launch, Connection, and Load Monitoring
# ==============================================================================
import logging
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse
import requests # Requires: pip install requests
import sys

# --- Basic Logging Setup ---
logLevel = logging.DEBUG
logFormat = '%(asctime)s - %(levelname)-8s - [%(threadName)s] - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
dateFormat = '%Y-%m-%d %H:%M:%S,%f'[:-3]
logging.basicConfig(level=logLevel, format=logFormat, datefmt=dateFormat, stream=sys.stdout)
logger = logging.getLogger("DummyClient")

# --- Configuration (Hardcoded for Simplicity) ---
config = {
    "wslDistro": "Ubuntu-22.04", # Your WSL distribution name
    "wslServerUrl": "http://localhost:5001",
    "wslUseSudo": True, # <<< SET THIS TO True or False based on your setup
    "modelName": "nvidia/canary-180m-flash", # Model for the server to load
    "serverConnectTimeout": 5.0,
    "serverRequestTimeout": 15.0,
    "serverReadyTimeout": 120.0, # Increased timeout for model loading
    "scriptDir": Path(os.path.dirname(os.path.abspath(__file__))), # Assumes dummyWslServer.py is relative
    "wslServerScriptName": "dummyWslServer.py"
}

# --- WSL Path Conversion Helper ---
def convertWindowsPathToWsl(windowsPath) -> str | None:
    """ Converts a Windows path (Path object) to its WSL equivalent. """
    try:
        pathStr = str(windowsPath.resolve()) # Resolve to absolute path
        if len(pathStr) >= 2 and pathStr[1] == ':':
            driveLetter = pathStr[0].lower()
            restOfPath = pathStr[2:].replace('\\', '/')
            wslPath = f"/mnt/{driveLetter}{restOfPath}"
            logger.debug(f"Converted '{pathStr}' to WSL path '{wslPath}'")
            return wslPath
        else:
            logger.error(f"Cannot convert non-drive path: {pathStr}")
            return None
    except Exception as e:
        logger.error(f"Error converting path {windowsPath}: {e}")
        return None

# --- Global State ---
wslServerProcess = None
wslLaunchCommand = []
serverReachable = None
modelLoaded = False

# --- Functions Replicating Orchestrator Logic ---
def prepareWslLaunchCommand():
    """ Prepares the wsl.exe command list. """
    global wslLaunchCommand
    wslLaunchCommand = []
    logger.info("Preparing WSL launch command...")
    wslDistro = config["wslDistro"]
    useSudo = config["wslUseSudo"]
    modelName = config["modelName"]
    try:
        serverUrl = config["wslServerUrl"]
        parsedUrl = urlparse(serverUrl)
        wslServerPort = parsedUrl.port
        if not wslServerPort: raise ValueError("No port in URL")

        scriptDir = config["scriptDir"]
        wslServerScriptFilename = config["wslServerScriptName"]
        wslServerScriptPathWindows = scriptDir / wslServerScriptFilename
        if not wslServerScriptPathWindows.is_file():
             raise FileNotFoundError(f"Script not found: {wslServerScriptPathWindows}")
        logger.debug(f"Found WSL server script: {wslServerScriptPathWindows}")

        wslServerScriptPathWsl = convertWindowsPathToWsl(wslServerScriptPathWindows)
        if not wslServerScriptPathWsl: raise ValueError("Path conversion failed")
        logger.debug(f"WSL script path: {wslServerScriptPathWsl}")

        pythonExecutable = "/usr/bin/python3"
        commandBase = ["wsl.exe", "-d", wslDistro, "--"]
        commandInsideWsl = []
        if useSudo:
            logger.warning("wslUseSudo is True. Adding 'sudo'. Ensure passwordless sudo is configured!")
            commandInsideWsl.append("sudo")
        commandInsideWsl.append(pythonExecutable)
        commandInsideWsl.append(wslServerScriptPathWsl)
        commandInsideWsl.extend([
            "--model_name", modelName,
            "--port", str(wslServerPort),
            "--load_on_start" # Let the dummy server handle this flag
        ])
        wslLaunchCommand = commandBase + commandInsideWsl
        logger.info("WSL launch command prepared.")
        logger.debug(f"Command List: {wslLaunchCommand}")
        cmdString = subprocess.list2cmdline(wslLaunchCommand)
        logger.info(f"Command String (for manual test): {cmdString}")
        return True
    except Exception as e:
        logger.error(f"Failed to prepare WSL launch command: {e}", exc_info=True)
        wslLaunchCommand = []
        return False

def launchWslServer():
    """ Launches the WSL server process. """
    global wslServerProcess
    if not wslLaunchCommand: logger.error("Cannot launch, command not prepared."); return False
    if wslServerProcess and wslServerProcess.poll() is None: logger.info("Server process already running."); return True
    if platform.system() != "Windows": logger.error("Cannot launch WSL, not on Windows."); return False

    logger.info("Attempting to launch WSL server...")
    try:
        creationFlags = subprocess.CREATE_NO_WINDOW
        startupinfo = subprocess.STARTUPINFO(); startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW; startupinfo.wShowWindow = subprocess.SW_HIDE
        wslServerProcess = subprocess.Popen(
            wslLaunchCommand,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace',
            creationflags=creationFlags, startupinfo=startupinfo, bufsize=1
        )
        logger.info(f"WSL server process launched (PID: {wslServerProcess.pid}).")
        return True
    except Exception as e:
        logger.error(f"Failed to launch WSL server process: {e}", exc_info=True)
        wslServerProcess = None
        return False

def logWslProcessOutputOnError():
    """ Reads and logs output from the WSL process handle if it exists. """
    if not wslServerProcess: logger.debug("No WSL process handle to read output from."); return
    logger.info(f"Attempting to read output from WSL process (PID: {wslServerProcess.pid or 'N/A'})...")
    outputLogged = False
    try:
        stdoutData, _ = wslServerProcess.communicate(timeout=2.0)
        if stdoutData and stdoutData.strip():
            logger.error(f"--- Captured WSL Output (PID: {wslServerProcess.pid or 'N/A'}) ---")
            for line in stdoutData.strip().splitlines(): logger.error(f"WSL_OUT: {line}")
            logger.error(f"--- End WSL Output ---")
            if "sudo: a password is required" in stdoutData: logger.error("!!! Detected sudo password error !!!")
            if "Traceback" in stdoutData: logger.error("!!! Detected Python Traceback !!!")
            if "Address already in use" in stdoutData: logger.error("!!! Detected Address already in use error !!!")
            outputLogged = True
        else: logger.debug("communicate() returned no data.")
    except subprocess.TimeoutExpired: logger.warning("Timeout reading WSL output via communicate().")
    except Exception as e: logger.warning(f"Error reading WSL output via communicate(): {e}")
    if not outputLogged: logger.warning("No output captured from WSL process.")

def terminateWslServer():
    """ Terminates the WSL server process. """
    global wslServerProcess
    if not wslServerProcess: return
    logger.info(f"Terminating WSL server process (PID: {wslServerProcess.pid or 'N/A'})...")
    if wslServerProcess.poll() is None: # Only terminate if running
        try:
            wslServerProcess.terminate()
            wslServerProcess.wait(timeout=3.0)
            logger.info("WSL process terminated gracefully.")
        except subprocess.TimeoutExpired:
            logger.warning("WSL process did not terminate gracefully, killing...")
            wslServerProcess.kill()
        except Exception as e:
            logger.error(f"Error terminating WSL process: {e}")
    logWslProcessOutputOnError() # Log any final output
    wslServerProcess = None

def checkServerStatus():
    """ Checks the /status endpoint. Updates global reachable/loaded state. """
    global serverReachable, modelLoaded
    url = f"{config['wslServerUrl'].rstrip('/')}/status"
    try:
        response = requests.get(url, timeout=(config['serverConnectTimeout'], config['serverRequestTimeout']))
        response.raise_for_status()
        data = response.json()
        logger.debug(f"Status Response: {data}")
        serverReachable = True
        status = data.get("status")
        if status == "loaded": modelLoaded = True
        else: modelLoaded = False # Covers unloaded, loading, error
        return True # Request succeeded
    except requests.exceptions.ConnectionError as e:
        logger.warning(f"Connection error checking status: {e}")
        serverReachable = False; modelLoaded = False; return False
    except requests.exceptions.Timeout as e:
        logger.warning(f"Timeout checking status: {e}")
        serverReachable = None; modelLoaded = False; return False # Timeout, state unknown
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error checking status: {e}")
        serverReachable = True # Got a response, even if error
        modelLoaded = False; return False # Treat HTTP errors as failure for readiness
    except Exception as e:
        logger.error(f"Unexpected error checking status: {e}", exc_info=True)
        serverReachable = False; modelLoaded = False; return False

def waitForServerReachable(timeoutSeconds):
    """ Waits until the server responds to /status. """
    logger.info("Waiting for server reachability...")
    startTime = time.time()
    while True:
        if time.time() - startTime > timeoutSeconds:
            logger.error("Timeout waiting for server reachability.")
            return False
        if wslServerProcess and wslServerProcess.poll() is not None:
            logger.error(f"WSL process exited prematurely (code: {wslServerProcess.returncode}) while waiting for reachability.")
            logWslProcessOutputOnError()
            return False
        if checkServerStatus(): # This updates serverReachable
            if serverReachable:
                logger.info("Server is reachable.")
                return True
        time.sleep(2.0) # Poll interval

def waitForServerReady(timeoutSeconds):
    """ Waits until the server reports 'loaded' status. """
    logger.info("Waiting for server model readiness ('loaded' status)...")
    startTime = time.time()
    while True:
        if time.time() - startTime > timeoutSeconds:
            logger.error("Timeout waiting for server readiness ('loaded' status).")
            return False
        if wslServerProcess and wslServerProcess.poll() is not None:
            logger.error(f"WSL process exited prematurely (code: {wslServerProcess.returncode}) while waiting for readiness.")
            logWslProcessOutputOnError()
            return False
        if checkServerStatus(): # Updates modelLoaded
            if modelLoaded:
                logger.info("Server reported model 'loaded'.")
                return True
        time.sleep(2.0) # Poll interval

# --- Main Execution Logic ---
if __name__ == "__main__":
    logger.info("--- Dummy Client Starting ---")
    success = False
    try:
        if prepareWslLaunchCommand():
            if launchWslServer():
                if waitForServerReachable(30.0): # Wait up to 30s for Flask to start
                    # Server is running, now wait for the model to load
                    if waitForServerReady(config["serverReadyTimeout"]):
                        logger.info("SUCCESS: Server launched and model reported loaded.")
                        success = True
                    else:
                        logger.error("FAILURE: Server reachable but model did not load within timeout.")
                else:
                    logger.error("FAILURE: Server did not become reachable.")
            else:
                logger.error("FAILURE: Could not launch WSL server process.")
        else:
            logger.error("FAILURE: Could not prepare WSL launch command.")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    except Exception as e:
        logger.error(f"Unexpected error in main dummy client: {e}", exc_info=True)
    finally:
        logger.info("--- Dummy Client Shutting Down ---")
        terminateWslServer()
        logger.info(f"Final Result: {'SUCCESS' if success else 'FAILURE'}")
        exit(0 if success else 1)