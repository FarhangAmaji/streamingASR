# utils.py
import logging
import sys
from pathlib import Path

# --- Global Logger Setup (Configured by main script) ---
# Get a specific logger instance for the application
# All other modules will import the helper functions below, which use this logger.
appLogger = logging.getLogger("SpeechToTextApp")


# --- Logging Configuration Function ---
def configureLogging(logFileName='mainService.log', debugMode=False):
    """Configures the application's logging to file and console."""
    logLevel = logging.DEBUG if debugMode else logging.INFO
    # More detailed format including module, function, line number, and thread name
    logFormat = '%(asctime)s - %(levelname)-8s - [%(threadName)s] - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
    dateFormat = '%Y-%m-%d %H:%M:%S,%f'[:-3]  # Milliseconds format
    formatter = logging.Formatter(logFormat, datefmt=dateFormat)
    # Use the logger obtained earlier
    logger = appLogger
    logger.setLevel(logging.DEBUG)  # Set root logger level to DEBUG to capture all messages
    # Prevent adding handlers multiple times if this function is called again
    if logger.hasHandlers():
        logger.handlers.clear()
    # --- File Handler ---
    try:
        fileHandler = logging.FileHandler(logFileName, mode='a',
                                          encoding='utf-8')  # Append mode, UTF-8
        fileHandler.setFormatter(formatter)
        # File handler logs everything from DEBUG level upwards
        fileHandler.setLevel(logging.DEBUG)
        logger.addHandler(fileHandler)
    except Exception as e:
        # Use basic print for critical logging setup errors
        print(f"CRITICAL ERROR: Could not configure file logging to '{logFileName}': {e}",
              file=sys.stderr)
    # --- Console Handler ---
    try:
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(formatter)
        # Console level depends on the debugMode setting passed to this function
        consoleHandler.setLevel(logLevel)  # Set level based on user setting
        logger.addHandler(consoleHandler)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not configure console logging: {e}", file=sys.stderr)
    # --- Silence Verbose Libraries ---
    # Set levels for other loggers to avoid excessive output
    logging.getLogger("sounddevice").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.INFO)  # Hub can be a bit noisy
    logging.getLogger("transformers").setLevel(logging.INFO)  # Transformers can be verbose
    logging.getLogger("torch").setLevel(logging.WARNING)  # PyTorch internal logs
    logging.getLogger("keyboard").setLevel(logging.INFO)  # Keyboard library logs
    # Initial log message using the configured logger
    logger.info(
        f"Logging configured. Console Level: {logging.getLevelName(logLevel)}. File: '{logFileName}'")


# --- Logging Helper Functions ---
# These functions simply call the corresponding method on the appLogger instance.
def logDebug(message):
    """Logs a message with level DEBUG."""
    appLogger.debug(message)


def logInfo(message):
    """Logs a message with level INFO."""
    appLogger.info(message)


def logWarning(message):
    """Logs a message with level WARNING."""
    appLogger.warning(message)


def logError(message, exc_info=False):  # Added exc_info parameter
    """Logs a message with level ERROR."""
    # Consider adding exc_info=True here automatically or optionally if needed often
    appLogger.error(message, exc_info=exc_info)


# --- Path Conversion ---
def convertWindowsPathToWsl(windowsPath) -> str | None:
    """
    Converts a Windows path (absolute path or Path object) to its WSL equivalent.
    Handles drive letters and basic UNC paths. Returns None on failure.
    """
    try:
        # Ensure we have a Path object and it's absolute
        windowsPath = Path(windowsPath)
        if not windowsPath.is_absolute():
            # Attempt to resolve relative to current working directory
            resolvedPath = windowsPath.resolve()
            logWarning(
                f"Path '{windowsPath}' is not absolute. Attempting resolution to '{resolvedPath}' for WSL conversion.")
            windowsPath = resolvedPath
            if not windowsPath.is_absolute():  # Check again
                logError("Cannot convert relative path to WSL path after resolution attempt.")
                return None
        pathStr = str(windowsPath)
        logDebug(f"Attempting WSL path conversion for: {pathStr}")
        # Handle standard drive paths (C:\, D:\ etc.)
        if len(pathStr) >= 2 and pathStr[1] == ':':
            driveLetter = pathStr[0].lower()
            restOfPath = pathStr[2:].replace('\\', '/')
            # Standard WSL mount point is /mnt/<drive_letter>
            wslPath = f"/mnt/{driveLetter}{restOfPath}"
            logDebug(f"Converted Windows drive path to WSL path: '{wslPath}'")
            return wslPath
        # Handle basic UNC paths (\\server\share\...)
        elif pathStr.startswith('\\\\'):
            # Simple replacement might work for some WSL UNC setups, but can be unreliable
            # A more robust solution might involve checking WSL mount points if possible
            logWarning(
                "Attempting basic UNC path conversion for WSL - might not work depending on WSL mount configuration.")
            wslPath = pathStr.replace('\\', '/')
            # Example: //server/share/folder -> //server/share/folder (may or may not work in WSL shell)
            logDebug(f"Converted Windows UNC path to potential WSL path: '{wslPath}'")
            return wslPath
        else:
            logError(f"Unrecognized Windows path format for WSL conversion: {pathStr}")
            return None
    except Exception as e:
        logError(f"Error during Windows path conversion to WSL: {e}", exc_info=True)
        return None
