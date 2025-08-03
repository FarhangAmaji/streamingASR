# utils.py
import \
    logging  # Still needed for logging constants (e.g., logging.DEBUG) and managing third-party loggers
import sys
from pathlib import Path
from dynamicLogger import DynamicLogger  # Import DynamicLogger
from define_logConfigSets import defaultLogConfigSets  # Import the configurations

# --- Global DynamicLogger Instance ---
appDynamicLogger = None
appLogger = None  # Kept for potential (but discouraged) direct use


# --- Logging Configuration Function ---
def configure_dynamic_logging(logConfigSets=None, highOrderOptions=None,
                              loggerName="SpeechToTextApp"):
    """
    Configures and initializes the global appDynamicLogger instance.
    This function should be called once at the beginning of the main script.
    """
    global appDynamicLogger, appLogger

    logConfigSetsToUse = logConfigSets if logConfigSets is not None else defaultLogConfigSets
    highOrderOptionsToUse = highOrderOptions if highOrderOptions is not None else {}

    try:
        appDynamicLogger = DynamicLogger(
            name=loggerName,
            logConfigSets=logConfigSetsToUse,
            highOrderOptions=highOrderOptionsToUse
        )
        # For any code that might still be trying to use the old appLogger directly
        # (though this should be minimal to none after refactoring).
        appLogger = appDynamicLogger

        libraries_to_silence = {
            "sounddevice": logging.WARNING, "requests": logging.WARNING,
            "urllib3": logging.WARNING, "huggingface_hub": logging.INFO,
            "transformers": logging.INFO, "torch": logging.WARNING,
            "keyboard": logging.INFO, "nemo_toolkit": logging.INFO,
            "werkzeug": logging.INFO
        }
        for lib_name, level in libraries_to_silence.items():
            try:
                # Get the standard Python logger for the library
                lib_logger = logging.getLogger(lib_name)
                # Set its level. This prevents it from propagating messages below this level.
                lib_logger.setLevel(level)
                # Crucially, ensure these loggers don't use DynamicLogger's handlers
                # by clearing their handlers and setting propagate to False if they
                # should *only* be silent and not log anywhere else.
                # Or, if they should log to a basic console, ensure they have a handler.
                # For now, just setting level is the primary goal.
                # If they acquire handlers from root, DynamicLogger's `propagate=False` on its own
                # underlying logger should prevent duplication if it's named differently than root.
                # If DynamicLogger's underlying logger *is* the root or a parent, this needs care.
                # DynamicLogger sets self.logger.propagate = False, which is good.
            except Exception as e_lib:
                print(f"Warning: Could not set log level for library '{lib_name}': {e_lib}",
                      file=sys.stderr)

        if appDynamicLogger:  # Check if initialization was successful before logging
            appDynamicLogger.info(f"DynamicLogging configured for '{loggerName}'.")

    except Exception as e:
        print(f"CRITICAL ERROR: Could not configure DynamicLogger: {e}", file=sys.stderr)

        # Fallback logger
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

            def error(self, msg, exc_info=False, **kwargs): self._log("ERROR", msg,
                                                                      exc_info=exc_info, **kwargs)

            def critical(self, msg, exc_info=False, **kwargs): self._log("CRITICAL", msg,
                                                                         exc_info=exc_info,
                                                                         **kwargs)

        appDynamicLogger = PrintLoggerFallback()
        appLogger = appDynamicLogger
        appDynamicLogger.critical(
            f"DynamicLogger initialization failed: {e}. Falling back to print-based logging.",
            exc_info=True)


# --- Logging Helper Functions (using DynamicLogger) ---

def logDebug(message, **kwargs):
    if appDynamicLogger:
        appDynamicLogger.debug(message, **kwargs)
    else:
        print(f"DEBUG (logger_uninit): {message}", file=sys.stderr)


def logInfo(message, **kwargs):
    if appDynamicLogger:
        appDynamicLogger.info(message, **kwargs)
    else:
        print(f"INFO (logger_uninit): {message}", file=sys.stderr)


def logWarning(message, **kwargs):
    if appDynamicLogger:
        appDynamicLogger.warning(message, **kwargs)
    else:
        print(f"WARNING (logger_uninit): {message}", file=sys.stderr)


def logError(message, exc_info=None, **kwargs):
    # DynamicLogger's .error() method defaults exc_info to True.
    # If exc_info is explicitly passed as False here, we honor that.
    # Otherwise, let DynamicLogger's default (True for .error()) take effect.
    if appDynamicLogger:
        if exc_info is False:  # Explicitly False
            appDynamicLogger.error(message, excInfo=False, **kwargs)
        else:  # None or True, rely on DynamicLogger's default or pass True if exc_info is a tuple
            passed_exc_info = exc_info if exc_info is not None else True  # Let DynamicLogger handle True correctly
            appDynamicLogger.error(message, excInfo=passed_exc_info, **kwargs)
    else:
        print(f"ERROR (logger_uninit): {message}", file=sys.stderr)
        if exc_info:
            import traceback
            traceback.print_exc(file=sys.stderr)


def logCritical(message, exc_info=None, **kwargs):
    # DynamicLogger's .critical() method defaults exc_info to True.
    if appDynamicLogger:
        if exc_info is False:  # Explicitly False
            appDynamicLogger.critical(message, excInfo=False, **kwargs)
        else:  # None or True, rely on DynamicLogger's default or pass True if exc_info is a tuple
            passed_exc_info = exc_info if exc_info is not None else True  # Let DynamicLogger handle True correctly
            appDynamicLogger.critical(message, excInfo=passed_exc_info, **kwargs)
    else:
        print(f"CRITICAL (logger_uninit): {message}", file=sys.stderr)
        if exc_info:
            import traceback
            traceback.print_exc(file=sys.stderr)


def get_logger_for_level_check(loggerName="SpeechToTextApp"):
    """
    Returns a standard Python logger instance.
    Used for isEnabledFor checks if needed, though DynamicLogger's
    filtering is comprehensive.
    """
    # This will return a standard logger. It won't reflect DynamicLogger's complex conditional logic
    # for whether a message will *actually* be output.
    # It's generally better to let DynamicLogger decide if a message should be logged.
    return logging.getLogger(loggerName)


# --- Path Conversion ---
def convertWindowsPathToWsl(windowsPath) -> str | None:
    """
    Converts a Windows path (absolute path or Path object) to its WSL equivalent.
    Handles drive letters and basic UNC paths. Returns None on failure.
    """
    try:
        windowsPathConverted = Path(windowsPath)
        if not windowsPathConverted.is_absolute():
            resolvedPath = windowsPathConverted.resolve()
            if appDynamicLogger:  # Check if logger is available
                logWarning(
                    f"Path '{windowsPathConverted}' is not absolute. Attempting resolution to '{resolvedPath}' for WSL conversion.")
            else:
                print(
                    f"WARNING: Path '{windowsPathConverted}' is not absolute. Attempting resolution to '{resolvedPath}' for WSL conversion.",
                    file=sys.stderr)
            windowsPathConverted = resolvedPath
            if not windowsPathConverted.is_absolute():
                if appDynamicLogger:
                    logError("Cannot convert relative path to WSL path after resolution attempt.")
                else:
                    print(
                        "ERROR: Cannot convert relative path to WSL path after resolution attempt.",
                        file=sys.stderr)
                return None
        pathStr = str(windowsPathConverted)
        if appDynamicLogger:
            logDebug(f"Attempting WSL path conversion for: {pathStr}")
        # Handle standard drive paths (C:\, D:\ etc.)
        if len(pathStr) >= 2 and pathStr[1] == ':':
            driveLetter = pathStr[0].lower()
            restOfPath = pathStr[2:].replace('\\', '/')
            # Standard WSL mount point is /mnt/<drive_letter>
            wslPath = f"/mnt/{driveLetter}{restOfPath}"
            if appDynamicLogger:
                logDebug(f"Converted Windows drive path to WSL path: '{wslPath}'")
            return wslPath
        # Handle basic UNC paths (\\server\share\...)
        elif pathStr.startswith('\\\\'):
            if appDynamicLogger:
                logWarning(
                    "Attempting basic UNC path conversion for WSL - might not work depending on WSL mount configuration.")
            else:
                print(
                    "WARNING: Attempting basic UNC path conversion for WSL - might not work depending on WSL mount configuration.",
                    file=sys.stderr)
            wslPath = pathStr.replace('\\', '/')
            if appDynamicLogger:
                logDebug(f"Converted Windows UNC path to potential WSL path: '{wslPath}'")
            return wslPath
        else:
            if appDynamicLogger:
                logError(f"Unrecognized Windows path format for WSL conversion: {pathStr}")
            else:
                print(f"ERROR: Unrecognized Windows path format for WSL conversion: {pathStr}",
                      file=sys.stderr)

            return None
    except Exception as e:
        if appDynamicLogger:
            logError(f"Error during Windows path conversion to WSL: {e}", exc_info=True)
        else:
            print(f"ERROR: Error during Windows path conversion to WSL: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
        return None
