# utils.py
import \
    logging  # Still needed for logging constants (e.g., logging.DEBUG) and managing third-party loggers
import sys
import traceback
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
                lib_logger = logging.getLogger(lib_name)
                lib_logger.setLevel(level)
                # Optional: If these libraries should *never* output through DynamicLogger's
                # handlers (e.g. if DynamicLogger is root), clear their handlers
                # and set propagate=False. For now, DynamicLogger's own propagate=False
                # on its named logger should prevent duplication.
                # lib_logger.handlers.clear()
                # lib_logger.propagate = False
            except Exception as e_lib:
                print(f"Warning: Could not set log level for library '{lib_name}': {e_lib}",
                      file=sys.stderr)

        if appDynamicLogger:  # Check if initialization was successful before logging
            appDynamicLogger.info(f"DynamicLogging configured for '{loggerName}'.")

    except Exception as e:
        print(f"CRITICAL ERROR: Could not configure DynamicLogger: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)  # Print traceback for config error

        # Fallback logger
        class PrintLoggerFallback:
            def _log(self, level, msg, exc_info=None, **_):  # Adjusted to accept exc_info
                print(f"{level}: {msg}",
                      file=sys.stderr if level in ["ERROR", "CRITICAL"] else sys.stdout)
                if exc_info:  # If exc_info is True or a tuple
                    current_exc_info = sys.exc_info()
                    if not (current_exc_info[0] is None and current_exc_info[1] is None and
                            current_exc_info[2] is None):
                        traceback.print_exception(*current_exc_info, file=sys.stderr)
                    elif isinstance(exc_info, tuple):  # If exc_info was passed as a tuple
                        traceback.print_exception(*exc_info, file=sys.stderr)

            def debug(self, msg, **kwargs):
                self._log("DEBUG", msg, **kwargs)

            def info(self, msg, **kwargs):
                self._log("INFO", msg, **kwargs)

            def warning(self, msg, **kwargs):
                self._log("WARNING", msg, **kwargs)

            def error(self, msg, exc_info=False,
                      **kwargs):  # Default exc_info to False for fallback
                self._log("ERROR", msg, exc_info=exc_info, **kwargs)

            def critical(self, msg, exc_info=False,
                         **kwargs):  # Default exc_info to False for fallback
                self._log("CRITICAL", msg, exc_info=exc_info, **kwargs)

        appDynamicLogger = PrintLoggerFallback()
        appLogger = appDynamicLogger  # Also set appLogger to the fallback
        appDynamicLogger.critical(
            f"DynamicLogger initialization failed: {e}. Falling back to print-based logging.",
            exc_info=True)  # Log the original error with traceback


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
    """
    Logs an error message.
    If exc_info is True, current exception information is captured and logged.
    If exc_info is an exception tuple (type, value, traceback), it's used.
    If exc_info is False or None, no exception info is logged beyond the message.
    DynamicLogger's .error() method defaults its internal excInfo to True.
    """
    final_exc_info_for_dynamic_logger = None
    if exc_info:  # Covers True or an actual exception tuple
        if isinstance(exc_info, bool):  # If it's specifically True
            # Capture current exception. If no exception, sys.exc_info() is (None, None, None)
            # DynamicLogger will handle this by not printing a traceback if it's all None.
            final_exc_info_for_dynamic_logger = True  # Let DynamicLogger call sys.exc_info()
        elif isinstance(exc_info, tuple) and len(exc_info) == 3:  # It's an explicit exception tuple
            final_exc_info_for_dynamic_logger = exc_info
        else:  # Unrecognized exc_info type, treat as no exception info for safety
            final_exc_info_for_dynamic_logger = False
    else:  # exc_info is False or None
        final_exc_info_for_dynamic_logger = False

    if appDynamicLogger:
        # Pass the processed exc_info value to DynamicLogger's `excInfo` parameter
        # DynamicLogger.error defaults its internal excInfo to True if not provided or None.
        # We are being more explicit here.
        if final_exc_info_for_dynamic_logger is True:  # Let DynamicLogger handle True
            appDynamicLogger.error(message,
                                   **kwargs)  # Relies on DynamicLogger's default excInfo=True for .error()
        elif final_exc_info_for_dynamic_logger:  # It's a tuple
            appDynamicLogger.error(message, excInfo=final_exc_info_for_dynamic_logger, **kwargs)
        else:  # It's False or None (from processing)
            appDynamicLogger.error(message, excInfo=False, **kwargs)

    else:  # Logger not initialized fallback
        print(f"ERROR (logger_uninit): {message}", file=sys.stderr)
        if final_exc_info_for_dynamic_logger:  # If True or a tuple
            current_exc_info_tuple = sys.exc_info()
            if not (current_exc_info_tuple[0] is None and current_exc_info_tuple[1] is None and
                    current_exc_info_tuple[2] is None):
                traceback.print_exception(*current_exc_info_tuple, file=sys.stderr)
            elif isinstance(final_exc_info_for_dynamic_logger, tuple):
                traceback.print_exception(*final_exc_info_for_dynamic_logger, file=sys.stderr)


def logCritical(message, exc_info=None, **kwargs):
    """
    Logs a critical message.
    If exc_info is True, current exception information is captured and logged.
    If exc_info is an exception tuple (type, value, traceback), it's used.
    If exc_info is False or None, no exception info is logged beyond the message.
    DynamicLogger's .critical() method defaults its internal excInfo to True.
    """
    final_exc_info_for_dynamic_logger = None
    if exc_info:  # Covers True or an actual exception tuple
        if isinstance(exc_info, bool):  # If it's specifically True
            final_exc_info_for_dynamic_logger = True  # Let DynamicLogger call sys.exc_info()
        elif isinstance(exc_info, tuple) and len(exc_info) == 3:  # It's an explicit exception tuple
            final_exc_info_for_dynamic_logger = exc_info
        else:  # Unrecognized exc_info type, treat as no exception info for safety
            final_exc_info_for_dynamic_logger = False
    else:  # exc_info is False or None
        final_exc_info_for_dynamic_logger = False

    if appDynamicLogger:
        if final_exc_info_for_dynamic_logger is True:
            appDynamicLogger.critical(message,
                                      **kwargs)  # Relies on DynamicLogger's default excInfo=True for .critical()
        elif final_exc_info_for_dynamic_logger:  # It's a tuple
            appDynamicLogger.critical(message, excInfo=final_exc_info_for_dynamic_logger, **kwargs)
        else:  # It's False or None
            appDynamicLogger.critical(message, excInfo=False, **kwargs)
    else:  # Logger not initialized fallback
        print(f"CRITICAL (logger_uninit): {message}", file=sys.stderr)
        if final_exc_info_for_dynamic_logger:  # If True or a tuple
            current_exc_info_tuple = sys.exc_info()
            if not (current_exc_info_tuple[0] is None and current_exc_info_tuple[1] is None and
                    current_exc_info_tuple[2] is None):
                traceback.print_exception(*current_exc_info_tuple, file=sys.stderr)
            elif isinstance(final_exc_info_for_dynamic_logger, tuple):
                traceback.print_exception(*final_exc_info_for_dynamic_logger, file=sys.stderr)


def get_logger_for_level_check(loggerName="SpeechToTextApp"):
    """
    Returns a standard Python logger instance.
    Used for isEnabledFor checks if needed, though DynamicLogger's
    filtering is comprehensive.
    """
    return logging.getLogger(loggerName)


# --- Path Conversion ---
def convertWindowsPathToWsl(windowsPath) -> str | None:
    """
    Converts a Windows path (absolute path or Path object) to its WSL equivalent.
    Handles drive letters and basic UNC paths. Returns None on failure.
    """
    try:
        windowsPathConverted = Path(windowsPath)
        # Define logging functions based on appDynamicLogger availability
        log_warning_func = logWarning if appDynamicLogger else lambda msg, **kwargs: print(
            f"WARNING: {msg}", file=sys.stderr)
        log_error_func = logError if appDynamicLogger else lambda msg, **kwargs: print(
            f"ERROR: {msg}", file=sys.stderr)
        # For debug, if logger not available, make it a no-op to avoid too much print spam
        log_debug_func = logDebug if appDynamicLogger else lambda msg, **kwargs: None

        if not windowsPathConverted.is_absolute():
            resolvedPath = windowsPathConverted.resolve()
            log_warning_func(
                f"Path '{windowsPathConverted}' is not absolute. Attempting resolution to '{resolvedPath}' for WSL conversion.")
            windowsPathConverted = resolvedPath
            if not windowsPathConverted.is_absolute():
                log_error_func("Cannot convert relative path to WSL path after resolution attempt.")
                return None

        pathStr = str(windowsPathConverted)
        log_debug_func(f"Attempting WSL path conversion for: {pathStr}")

        # Handle standard drive paths (C:\, D:\ etc.)
        if len(pathStr) >= 2 and pathStr[1] == ':':
            driveLetter = pathStr[0].lower()
            restOfPath = pathStr[2:].replace('\\', '/')
            wslPath = f"/mnt/{driveLetter}{restOfPath}"
            log_debug_func(f"Converted Windows drive path to WSL path: '{wslPath}'")
            return wslPath
        # Handle basic UNC paths (\\server\share\...)
        elif pathStr.startswith('\\\\'):
            log_warning_func(
                "Attempting basic UNC path conversion for WSL - might not work depending on WSL mount configuration.")
            wslPath = pathStr.replace('\\', '/')  # Basic replacement
            log_debug_func(f"Converted Windows UNC path to potential WSL path: '{wslPath}'")
            return wslPath
        else:
            log_error_func(f"Unrecognized Windows path format for WSL conversion: {pathStr}")
            return None
    except Exception as e:
        # Define or redefine the error logging function for the except block
        # This ensures it handles the parameters correctly if appDynamicLogger is not available.
        if appDynamicLogger:
            current_log_error_func = logError
        else:
            def fallback_error_logger(msg, exc_info=None, **kwargs):
                print(f"ERROR: {msg}", file=sys.stderr)
                # If exc_info is True (passed from logError call), print current exception
                # This lambda doesn't directly receive sys.exc_info(), so we rely on the caller
                # of logError to have passed exc_info=True, which then logError (in utils)
                # should handle by calling sys.exc_info() or letting DynamicLogger do it.
                # For this direct fallback, if exc_info is True, we print the current traceback.
                if exc_info is True:  # Check specifically for True
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                elif isinstance(exc_info, tuple):  # If it's an explicit tuple
                    import traceback
                    traceback.print_exception(*exc_info, file=sys.stderr)

            current_log_error_func = fallback_error_logger

        current_log_error_func(f"Error during Windows path conversion to WSL: {e}", exc_info=True)
        # Note: If appDynamicLogger was None, the fallback_error_logger already printed the traceback
        # if exc_info=True was passed. If appDynamicLogger was active, logError in utils.py handles exc_info.
        return None
