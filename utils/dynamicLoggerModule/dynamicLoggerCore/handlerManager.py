# dynamicLoggerCore/handlerManager.py
import logging
import sys
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List

from utils.dynamicLoggerModule.dynamicLoggerCore.customFormatter import DynamicFormatter
from utils.dynamicLoggerModule.dyLogUtils import constants


# HANDLER_MANAGER_DEBUG_MODE = True # Replaced by instance-level debugPrint

# def _debugPrint(message: str): # Replaced by instance method _instanceDebugPrint
#     if HANDLER_MANAGER_DEBUG_MODE:
#         print(f"DEBUG_HandlerManager: {message}", file=sys.stderr, flush=True)


class HandlerManager:
    def __init__(self, threadLock: threading.RLock, debugPrint: bool = False):  # Added debugPrint
        self._lock = threadLock  # Shared lock from DynamicLogger instance
        self._debugPrintActive = debugPrint  # Store debugPrint state
        self._fileHandlersCache: Dict[Path, logging.FileHandler] = {}
        self._consoleHandler: Optional[logging.StreamHandler] = None
        self._instanceDebugPrint(f"Initialized. Lock ID: {id(threadLock)}")

    def _instanceDebugPrint(self, message: str):
        """Prints debug messages for this HandlerManager instance if debugPrint is enabled."""
        if self._debugPrintActive:
            print(f"DEBUG_HandlerManager ({id(self)}): {message}", file=sys.stderr, flush=True)

    def _createFormatter(self,
                         # finalResolvedOptions: Dict[str, Any], # No longer directly needed
                         # recordForFormattingHints: Optional[logging.LogRecord] # No longer directly needed
                         ) -> DynamicFormatter:
        """
        Creates a DynamicFormatter instance.
        The formatter itself will use attributes on the LogRecord (set by DynamicLogger.log)
        to determine how to format the message.
        """
        self._instanceDebugPrint("Creating new DynamicFormatter instance.")
        # Pass the debugPrint state to the formatter if it also supports it
        return DynamicFormatter(debugPrint=self._debugPrintActive)  # Pass debugPrint to formatter

    def getActiveHandlers(self,
                          finalResolvedOptions: Dict[str, Any],
                          finalEffectiveMessageLevel: int,
                          resolvedFilePath: Optional[Path],
                          recordForFormattingHints: Optional[logging.LogRecord]
                          # Kept for signature compatibility if needed later
                          ) -> List[logging.Handler]:
        """
        Gets a list of active handlers (console and/or file) based on resolved logging arguments.
        Manages handler creation, caching, and configuration (formatter, level).
        The `fileWriteMode` from `finalResolvedOptions` is implicitly handled by DynamicLogger
        *before* file handlers are created or used for writing, by deleting files/folders.
        FileHandlers here are always opened in append mode ('a').
        """
        self._instanceDebugPrint(
            f"getActiveHandlers called. printToConsole: {finalResolvedOptions.get(constants.ARG_PRINT_TO_CONSOLE)}, "
            f"resolvedFilePath: {resolvedFilePath}, finalEffMsgLevel: {finalEffectiveMessageLevel}")
        activeHandlers: List[logging.Handler] = []

        shouldPrintToConsole = finalResolvedOptions.get(constants.ARG_PRINT_TO_CONSOLE, False)
        self._instanceDebugPrint(f"Resolved shouldPrintToConsole: {shouldPrintToConsole}")

        # --- Console Handler ---
        if shouldPrintToConsole:
            # The lock is already held by DynamicLogger.log() when this is called,
            # but internal locking for self._consoleHandler access is still good practice
            # if this method could somehow be called outside that main lock.
            with self._lock:  # Using the shared lock
                self._instanceDebugPrint(
                    "Processing console handler (lock acquired by caller, re-entered by self._lock).")
                if self._consoleHandler is None:
                    self._instanceDebugPrint(
                        "Console handler is None, creating new StreamHandler(sys.stdout).")
                    self._consoleHandler = logging.StreamHandler(
                        sys.stdout)  # Default to current sys.stdout
                    self._instanceDebugPrint(
                        f"New console handler created: {self._consoleHandler}, initial stream: {getattr(self._consoleHandler, 'stream', 'N/A')}")
                else:
                    self._instanceDebugPrint(
                        f"Using existing console handler: {self._consoleHandler}. Current stream: {getattr(self._consoleHandler, 'stream', 'N/A')}. Target sys.stdout: {sys.stdout}")
                    # CRITICAL: Ensure the handler's stream is the *current* sys.stdout.
                    # This is vital for contexts like testing where sys.stdout might be temporarily redirected.
                    if self._consoleHandler.stream is not sys.stdout:
                        self._instanceDebugPrint(
                            f"Stream mismatch! Re-pointing console handler stream from '{getattr(self._consoleHandler.stream, '__name__', type(self._consoleHandler.stream))}' to current sys.stdout.")
                        try:
                            if hasattr(self._consoleHandler.stream, 'getvalue') and \
                                    self._consoleHandler.stream is not sys.__stdout__:
                                self._instanceDebugPrint(
                                    f"  Flushing previous buffered stream: {self._consoleHandler.stream}")
                                self._consoleHandler.flush()
                        except Exception as e_flush:  # pylint: disable=broad-except
                            self._instanceDebugPrint(
                                f"  Exception while flushing old stream: {e_flush}")
                            pass

                        new_stream_set_result = self._consoleHandler.setStream(
                            sys.stdout)
                        self._instanceDebugPrint(
                            f"Stream re-pointed. Handler stream is now: {getattr(self._consoleHandler, 'stream', 'N/A')}. setStream old stream: {new_stream_set_result}")
                    else:
                        self._instanceDebugPrint(
                            f"Console handler stream already matches current sys.stdout.")

                formatter = self._createFormatter()
                self._instanceDebugPrint(
                    f"Setting console handler formatter ({formatter}) and level ({finalEffectiveMessageLevel}).")
                self._consoleHandler.setFormatter(formatter)
                self._consoleHandler.setLevel(finalEffectiveMessageLevel)
                activeHandlers.append(self._consoleHandler)
                self._instanceDebugPrint(
                    f"Console handler configured and added. Current active handlers: {len(activeHandlers)}")
        else:
            self._instanceDebugPrint(
                "Skipping console handler based on resolved options (printToConsole=False).")

        # --- File Handler ---
        if resolvedFilePath:  # resolvedFilePath is an absolute Path object or None
            with self._lock:  # Using the shared lock
                self._instanceDebugPrint(
                    f"Processing file handler for path: '{resolvedFilePath}' (lock acquired by caller, re-entered by self._lock).")
                if resolvedFilePath not in self._fileHandlersCache:
                    self._instanceDebugPrint(
                        f"Path '{resolvedFilePath}' not in cache. Attempting to create FileHandler.")
                    try:
                        # Parent directories should have been created by PathResolver.
                        # DynamicLogger's 'w' mode is handled by deleting files/folders *before* this.
                        # So, FileHandler is ALWAYS opened in append mode ('a').
                        self._instanceDebugPrint(
                            f"Instantiating FileHandler for '{resolvedFilePath}' (mode 'a', encoding 'utf-8').")
                        fileHandler = logging.FileHandler(str(resolvedFilePath), mode='a',
                                                          encoding='utf-8')
                        self._instanceDebugPrint(
                            f"FileHandler for '{resolvedFilePath}' created successfully: {fileHandler}")
                        self._fileHandlersCache[resolvedFilePath] = fileHandler
                    except Exception as e_fh_create:
                        print(
                            f"ERROR_HandlerManager ({id(self)}): Error creating file handler for '{resolvedFilePath}': {e_fh_create}",
                            file=sys.stderr, flush=True)
                        self._instanceDebugPrint(f"  Failed to create FileHandler: {e_fh_create}")

                if resolvedFilePath in self._fileHandlersCache:
                    cachedFileHandler = self._fileHandlersCache[resolvedFilePath]
                    formatter = self._createFormatter()
                    self._instanceDebugPrint(
                        f"Setting file handler formatter ({formatter}) and level ({finalEffectiveMessageLevel}) for '{resolvedFilePath}'.")
                    cachedFileHandler.setFormatter(formatter)
                    cachedFileHandler.setLevel(finalEffectiveMessageLevel)
                    activeHandlers.append(cachedFileHandler)
                    self._instanceDebugPrint(
                        f"File handler for '{resolvedFilePath}' configured and added. Current active handlers: {len(activeHandlers)}")
                else:
                    self._instanceDebugPrint(
                        f"File handler for '{resolvedFilePath}' could not be obtained (e.g., creation failed).")
        else:
            self._instanceDebugPrint("Skipping file handler as resolvedFilePath is None.")

        self._instanceDebugPrint(
            f"Returning {len(activeHandlers)} active handlers: {activeHandlers}")
        return activeHandlers

    def shutdown(self) -> None:
        """
        Shuts down the HandlerManager: closes all cached file handlers.
        Console handler is typically not closed if it points to sys.stdout.
        """
        self._instanceDebugPrint("shutdown() called.")
        with self._lock:  # Ensure thread safety during shutdown
            self._instanceDebugPrint(
                f"Found {len(self._fileHandlersCache)} file handlers in cache to close.")
            for path, handler in list(self._fileHandlersCache.items()):
                self._instanceDebugPrint(f"Closing file handler for '{path}': {handler}")
                try:
                    handler.flush()
                    handler.close()
                    self._instanceDebugPrint(
                        f"  Handler for '{path}' successfully flushed and closed.")
                except Exception as e_close:
                    print(
                        f"ERROR_HandlerManager ({id(self)}): Error closing file handler for '{path}': {e_close}",
                        file=sys.stderr, flush=True)
                    self._instanceDebugPrint(
                        f"  Exception while closing handler for '{path}': {e_close}")
                if path in self._fileHandlersCache:
                    del self._fileHandlersCache[path]

            if self._consoleHandler:
                self._instanceDebugPrint(f"Flushing console handler: {self._consoleHandler}")
                try:
                    self._consoleHandler.flush()
                    self._instanceDebugPrint("  Console handler flushed.")
                except Exception as e_console_flush:
                    self._instanceDebugPrint(
                        f"  Exception while flushing console handler: {e_console_flush}")
                    pass

            self._instanceDebugPrint("HandlerManager shutdown complete.")
