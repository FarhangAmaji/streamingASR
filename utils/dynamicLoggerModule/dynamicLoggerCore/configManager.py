# dynamicLoggerCore/configManager.py
import threading
from typing import Dict, Any, Optional
import sys

from utils.dynamicLoggerModule.dyLogUtils import constants
from utils.dynamicLoggerModule.dyLogUtils import validation


class ConfigManager:
    def __init__(self,
                 initIndicatorName: Optional[str],
                 initDirectArgs: Dict[str, Any],
                 initConfgiSetArgs: Dict[str, Any],
                 highOrderOptions: Optional[Dict[str, Any]],
                 debugPrint: bool = False):  # Added debugPrint

        self._lock = threading.RLock()  # Used for thread-safe updates to _highOrderOptions
        self._debugPrintActive = debugPrint  # Store debugPrint state

        self._instanceDebugPrint(
            f"Initialized. InitIndicator: '{initIndicatorName}', InitDirect: {len(initDirectArgs)} keys, InitCS: {len(initConfgiSetArgs)} keys, HOs: {len(highOrderOptions or {})} keys.")

        self._instanceIndicatorName: Optional[str] = initIndicatorName
        self._initDirectArgs: Dict[str, Any] = initDirectArgs.copy()  # Shallow copy is fine
        self._initConfgiSetArgs: Dict[str, Any] = initConfgiSetArgs.copy()

        self._highOrderOptions: Dict[str, Any] = {}  # Holds the validated HO rules
        if highOrderOptions is not None:
            # Validate the structure of the provided highOrderOptions
            errors = validation.isValidHighOrderOptionsStructure(
                highOrderOptions,
                constants.ALL_MAIN_LOG_ARGS_KEYS
            )
            if errors:
                # Log a prominent error if HOs are invalid.
                # Depending on strictness, might raise an error or proceed with empty HOs.
                # Current behavior: print error and use empty HOs if initial ones are invalid.
                error_message = f"ERROR_ConfigManager ({id(self)}): Invalid highOrderOptions provided during initialization: {errors}. Using empty HOs."
                print(error_message, file=sys.stderr, flush=True)
                self._instanceDebugPrint(error_message)
                self._highOrderOptions = {}  # Default to empty if validation fails
            else:
                self._highOrderOptions = highOrderOptions.copy()  # Store a copy
                self._instanceDebugPrint(
                    f"  Successfully validated and stored {len(self._highOrderOptions)} HO rules.")
        else:
            self._highOrderOptions = {}  # No HOs provided
            self._instanceDebugPrint("  No initial highOrderOptions provided.")

        # Store copies of global defaults to prevent modification of constants
        # and ensure instance-specific copies if mutable defaults were used (e.g. for 'extra').
        self._finalDefaults: Dict[str, Any] = constants.FINAL_DEFAULTS.copy()
        if constants.ARG_EXTRA in self._finalDefaults and \
                isinstance(self._finalDefaults[constants.ARG_EXTRA], dict):
            # Ensure 'extra' in finalDefaults is a distinct copy if it's a dict
            self._finalDefaults[constants.ARG_EXTRA] = self._finalDefaults[
                constants.ARG_EXTRA].copy()

        self._onLevelDefaults: Dict[str, Any] = constants.ON_LEVEL_DEFAULTS.copy()
        self._instanceDebugPrint(
            f"  Copied FINAL_DEFAULTS ({len(self._finalDefaults)} keys) and ON_LEVEL_DEFAULTS ({len(self._onLevelDefaults)} keys).")

    def _instanceDebugPrint(self, message: str):
        """Prints debug messages for this ConfigManager instance if debugPrint is enabled."""
        if self._debugPrintActive:
            print(f"DEBUG_ConfigManager ({id(self)}): {message}", file=sys.stderr, flush=True)

    @property
    def instanceIndicatorName(self) -> Optional[str]:
        """The default indicatorName for the DynamicLogger instance, set at initialization."""
        return self._instanceIndicatorName

    @property
    def initDirectArgs(self) -> Dict[str, Any]:
        """A copy of direct arguments passed to DynamicLogger.__init__."""
        return self._initDirectArgs.copy()

    @property
    def initConfgiSetArgs(self) -> Dict[str, Any]:
        """A copy of arguments from the configSet passed to DynamicLogger.__init__."""
        return self._initConfgiSetArgs.copy()

    @property
    def finalDefaults(self) -> Dict[str, Any]:
        """
        A copy of the ultimate fallback default values for all main log arguments.
        Ensures that mutable defaults like `extra: {}` are new copies.
        """
        defaultsCopy = self._finalDefaults.copy()
        # Ensure 'extra' is a new dictionary if it's a dict type in defaults
        if constants.ARG_EXTRA in defaultsCopy and isinstance(defaultsCopy[constants.ARG_EXTRA],
                                                              dict):
            defaultsCopy[constants.ARG_EXTRA] = defaultsCopy[constants.ARG_EXTRA].copy()
        return defaultsCopy

    @property
    def onLevelDefaults(self) -> Dict[str, Any]:
        """
        A copy of the 'onLevelDefault' values. These signify "no opinion" or "passthrough"
        at a specific configuration level in the priority chain.
        """
        return self._onLevelDefaults.copy()

    def getHighOrderOptions(self) -> Dict[str, Any]:
        """
        Returns a (shallow) copy of the current highOrderOptions dictionary.
        Thread-safe read.
        """
        with self._lock:
            self._instanceDebugPrint(
                f"getHighOrderOptions called. Returning copy of {len(self._highOrderOptions)} rules.")
            return self._highOrderOptions.copy()

    def updateHighOrderOptions(self, newHighOrderOptions: Dict[str, Any]) -> None:
        """
        Updates the highOrderOptions with a new set of rules.
        The new rules are validated before being applied.
        Thread-safe update.
        """
        self._instanceDebugPrint(
            f"updateHighOrderOptions called with {len(newHighOrderOptions or {})} new rules.")
        if not isinstance(newHighOrderOptions, dict):
            warning_msg = f"Warning_ConfigManager ({id(self)}): updateHighOrderOptions received non-dict input ({type(newHighOrderOptions)}). No update performed."
            print(warning_msg, file=sys.stderr, flush=True)
            self._instanceDebugPrint(warning_msg)
            return

        # Validate the new HO structure
        errors = validation.isValidHighOrderOptionsStructure(
            newHighOrderOptions,
            constants.ALL_MAIN_LOG_ARGS_KEYS
        )
        if errors:
            error_message = f"ERROR_ConfigManager ({id(self)}): Invalid newHighOrderOptions provided during update: {errors}. No update performed."
            print(error_message, file=sys.stderr, flush=True)
            self._instanceDebugPrint(error_message)
            return

        with self._lock:
            self._highOrderOptions = newHighOrderOptions.copy()  # Store a copy
            self._instanceDebugPrint(
                f"  Successfully validated and updated highOrderOptions to {len(self._highOrderOptions)} rules.")

    def getHoRuleBody(self, hoKey: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the rule body (a dictionary of settings) for a given High-Order Option key
        (which can be an indicatorName or a funcMethIndicator).
        Returns None if the key is not found or if hoKey is empty/None.
        Thread-safe read.
        """
        if not hoKey:  # An empty key cannot match anything.
            self._instanceDebugPrint(f"getHoRuleBody called with empty hoKey. Returning None.")
            return None

        with self._lock:
            ruleBody = self._highOrderOptions.get(hoKey)
            # self._instanceDebugPrint(f"getHoRuleBody for key '{hoKey}': Found ruleBody: {ruleBody is not None}.") # Can be verbose
            # Rule body itself is a dict. A copy is not strictly necessary here for read if
            # the caller doesn't modify it, but ArgumentProcessor will extract from it.
            # If ruleBody is returned and modified by ArgumentProcessor, it could affect stored HOs if not copied.
            # However, ArgumentProcessor just reads from it to build its sources list.
            # For safety, returning a copy if found.
            return ruleBody.copy() if ruleBody is not None else None
