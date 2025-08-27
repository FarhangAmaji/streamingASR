# dynamicLoggerCore/argumentProcessor.py
from typing import Dict, Any, List, Optional
import sys  # For _instanceDebugPrint

from utils.dynamicLoggerModule.dyLogUtils import constants
from utils.dynamicLoggerModule.dyLogUtils import validation


# from utils.dynamicLoggerCore.configManager import ConfigManager # Type hint only, avoid circular import for runtime

# ARGUMENT_PROCESSOR_DEBUG_MODE = True # Replaced by instance-level debugPrint

# def _debugPrint(message: str): # Replaced by instance method _instanceDebugPrint
#     if ARGUMENT_PROCESSOR_DEBUG_MODE:
#         print(f"DEBUG_ArgProcessor: {message}", file=__import__('sys').stderr, flush=True)


class ArgumentProcessor:
    def __init__(self, configManager, debugPrint: bool = False):  # Added debugPrint
        self._configManager = configManager
        self._debugPrintActive = debugPrint  # Store debugPrint state
        self._instanceDebugPrint(f"Initialized. ConfigManager ID: {id(configManager)}")

    def _instanceDebugPrint(self, message: str):
        """Prints debug messages for this ArgumentProcessor instance if debugPrint is enabled."""
        if self._debugPrintActive:
            # Use sys.stderr directly to avoid potential issues with redirected stdout during tests
            print(f"DEBUG_ArgProcessor ({id(self)}): {message}", file=sys.stderr, flush=True)

    def _mergeExtraDicts(self, prioritySourcesList: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merges 'extra' dictionaries from all sources.
        Sources with higher priority (earlier in the list) override keys from lower priority ones.
        Starts with the finalDefault for 'extra'.
        """
        self._instanceDebugPrint(f"Merging 'extra' dicts from {len(prioritySourcesList)} sources.")
        finalDefaults = self._configManager.finalDefaults
        # Start with a copy of the default 'extra' dictionary
        mergedExtra = finalDefaults.get(constants.ARG_EXTRA, {}).copy()
        self._instanceDebugPrint(f"  Initial mergedExtra from finalDefaults: {mergedExtra}")

        # Iterate `prioritySourcesList` from lowest to highest to build up the dict.
        # The `prioritySourcesList` is from Highest (index 0) to Lowest.
        # So, to merge with higher overriding lower:
        #   Initialize with lowest.
        #   Then update with next lowest, etc., up to highest.
        # This means iterating `prioritySourcesList` in REVERSE.

        tempMergedExtra = {}  # Start with an empty dict
        sourcesInReverseOrder = reversed(prioritySourcesList)  # Lowest priority first

        sourceIdx = len(prioritySourcesList) - 1
        for source in sourcesInReverseOrder:
            if constants.ARG_EXTRA in source:
                extraFromSource = source[constants.ARG_EXTRA]
                if isinstance(extraFromSource, dict):
                    self._instanceDebugPrint(
                        f"  Merging from source (idx {sourceIdx}, reversed): {extraFromSource}")
                    tempMergedExtra.update(
                        extraFromSource)  # Higher priority (later in this loop) overrides
                else:
                    self._instanceDebugPrint(
                        f"  WARN: '{constants.ARG_EXTRA}' in source (idx {sourceIdx}, reversed) is not a dict: {extraFromSource}. Skipping.")
            sourceIdx -= 1

        # Finally, merge with the `finalDefaults` 'extra', ensuring `tempMergedExtra` (from actual sources) overrides.
        finalMergedExtra = finalDefaults.get(constants.ARG_EXTRA, {}).copy()
        finalMergedExtra.update(tempMergedExtra)

        self._instanceDebugPrint(f"  Final merged 'extra': {finalMergedExtra}")
        return finalMergedExtra

    def resolveEffectiveArgs(self,
                             initialMessageLevel: int,
                             inlineDirectArgs: Dict[str, Any],
                             # This comes from DynamicLogger.log's collected kwargs (L5)
                             inlineConfigSetFromKwargs: Optional[Dict[str, Any]],
                             # This is the configSet from log()'s configSet param (L6)
                             effectiveIndicatorNameForHoLookup: Optional[str],
                             funcMethIndicatorForHoLookup: Optional[str],
                             isSimpleLogMode: bool
                             ) -> Dict[str, Any]:
        self._instanceDebugPrint(f"--- Resolving Effective Args ---")
        self._instanceDebugPrint(
            f"InitialMsgLevel: {initialMessageLevel}, SimpleMode: {isSimpleLogMode}")
        self._instanceDebugPrint(f"InlineDirectArgs (L5 source): {inlineDirectArgs}")
        self._instanceDebugPrint(f"InlineConfigSet (L6 source): {inlineConfigSetFromKwargs}")
        self._instanceDebugPrint(
            f"HO IndicatorName: {effectiveIndicatorNameForHoLookup}, HO funcMethIndicator: {funcMethIndicatorForHoLookup}")

        finalResolvedOptions: Dict[str, Any] = {}
        prioritySourcesList: List[Dict[str, Any]] = []  # Highest priority at index 0

        # --- Build Priority Sources List ---
        # The order of appending to this list defines the priority (Highest to Lowest).
        self._instanceDebugPrint("Building prioritySourcesList...")
        if not isSimpleLogMode:
            self._instanceDebugPrint("Full mode: Adding HOs to prioritySourcesList.")
            # Level 1: HO triggered by indicatorName (Direct args from the HO rule body)
            # Level 2: HO triggered by indicatorName (Args from a configSet key within that HO rule body)
            if effectiveIndicatorNameForHoLookup:
                hoRuleBody = self._configManager.getHoRuleBody(effectiveIndicatorNameForHoLookup)
                self._instanceDebugPrint(
                    f"HO Rule for Indicator '{effectiveIndicatorNameForHoLookup}': {hoRuleBody}")
                if hoRuleBody and isinstance(hoRuleBody, dict):
                    # Extract direct args (all keys except ARG_CONFIG_SET)
                    directHoArgs = {k: v for k, v in hoRuleBody.items() if
                                    k != constants.ARG_CONFIG_SET}
                    if directHoArgs:  # L1
                        prioritySourcesList.append(directHoArgs)
                        self._instanceDebugPrint(
                            f"  Added HO Indicator Direct (L1): {directHoArgs}")
                    # Extract configSet if present
                    if constants.ARG_CONFIG_SET in hoRuleBody and isinstance(
                            hoRuleBody[constants.ARG_CONFIG_SET], dict):  # L2
                        prioritySourcesList.append(hoRuleBody[constants.ARG_CONFIG_SET])
                        self._instanceDebugPrint(
                            f"  Added HO Indicator ConfigSet (L2): {hoRuleBody[constants.ARG_CONFIG_SET]}")

            # Level 3: HO triggered by funcMethIndicator (Direct args)
            # Level 4: HO triggered by funcMethIndicator (Args from configSet)
            if funcMethIndicatorForHoLookup:
                hoRuleBody = self._configManager.getHoRuleBody(funcMethIndicatorForHoLookup)
                self._instanceDebugPrint(
                    f"HO Rule for funcMeth '{funcMethIndicatorForHoLookup}': {hoRuleBody}")
                if hoRuleBody and isinstance(hoRuleBody, dict):
                    directHoArgs = {k: v for k, v in hoRuleBody.items() if
                                    k != constants.ARG_CONFIG_SET}
                    if directHoArgs:  # L3
                        prioritySourcesList.append(directHoArgs)
                        self._instanceDebugPrint(f"  Added HO funcMeth Direct (L3): {directHoArgs}")
                    if constants.ARG_CONFIG_SET in hoRuleBody and isinstance(
                            hoRuleBody[constants.ARG_CONFIG_SET], dict):  # L4
                        prioritySourcesList.append(hoRuleBody[constants.ARG_CONFIG_SET])
                        self._instanceDebugPrint(
                            f"  Added HO funcMeth ConfigSet (L4): {hoRuleBody[constants.ARG_CONFIG_SET]}")
        else:
            self._instanceDebugPrint("Simple mode: Skipping HOs for prioritySourcesList.")

        # Level 5: Inline Arguments (Directly passed as named parameters to All Log Funcs)
        if inlineDirectArgs:  # This is a dict of args like {printToConsole: True, ...}
            prioritySourcesList.append(inlineDirectArgs)
            self._instanceDebugPrint(f"  Added Inline Direct (L5): {inlineDirectArgs}")

        # Level 6: Inline Arguments (Args from a configSet dictionary passed inline to All Log Funcs)
        if inlineConfigSetFromKwargs:  # This is a dict from log(configSet={...})
            prioritySourcesList.append(inlineConfigSetFromKwargs)
            self._instanceDebugPrint(f"  Added Inline ConfigSet (L6): {inlineConfigSetFromKwargs}")

        # Initialization sources are only considered if not in simpleLog mode
        if not isSimpleLogMode:
            # Level 7: DynamicLogger Initialization (Directly passed as named parameters to __init__)
            initDirect = self._configManager.initDirectArgs
            if initDirect:
                prioritySourcesList.append(initDirect)
                self._instanceDebugPrint(f"  Added Init Direct (L7): {initDirect}")

            # Level 8: DynamicLogger Initialization (Args from a configSet dictionary passed to __init__)
            initCS = self._configManager.initConfgiSetArgs
            if initCS:
                prioritySourcesList.append(initCS)
                self._instanceDebugPrint(f"  Added Init ConfigSet (L8): {initCS}")

        self._instanceDebugPrint(
            f"Full prioritySourcesList (len {len(prioritySourcesList)}), highest to lowest prio: {prioritySourcesList}")

        # --- Resolve Each Argument ---
        finalDefaults = self._configManager.finalDefaults
        onLevelDefaults = self._configManager.onLevelDefaults
        self._instanceDebugPrint(f"FinalDefaults to be used: {finalDefaults}")
        self._instanceDebugPrint(f"OnLevelDefaults for passthrough check: {onLevelDefaults}")

        for argKey in constants.ALL_MAIN_LOG_ARGS_KEYS:
            self._instanceDebugPrint(f"Resolving arg: '{argKey}'")

            # ARG_CONFIG_SET is a container, not a resolvable value itself.
            if argKey == constants.ARG_CONFIG_SET:
                self._instanceDebugPrint(f"  Skipping '{argKey}' (not a resolvable value).")
                continue

            # ARG_SIMPLE_LOG's role in mode determination is handled before this function.
            # Here, we just ensure its resolved value reflects the mode.
            if argKey == constants.ARG_SIMPLE_LOG:
                finalResolvedOptions[argKey] = isSimpleLogMode
                self._instanceDebugPrint(
                    f"  Set '{argKey}' to pre-determined simpleLogMode: {isSimpleLogMode}.")
                continue

            # ARG_EXTRA is handled separately with a merge strategy.
            if argKey == constants.ARG_EXTRA:
                self._instanceDebugPrint(f"  Skipping '{argKey}' for now (special merge later).")
                continue

            resolvedValue = None
            foundActiveSetting = False

            sourceIndex = 0
            for sourceDict in prioritySourcesList:  # Iterates from highest priority to lowest
                # self._instanceDebugPrint(f"  Checking source {sourceIndex}: {sourceDict}") # Can be very verbose
                if argKey in sourceDict:
                    valueFromSource = sourceDict[argKey]
                    onLevelDefaultForArg = onLevelDefaults.get(
                        argKey)  # Get the "passthrough" sentinel

                    # An active setting is one that is present in the source
                    # AND is not the 'onLevelDefault' value for that argument.
                    # If valueFromSource IS onLevelDefault, it's an explicit passthrough.
                    isActive = False
                    if valueFromSource != onLevelDefaultForArg:
                        isActive = True
                        # self._instanceDebugPrint(f"    Value '{valueFromSource}' != OnLevelDefault '{onLevelDefaultForArg}'. Active setting.")
                    # else:
                    # self._instanceDebugPrint(f"    Value '{valueFromSource}' IS OnLevelDefault. Passing through.")

                    if isActive:
                        resolvedValue = valueFromSource
                        foundActiveSetting = True
                        self._instanceDebugPrint(
                            f"    '{argKey}' resolved to '{resolvedValue}' from source index {sourceIndex}. Breaking search for this arg.")
                        break  # Found in current source, move to next argKey
                sourceIndex += 1

            if foundActiveSetting:
                finalResolvedOptions[argKey] = resolvedValue
                self._instanceDebugPrint(
                    f"  Final value for '{argKey}': {resolvedValue} (from active setting).")
            else:
                # If no active setting found in any source, use the finalDefault.
                # For 'extra', finalDefault is `{}`, a new empty dict.
                # For other args, it's their ultimate fallback value.
                defaultVal = finalDefaults.get(argKey)
                # Handle callable defaults like `dict.copy` or `lambda: {}` if any were used for mutable defaults
                # Currently, FINAL_DEFAULTS['extra'] is just `{}`, so simple copy works.
                # If it were a function, `defaultVal()` would be called.
                # For ARG_EXTRA, default is `{}`. Ensure a new dict if copied.
                finalValueFromDefault = defaultVal() if callable(
                    defaultVal) and argKey == constants.ARG_EXTRA else (
                    defaultVal.copy() if isinstance(defaultVal, dict) else defaultVal)

                finalResolvedOptions[argKey] = finalValueFromDefault
                self._instanceDebugPrint(
                    f"  Final value for '{argKey}': {finalValueFromDefault} (from final default).")

        # Handle ARG_EXTRA merging
        finalResolvedOptions[constants.ARG_EXTRA] = self._mergeExtraDicts(prioritySourcesList)
        self._instanceDebugPrint(f"  Merged ARG_EXTRA: {finalResolvedOptions[constants.ARG_EXTRA]}")

        # --- Post-Processing and Validation of Resolved Values ---
        # IndicatorName for formatting (used in %(indicatorName)s)
        fmtIndicatorName = finalResolvedOptions.get(constants.ARG_INDICATOR_NAME)
        if fmtIndicatorName is None:
            fmtIndicatorName = self._configManager.instanceIndicatorName

        if not validation.isNonEmptyString(fmtIndicatorName):
            fmtIndicatorName = None
        finalResolvedOptions[
            constants.ARG_INDICATOR_NAME] = fmtIndicatorName
        self._instanceDebugPrint(
            f"Final indicatorName for formatting (%(indicatorName)s): {fmtIndicatorName}")

        # LogLevel override
        resolvedLogLevel = finalResolvedOptions.get(constants.ARG_LOG_LEVEL)
        if resolvedLogLevel is not None and not validation.isValidLogLevel(resolvedLogLevel):
            print(
                f"Warning_ArgProcessor ({id(self)}): Invalid logLevel value '{resolvedLogLevel}' resolved. Ignoring override.",
                file=sys.stderr, flush=True)
            resolvedLogLevel = None
        finalResolvedOptions[constants.ARG_LOG_LEVEL] = resolvedLogLevel

        # Determine final effective message level for the LogRecord
        finalEffectiveMessageLevel = resolvedLogLevel if resolvedLogLevel is not None else initialMessageLevel
        finalResolvedOptions[
            '_finalEffectiveMessageLevel'] = finalEffectiveMessageLevel
        self._instanceDebugPrint(
            f"Final effective message level: {finalEffectiveMessageLevel} (initial: {initialMessageLevel}, resolved override: {resolvedLogLevel})")

        # Validate logFormat type
        logFormatArg = finalResolvedOptions.get(constants.ARG_LOG_FORMAT)
        if not (isinstance(logFormatArg,
                           (str, bool)) or logFormatArg is None):
            print(
                f"Warning_ArgProcessor ({id(self)}): Invalid type for logFormat '{logFormatArg}' (type: {type(logFormatArg)}). Using final default.",
                file=sys.stderr, flush=True)
            finalResolvedOptions[constants.ARG_LOG_FORMAT] = finalDefaults[constants.ARG_LOG_FORMAT]

        # Validate timestampFormat type
        tsFormatArg = finalResolvedOptions.get(constants.ARG_TIMESTAMP_FORMAT)
        if not (isinstance(tsFormatArg, (str, bool)) or tsFormatArg is None):
            print(
                f"Warning_ArgProcessor ({id(self)}): Invalid type for timestampFormat '{tsFormatArg}' (type: {type(tsFormatArg)}). Using final default.",
                file=sys.stderr, flush=True)
            finalResolvedOptions[constants.ARG_TIMESTAMP_FORMAT] = finalDefaults[
                constants.ARG_TIMESTAMP_FORMAT]

        # Ensure messageArgs is a tuple
        msgArgs = finalResolvedOptions.get(constants.ARG_MESSAGE_ARGS)
        if msgArgs is not None and not isinstance(msgArgs, tuple):
            finalResolvedOptions[constants.ARG_MESSAGE_ARGS] = tuple(msgArgs) if isinstance(msgArgs,
                                                                                            list) else (
                msgArgs,)
        elif msgArgs is None:
            finalResolvedOptions[constants.ARG_MESSAGE_ARGS] = finalDefaults[
                constants.ARG_MESSAGE_ARGS]

        # Validate fileWriteMode (NEW)
        fileWriteModeArg = finalResolvedOptions.get(constants.ARG_FILE_WRITE_MODE)
        if not validation.isValidFileWriteMode(fileWriteModeArg):
            print(
                f"Warning_ArgProcessor ({id(self)}): Invalid fileWriteMode value '{fileWriteModeArg}' (type: {type(fileWriteModeArg)}). Using final default '{finalDefaults[constants.ARG_FILE_WRITE_MODE]}'.",
                file=sys.stderr, flush=True)
            finalResolvedOptions[constants.ARG_FILE_WRITE_MODE] = finalDefaults[
                constants.ARG_FILE_WRITE_MODE]

        self._instanceDebugPrint(
            f"--- Finished Resolving Effective Args. Final Options: {finalResolvedOptions} ---")
        return finalResolvedOptions
