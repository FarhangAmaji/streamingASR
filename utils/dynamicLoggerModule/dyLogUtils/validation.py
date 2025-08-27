# dyLogUtils/validation.py
import logging
import re
from typing import Any

from utils.dynamicLoggerModule.dyLogUtils import constants  # Assuming constants.py is in the same directory or accessible
from utils.dynamicLoggerModule.dyLogUtils.projectTypes import FileWriteModeType  # NEW


# --- Validation Functions ---

def isValidLogFormatString(formatStr: Any) -> bool:
    return isinstance(formatStr, str) and bool(formatStr)


def isValidTimestampFormatString(formatStr: Any) -> bool:
    return isinstance(formatStr, str) and bool(formatStr)


def isValidLogLevel(level: Any) -> bool:
    if not isinstance(level, int):
        return False
    return level in [
        logging.CRITICAL, logging.ERROR, logging.WARNING,
        logging.INFO, logging.DEBUG, logging.NOTSET
    ] or level > logging.NOTSET


def isNonEmptyString(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def isValidPythonIdentifier(name: str) -> bool:
    return isinstance(name, str) and name.isidentifier()


def isValidFileWriteMode(mode: Any) -> bool:  # NEW
    return isinstance(mode, str) and mode in ['w', 'a']


def isValidConfigSetDict(csDict: Any, allMainLogArgsKeys: frozenset) -> list[str]:
    # from .constants import ARG_CONFIG_SET # Not needed here, ARG_CONFIG_SET is a string from constants

    if not isinstance(csDict, dict):
        return ["<NotADictionaryObject>"]

    unknownKeys = []
    # configSet itself cannot be a key within a configSet dictionary.
    if constants.ARG_CONFIG_SET in csDict:
        # This specific string is checked in isValidHighOrderOptionsStructure
        unknownKeys.append(
            f"'{constants.ARG_CONFIG_SET}' key not allowed within a configSet dictionary itself.")

    for key in csDict.keys():
        if key not in allMainLogArgsKeys:  # Check against all possible log args
            unknownKeys.append(key)
    return unknownKeys


def isValidHighOrderOptionsStructure(hoOptions: Any, allMainLogArgsKeys: frozenset) -> list[str]:
    """
    Validates the NEW structure of the highOrderOptions dictionary.
    An HO Rule Body is a dict containing direct args (any AllMainLogArgs)
    and an optional 'configSet' key.
    """
    errorMessages = []
    if not isinstance(hoOptions, dict):
        errorMessages.append("highOrderOptions must be a dictionary.")
        return errorMessages

    for hoKey, hoRuleBody in hoOptions.items():  # hoKey is indicatorName or funcMethIndicator
        if not isinstance(hoKey, str) or not hoKey:
            errorMessages.append(f"Invalid HO key: '{hoKey}'. Must be a non-empty string.")
        if not isinstance(hoRuleBody, dict):
            errorMessages.append(f"Rule body for HO key '{hoKey}' must be a dictionary.")
            continue  # Skip further checks for this malformed rule body

        # Check keys within the hoRuleBody
        for ruleArgKey, ruleArgValue in hoRuleBody.items():
            if ruleArgKey == constants.ARG_CONFIG_SET:
                # This is the special 'configSet' key within an HO rule body
                if not isinstance(ruleArgValue, dict):
                    errorMessages.append(
                        f"Value for '{constants.ARG_CONFIG_SET}' in HO rule '{hoKey}' must be a dictionary.")
                else:
                    # Validate the embedded configSet
                    configSetErrors = isValidConfigSetDict(ruleArgValue, allMainLogArgsKeys)
                    for csError in configSetErrors:
                        if csError.startswith("<"):  # Special marker for type mismatch
                            errorMessages.append(
                                f"Embedded '{constants.ARG_CONFIG_SET}' for HO '{hoKey}' is not a dict.")
                        elif csError.startswith(
                                f"'{constants.ARG_CONFIG_SET}' key not allowed"):  # Nested configSet check
                            errorMessages.append(
                                f"'{constants.ARG_CONFIG_SET}' key not allowed within an embedded configSet for HO '{hoKey}'.")
                        else:  # Other unknown keys
                            errorMessages.append(
                                f"Unknown key '{csError}' in embedded configSet for HO '{hoKey}'.")
                    # Check if the embedded configSet tries to set indicatorName
                    if constants.ARG_INDICATOR_NAME in ruleArgValue:
                        errorMessages.append(
                            f"'{constants.ARG_INDICATOR_NAME}' cannot be set by an embedded configSet in HO rule '{hoKey}'.")

            elif ruleArgKey == constants.ARG_INDICATOR_NAME:
                errorMessages.append(
                    f"'{constants.ARG_INDICATOR_NAME}' cannot be set directly as a direct argument in HO rule '{hoKey}'. HOs are triggered by indicators, they don't set them.")
            elif ruleArgKey == constants.ARG_SIMPLE_LOG:
                errorMessages.append(
                    f"'{constants.ARG_SIMPLE_LOG}' cannot be set by an HO rule ('{hoKey}'). It is only effective inline direct.")
            elif ruleArgKey not in allMainLogArgsKeys:
                errorMessages.append(
                    f"Unknown argument key '{ruleArgKey}' in HO rule body for '{hoKey}'.")
            # No specific validation for values of direct args here; that happens during argument resolution.
            # This function primarily checks structure and valid keys.

    return errorMessages


_INVALID_FILENAME_CHARS_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_NAMES = frozenset([
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
])


def isValidFilePathSegment(segment: str) -> bool:
    if not isinstance(segment, str) or not segment or segment == "." or segment == "..":
        return False
    if _INVALID_FILENAME_CHARS_PATTERN.search(segment):
        return False
    if segment.upper() in _WINDOWS_RESERVED_NAMES:  # For Windows
        return False
    return True
