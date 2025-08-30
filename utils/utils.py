# utils.py
# ==============================================================================
# General Utility Functions
# ==============================================================================
#
# Purpose:
# - This module contains miscellaneous helper functions used across the application.
# - Currently, it provides a utility for converting Windows file paths to their
#   WSL (Windows Subsystem for Linux) equivalents.
# ==============================================================================
import sys
from pathlib import Path

# Import the new universal logger instances for logging messages
from utils.loggerInstance import uniLogger, uniDebugLogger


def convertWindowsPathToWsl(windowsPath) -> str | None:
    """
    Converts a Windows path (absolute path or Path object) to its WSL equivalent.
    Handles standard drive letters (e.g., C:\) and basic UNC paths (e.g., \\server\share).
    Returns the converted WSL path as a string, or None on failure.
    """
    try:
        # Use pathlib for robust path handling
        windowsPathConverted = Path(windowsPath)

        # If the provided path is relative, resolve it to an absolute path first.
        if not windowsPathConverted.is_absolute():
            resolvedPath = windowsPathConverted.resolve()
            uniLogger.warning(
                f"Path '{windowsPathConverted}' is not absolute. Attempting resolution to '{resolvedPath}' for WSL conversion."
            )
            windowsPathConverted = resolvedPath
            # If resolution fails, the path cannot be converted.
            if not windowsPathConverted.is_absolute():
                uniLogger.error(
                    "Cannot convert relative path to WSL path after resolution attempt.")
                return None

        pathStr = str(windowsPathConverted)
        uniDebugLogger.debug(f"Attempting WSL path conversion for: {pathStr}")

        # --- Handle standard drive paths (e.g., "C:\Users\...") ---
        if len(pathStr) >= 2 and pathStr[1] == ':':
            driveLetter = pathStr[0].lower()
            # Replace backslashes with forward slashes for Linux compatibility
            restOfPath = pathStr[2:].replace('\\', '/')
            # Construct the WSL path under the /mnt/ directory
            wslPath = f"/mnt/{driveLetter}{restOfPath}"
            uniDebugLogger.debug(f"Converted Windows drive path to WSL path: '{wslPath}'")
            return wslPath
        # --- Handle basic UNC paths (e.g., "\\server\share") ---
        elif pathStr.startswith('\\\\'):
            uniLogger.warning(
                "Attempting basic UNC path conversion for WSL - might not work depending on WSL mount configuration."
            )
            # A simple replacement might work if the UNC path is accessible from WSL
            wslPath = pathStr.replace('\\', '/')
            uniDebugLogger.debug(f"Converted Windows UNC path to potential WSL path: '{wslPath}'")
            return wslPath
        # --- Handle unrecognized formats ---
        else:
            uniLogger.error(f"Unrecognized Windows path format for WSL conversion: {pathStr}")
            return None
    except Exception as e:
        # Catch any unexpected errors during the conversion process
        uniLogger.error(f"Error during Windows path conversion to WSL: {e}", excInfo=True)
        return None