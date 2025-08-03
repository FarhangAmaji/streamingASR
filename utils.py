# utils.py
import platform
from pathlib import Path


def convertWindowsPathToWsl(windowsPath):
    """Converts an absolute Windows path to its WSL equivalent (/mnt/...)."""
    pathStr = str(
        Path(windowsPath).resolve())  # Ensure absolute path, convert Path object to string
    if platform.system() != "Windows":
        # If not on Windows, assume it's already a Linux-style path
        return pathStr

    if ':' not in pathStr:
        # Cannot reliably convert relative paths
        logWarning(f"Cannot convert potentially relative Windows path to WSL: {pathStr}")
        return pathStr

    drive, tail = pathStr.split(':', 1)
    driveLower = drive.lower()
    # Convert backslashes and remove leading slash if present from split
    tail = tail.replace('\\', '/').lstrip('/')
    wslPath = f"/mnt/{driveLower}/{tail}"
    return wslPath


# ==================================
# Helper Functions & Configuration
# ==================================
def logDebug(message, debugPrintFlag):
    """Helper function for conditional debug printing."""
    if debugPrintFlag:
        print(f"DEBUG: {message}")


def logInfo(message):
    """Helper function for standard info messages."""
    print(f"INFO: {message}")


def logWarning(message):
    """Helper function for warning messages."""
    print(f"WARNING: {message}")


def logError(message):
    """Helper function for error messages."""
    print(f"ERROR: {message}")
