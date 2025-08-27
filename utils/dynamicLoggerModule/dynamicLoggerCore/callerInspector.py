# dynamicLoggerCore/callerInspector.py
import inspect
import re
from typing import Optional, List, Callable, Dict, Any
import sys

from utils.dynamicLoggerModule.dyLogUtils import constants

# from utils.dynamicLoggerModule.dyLogUtils import validation # Not strictly needed by this version of _sanitizeSegment

# Pattern to sanitize funcMethIndicator parts (class/method/function names)
# Allows alphanumeric and underscore, replaces others with underscore.
_SANITIZE_PATTERN = re.compile(r'[^a-zA-Z0-9_]')
# Pattern to remove leading/trailing underscores that might result from sanitization.
_LEADING_TRAILING_UNDERSCORE_PATTERN = re.compile(r'^_+|_+$')


class CallerInspector:
    def __init__(self,
                 wrappersToSkip: Optional[List[Callable[..., Any]]] = None,
                 debugPrint: bool = False):  # Added debugPrint
        self._debugPrintActive = debugPrint  # Store debugPrint state
        self._wrappersToSkipCodes = set()
        if wrappersToSkip:
            for wrapper in wrappersToSkip:
                if hasattr(wrapper, '__code__'):
                    self._wrappersToSkipCodes.add(wrapper.__code__)
        # Determine the filename of the current module (callerInspector.py) to skip its frames.
        try:
            # inspect.getfile(inspect.currentframe()) should give the path to this file (callerInspector.py)
            self._callerInspectorModuleFilename = inspect.getfile(inspect.currentframe())
            self._instanceDebugPrint(
                f"Initialized. Wrappers to skip codes: {len(self._wrappersToSkipCodes)}. Self filename: {self._callerInspectorModuleFilename}")
        except TypeError:
            # This can happen if run in an environment where __file__ is not available (e.g. frozen)
            # Fallback to a less precise check or handle as an error.
            self._callerInspectorModuleFilename = "callerInspector.py"  # A common enough name
            self._instanceDebugPrint(
                f"Initialized (fallback filename). Wrappers: {len(self._wrappersToSkipCodes)}.")
            print(
                f"Warning_CallerInspector ({id(self)}): Could not reliably determine module filename. Using fallback.",
                file=sys.stderr, flush=True)

    def _instanceDebugPrint(self, message: str):
        """Prints debug messages for this CallerInspector instance if debugPrint is enabled."""
        if self._debugPrintActive:
            print(f"DEBUG_CallerInspector ({id(self)}): {message}", file=sys.stderr, flush=True)

    def _sanitizeSegment(self, segment: str) -> str:
        """
        Sanitizes a string segment (like a class or function name) for use in
        funcMethIndicator or default file naming.
        Replaces invalid characters with underscores and removes leading/trailing underscores.
        """
        if not segment:
            # This should ideally not happen for function/class names from inspect.
            return "empty_segment"
        # Replace any character not alphanumeric or underscore with an underscore
        sanitized = _SANITIZE_PATTERN.sub('_', segment)
        # Remove any leading or trailing underscores that might have been created
        sanitized = _LEADING_TRAILING_UNDERSCORE_PATTERN.sub('', sanitized)
        # If sanitization results in an empty string (e.g., "____" became ""), return a placeholder.
        return sanitized if sanitized else "sanitized_to_empty"

    def determineCallerDetails(self) -> Dict[str, Any]:
        """
        Determines details about the caller of a DynamicLogger logging method.
        Walks the stack to find the first frame outside DynamicLogger's own core modules
        and any specified wrapper functions.
        Returns a dictionary with keys like "funcMethIndicator", "className", "funcName",
        "pathname", "lineno", "moduleName".
        """
        self._instanceDebugPrint("Determining caller details...")
        defaultDetails = {
            "funcMethIndicator": constants.PLACEHOLDER_NA,
            # e.g., ClassName.methodName or functionName (sanitized)
            "className": None,  # e.g., ClassName (original, or None if not in class)
            "funcName": constants.PLACEHOLDER_NA,  # e.g., methodName or functionName (original)
            "pathname": constants.PLACEHOLDER_NA,  # Full path to the caller's file
            "lineno": 0,  # Line number in the caller's file
            "moduleName": constants.PLACEHOLDER_NA,  # Module name of the caller
        }
        try:
            # inspect.stack(0) gives the current frame upwards.
            # We need to skip frames from within dynamicLoggerCore itself.
            frames = inspect.stack(0)
            frame_idx = 0
            for frameInfo in frames:
                frame = frameInfo.frame
                filename = frame.f_code.co_filename
                funcCode = frame.f_code
                func_name_from_code = frame.f_code.co_name
                self._instanceDebugPrint(
                    f"  Stack frame {frame_idx}: file='{filename}', func='{func_name_from_code}'")

                # Skip frames from this module (callerInspector.py)
                if filename == self._callerInspectorModuleFilename:
                    self._instanceDebugPrint(
                        f"    Skipping: Frame is from CallerInspector module itself.")
                    frame_idx += 1
                    continue

                # Skip frames from other dynamicLoggerCore modules (more general check)
                # This relies on the 'dynamicLoggerCore' directory name being in the path.
                # A more robust check might involve comparing with known module objects/paths if available.
                if 'dynamicLoggerCore' in filename.replace('\\', '/'):  # Normalize path separators
                    self._instanceDebugPrint(
                        f"    Skipping: Frame appears to be from dynamicLoggerCore ('{filename}').")
                    frame_idx += 1
                    continue

                # Skip frames from specified wrapper functions
                if funcCode in self._wrappersToSkipCodes:
                    self._instanceDebugPrint(
                        f"    Skipping: Frame's code object is in wrappersToSkip.")
                    frame_idx += 1
                    continue

                # If we reach here, this is considered the relevant calling frame.
                self._instanceDebugPrint(f"    Found relevant caller frame at index {frame_idx}.")
                originalFuncName = frame.f_code.co_name  # This is the raw function/method name

                module = inspect.getmodule(frame)
                moduleName = module.__name__ if module else constants.PLACEHOLDER_NA

                classNameStr: Optional[str] = None
                # Heuristic to find class name:
                # If 'self' is in locals, it's likely an instance method.
                # If 'cls' is in locals, it's likely a class method.
                if 'self' in frame.f_locals:
                    instance_obj = frame.f_locals['self']
                    if hasattr(instance_obj, '__class__'):
                        classNameStr = instance_obj.__class__.__name__
                elif 'cls' in frame.f_locals:
                    class_obj = frame.f_locals['cls']
                    if hasattr(class_obj, '__name__'):  # class_obj itself is the class
                        classNameStr = class_obj.__name__

                # Sanitize parts for funcMethIndicator
                sanitizedFuncName = self._sanitizeSegment(originalFuncName)
                funcMethIndicator = sanitizedFuncName  # Default if not in class

                if classNameStr:
                    sanitizedClassName = self._sanitizeSegment(classNameStr)
                    funcMethIndicator = f"{sanitizedClassName}.{sanitizedFuncName}"

                details = {
                    "funcMethIndicator": funcMethIndicator,
                    # Sanitized, for HO matching & %(callerInfo)s
                    "className": classNameStr,
                    # Original class name (if any), for default file naming
                    "funcName": originalFuncName,
                    # Original func/method name, for LogRecord.funcName & file naming
                    "pathname": filename,  # Full path to source file
                    "lineno": frame.f_lineno,  # Line number
                    "moduleName": moduleName,  # Module name
                }
                self._instanceDebugPrint(f"    Determined caller details: {details}")
                return details

            # If loop completes without returning, no suitable frame was found (highly unlikely)
            self._instanceDebugPrint(
                "    No suitable caller frame found after iterating stack. Returning default details.")
            return defaultDetails
        except Exception as e:
            # In case of any error during inspection, return defaults to avoid crashing logger.
            print(
                f"ERROR_CallerInspector ({id(self)}): Exception during caller detail determination: {e}. Returning default details.",
                file=sys.stderr, flush=True)
            self._instanceDebugPrint(
                f"    Exception during inspection: {e}. Returning default details.")
            return defaultDetails
        finally:
            # inspect.stack() in modern Python doesn't usually require manual frame deletion (del frame).
            # It's good practice to ensure frames are not held longer than necessary if issues arise.
            # For Python 3.4+, frame objects are cleared when the FrameInfo objects go out of scope.
            pass
