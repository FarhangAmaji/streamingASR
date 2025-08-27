# dynamicLoggerCore/pathResolver.py
import os
import sys
import re
from pathlib import Path
from typing import Optional, Union

from utils.dynamicLoggerModule.dyLogUtils import constants
from utils.dynamicLoggerModule.dyLogUtils import validation


class PathResolver:
    def __init__(self, explicitBasePath: Optional[Path] = None, debugPrint: bool = False):
        self._debugPrintActive = debugPrint
        self._basePath: Optional[Path] = None
        self._instanceDebugPrint(
            f"PathResolver Initializing. explicitBasePath='{explicitBasePath}', debugPrint={debugPrint}")

        if explicitBasePath:
            # Validate and use the explicitly provided basePath
            if isinstance(explicitBasePath, Path) and explicitBasePath.is_dir():
                self._basePath = explicitBasePath.resolve()
                self._instanceDebugPrint(
                    f"  Using explicitly provided valid basePath: {self._basePath}")
            else:
                self._instanceDebugPrint(
                    f"  Explicitly provided basePath '{explicitBasePath}' is invalid (not a Path object or not a directory). "
                    f"Falling back to heuristic determination.")
                self._determineBasePathFallbackHeuristics()  # Use original heuristics
        else:
            # No explicitBasePath provided, use fallback heuristics
            self._instanceDebugPrint(
                "  No explicitBasePath provided. Using fallback heuristic determination.")
            self._determineBasePathFallbackHeuristics()

        # Final check: if _basePath is still None after all attempts, default to CWD.
        # This is a critical safeguard.
        if self._basePath is None:
            emergency_msg = f"CRITICAL_PathResolver ({id(self)}): _basePath is STILL None after all initialization logic. THIS IS UNEXPECTED. Defaulting to CWD."
            self._instanceDebugPrint(emergency_msg)
            print(emergency_msg, file=sys.stderr, flush=True)
            try:
                self._basePath = Path.cwd().resolve()
            except Exception as e_cwd_final:
                # If even CWD fails, this is a very broken environment.
                # Use a hardcoded relative path as an absolute last resort.
                self._basePath = Path("").resolve()  # Should almost always work
                print(
                    f"ULTRA_CRITICAL_PathResolver ({id(self)}): Fallback to CWD also failed ({e_cwd_final}). Using '.' resolved to '{self._basePath}'. Log paths will be highly unpredictable.",
                    file=sys.stderr, flush=True)

        self._instanceDebugPrint(
            f"PathResolver Initialized. Final effective basePath: {self._basePath}")

    def _instanceDebugPrint(self, message: str):
        if self._debugPrintActive:
            print(f"DEBUG_PathResolver ({id(self)}): {message}", file=sys.stderr, flush=True)

    @property
    def basePath(self) -> Path:
        """
        The resolved base path for the DynamicLogger instance.
        Used as a root for relative file paths and default log directories.
        This property ensures _basePath is never None when accessed externally.
        """
        if self._basePath is None:
            # This condition implies a severe failure during __init__ if reached.
            # The __init__ logic should always set _basePath.
            critical_error_msg = f"CRITICAL_ERROR_PathResolver ({id(self)}): Accessing basePath property but internal _basePath is None. " \
                                 f"This indicates a major issue in PathResolver's initialization. Attempting emergency CWD fallback."
            print(critical_error_msg, file=sys.stderr, flush=True)
            self._instanceDebugPrint(critical_error_msg)
            try:
                return Path.cwd().resolve()  # Emergency fallback
            except Exception:
                return Path("").resolve()  # Absolute last resort
        return self._basePath

    def _isIdeOrTestRunnerPath(self, path_str: str) -> bool:
        """Checks if a given path string seems to belong to an IDE or test runner (for fallback heuristics)."""
        if not path_str: return False
        path_str_lower = path_str.lower()
        patterns = [
            "pycharm", "_jb_pytest_runner.py", "_jb_unittest_runner.py",
            "PYTHON_UNITTEST_DISCOVERER",
            "vscode", "ptvsd", "debugpy", "unittest", "pytest", "nosetests", "tox",
            "thonny", "spyder", "idlelib"
        ]
        for p in patterns:
            if p in path_str_lower:
                self._instanceDebugPrint(
                    f"  (Fallback Heuristic) Path '{path_str}' matches IDE/runner pattern '{p}'.")
                return True
        return False

    def _determineBasePathFallbackHeuristics(self) -> None:
        """
        Original heuristic-based basePath determination.
        Used if no valid explicitBasePath is provided during initialization.
        Sets self._basePath.
        Priority:
        1. Root directory of the DynamicLogger library itself (if identifiable).
        2. Directory of the main script that launched the application (sys.argv[0], if not an IDE runner).
        3. Current working directory (Path.cwd()).
        """
        self._instanceDebugPrint("Attempting to determine basePath using fallback heuristics...")
        provisional_base_path_heuristic0: Optional[Path] = None

        # Heuristic 0: DynamicLogger library root
        try:
            # __file__ for pathResolver.py -> .parent (dynamicLoggerCore) -> .parent.parent (project root of DL)
            loggerLibRoot = Path(__file__).resolve().parent.parent
            self._instanceDebugPrint(
                f"  Fallback Heuristic 0 (Library Root): Path(__file__) is '{Path(__file__).resolve()}', potential DL lib root is '{loggerLibRoot}'")
            if loggerLibRoot.is_dir() and \
                    (loggerLibRoot / "dynamicLoggerCore").is_dir() and \
                    (loggerLibRoot / "dyLogUtils").is_dir():
                provisional_base_path_heuristic0 = loggerLibRoot
                self._instanceDebugPrint(
                    f"    Fallback Heuristic 0: DL lib root '{provisional_base_path_heuristic0}' identified based on subdir structure.")
            else:
                self._instanceDebugPrint(
                    f"    Fallback Heuristic 0: Potential DL lib root '{loggerLibRoot}' lacks expected subdirs. Not using.")
        except Exception as e_h0:
            self._instanceDebugPrint(f"    Fallback Heuristic 0: Exception: {e_h0}")

        # Heuristic 1: Main script's directory (from sys.argv[0])
        # This heuristic is considered if it provides a "better" or "more specific" path than Heuristic 0.
        try:
            if hasattr(sys, 'argv') and sys.argv and sys.argv[0] and sys.argv[0].strip():
                main_script_arg = sys.argv[0]
                self._instanceDebugPrint(
                    f"  Fallback Heuristic 1 (Main Script Dir): sys.argv[0] is '{main_script_arg}'")
                if not self._isIdeOrTestRunnerPath(main_script_arg):
                    mainScriptPath = Path(main_script_arg)
                    try:
                        resolvedMainScriptPath = mainScriptPath.resolve(strict=False)
                        self._instanceDebugPrint(
                            f"    Fallback Heuristic 1: mainScriptPath '{mainScriptPath}' resolved to '{resolvedMainScriptPath}'.")
                        potentialBasePathFromArgv = resolvedMainScriptPath.parent
                        if potentialBasePathFromArgv.is_dir():
                            # Decision logic: When to prefer argv-based path over library root path
                            if provisional_base_path_heuristic0 is None:  # No lib root found, argv path is best so far
                                self._instanceDebugPrint(
                                    f"    Fallback Heuristic 1: Lib root not found, using main script parent: '{potentialBasePathFromArgv}'.")
                                self._basePath = potentialBasePathFromArgv
                                return
                            # Example: If main script is deeper within the lib root, lib root is still preferred.
                            # If main script is outside lib root, prefer main script's dir.
                            elif not str(potentialBasePathFromArgv.resolve()).startswith(
                                    str(provisional_base_path_heuristic0.resolve())):  # Compare resolved paths
                                self._instanceDebugPrint(
                                    f"    Fallback Heuristic 1: Main script parent '{potentialBasePathFromArgv}' is outside lib root '{provisional_base_path_heuristic0}'. Preferring main script parent.")
                                self._basePath = potentialBasePathFromArgv
                                return
                            else:  # Main script path is within or same as lib root, stick with lib root.
                                self._instanceDebugPrint(
                                    f"    Fallback Heuristic 1: Main script parent '{potentialBasePathFromArgv}' is within/same as lib root. Sticking with lib root provisional: '{provisional_base_path_heuristic0}'.")
                                self._basePath = provisional_base_path_heuristic0
                                return
                        else:  # Parent dir of main script doesn't exist
                            self._instanceDebugPrint(
                                f"    Fallback Heuristic 1: Parent directory of main script '{potentialBasePathFromArgv}' does not exist.")
                    except Exception as e_resolve_main:  # Error resolving main script path
                        self._instanceDebugPrint(
                            f"    Fallback Heuristic 1: Error resolving main script path '{mainScriptPath}': {e_resolve_main}.")
                else:  # sys.argv[0] looks like an IDE/runner
                    self._instanceDebugPrint(
                        f"    Fallback Heuristic 1: sys.argv[0] '{main_script_arg}' is an IDE/test runner. This heuristic won't override.")
            else:  # sys.argv[0] unavailable
                self._instanceDebugPrint(
                    "    Fallback Heuristic 1: sys.argv[0] is missing or empty.")
        except Exception as e_h1:  # General error in Heuristic 1 block
            self._instanceDebugPrint(f"    Fallback Heuristic 1: Exception: {e_h1}")

        # If _basePath is still not set after Heuristic 1 (e.g., Heuristic 1 didn't apply or didn't override)
        if self._basePath is None:
            if provisional_base_path_heuristic0 is not None:  # Use Heuristic 0 if it found something
                self._instanceDebugPrint(
                    f"  Using provisional basePath from Fallback Heuristic 0: '{provisional_base_path_heuristic0}'.")
                self._basePath = provisional_base_path_heuristic0
            else:  # All above failed, fall back to CWD
                self._instanceDebugPrint(
                    "  All primary fallback heuristics (Lib Root, Main Script Dir) failed or were inconclusive. Using CWD.")
                try:
                    self._basePath = Path.cwd().resolve()
                    self._instanceDebugPrint(
                        f"    Fallback Heuristic (CWD): basePath set to CWD: '{self._basePath}'")
                except Exception as e_cwd:  # Should be rare
                    self._instanceDebugPrint(
                        f"    Fallback Heuristic (CWD): Failed to get CWD: {e_cwd}. Critical.")
                    self._basePath = Path("").resolve()  # Last resort relative path
                    print(
                        f"CRITICAL_WARNING_PathResolver ({id(self)}): Fallback basePath determination (CWD error). Using relative '.' resolved to '{self._basePath}'. Log paths may be unpredictable.",
                        file=sys.stderr, flush=True)
        # else: _basePath was already set by Heuristic 1 if it was chosen.

    def _sanitizeForFilename(self, name: str, maxLength: int = 60) -> str:
        if not name or name == constants.PLACEHOLDER_NA:
            return "unspecified_log_segment"
        sanitized = re.sub(r'[\x00-\x1F\x7F-\x9F\s]', '_', name, flags=re.UNICODE)
        sanitized = re.sub(r'[<>:"/\\|?*&!,;=\[\]{}]', '_', sanitized)
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_')
        if not sanitized:
            return "sanitized_to_empty"
        if len(sanitized) > maxLength:
            sanitized = sanitized[:maxLength].strip('_')
        if not sanitized:
            return "shortened_to_empty"
        return sanitized

    def resolveFilePath(self,
                        filePathArg: Union[Path, str, bool, None],
                        callerClassName: Optional[str] = None,
                        callerFuncName: Optional[str] = None,
                        isSimpleLogMode: bool = False) -> Optional[Path]:
        self._instanceDebugPrint(
            f"resolveFilePath called: filePathArg='{str(filePathArg)[:100]}' (type: {type(filePathArg)}), "
            f"callerClassName='{callerClassName}', callerFuncName='{callerFuncName}', simpleLogMode={isSimpleLogMode}")

        if filePathArg is False:
            self._instanceDebugPrint("  filePathArg is False. No file logging.")
            return None
        if filePathArg is None:  # This means the argument was not provided or passed through the priority chain.
            # Its effective value will be its FINAL_DEFAULT (which is True).
            # The argument processor will have already substituted the final default value.
            # So, if filePathArg arrives here as None, it means it resolved to None *after* defaults.
            # This should only happen if FINAL_DEFAULTS for filePath was None (which it isn't, it's True).
            # Or if an explicit 'None' was passed through all levels and was the actual winning value.
            # For safety, if it IS None here, we'll treat it based on final default.
            if not constants.FINAL_DEFAULTS.get(constants.ARG_FILE_PATH,
                                                False):  # Check actual final default
                self._instanceDebugPrint(
                    "  filePathArg is None and final default for filePath is configured as False. No file logging.")
                return None
            else:
                self._instanceDebugPrint(
                    "  filePathArg is None. Proceeding as if filePath=True (based on final default).")
                filePathArg = True  # Default behavior for True follows

        currentBasePath = self.basePath  # Access via property to ensure it's resolved
        defaultLogsSubDir = currentBasePath / constants.DEFAULT_LOGS_DIR_NAME
        self._instanceDebugPrint(
            f"  Using resolved basePath: '{currentBasePath}'. Default logs subdir target: '{defaultLogsSubDir}'")

        preliminaryPath: Optional[Path] = None

        if isinstance(filePathArg, Path):
            self._instanceDebugPrint(f"  filePathArg is a Path object: '{filePathArg}'")
            preliminaryPath = filePathArg if filePathArg.is_absolute() else currentBasePath / filePathArg
        elif isinstance(filePathArg, str):
            self._instanceDebugPrint(f"  filePathArg is a str: '{filePathArg}'")
            if not validation.isNonEmptyString(filePathArg):
                self._instanceDebugPrint(
                    "    filePathArg string is empty. Treating as no file logging for this call.")
                # Consistent with brief: "" should be treated as None (no file logging for call)
                # Or rather, an empty string filePath should not result in a file.
                return None

            pathArgAsPathObj = Path(filePathArg)
            if pathArgAsPathObj.is_absolute():
                self._instanceDebugPrint(f"    String is an absolute path: '{pathArgAsPathObj}'")
                preliminaryPath = pathArgAsPathObj
            elif os.sep in filePathArg or (os.altsep and os.altsep in filePathArg):
                self._instanceDebugPrint(
                    f"    String is a relative path with separators: '{filePathArg}'")
                preliminaryPath = currentBasePath / pathArgAsPathObj
            else:  # "simple string" (no OS separators)
                self._instanceDebugPrint(
                    f"    String is a 'simple string' (no separators): '{filePathArg}'")
                # Sanitize the whole string as a potential filename stem
                sanitizedFileNameStem = self._sanitizeForFilename(
                    Path(filePathArg).stem if Path(filePathArg).stem else filePathArg)
                extension = Path(filePathArg).suffix
                if not extension and constants.LOG_FILE_EXTENSION:  # Use default if no ext and default exists
                    extension = constants.LOG_FILE_EXTENSION
                elif extension and not validation.isValidFilePathSegment(
                        extension[1:]):  # Basic check for valid extension part
                    self._instanceDebugPrint(
                        f"      Warning: Simple string has unusual extension '{extension}'. Using as is.")
                    # Allow it, but it's unusual. User might intend e.g. "log.backup.2023"

                finalFileName = sanitizedFileNameStem + extension
                preliminaryPath = defaultLogsSubDir / finalFileName
                self._instanceDebugPrint(
                    f"      Sanitized filename: '{finalFileName}'. Path: '{preliminaryPath}'")
        elif filePathArg is True:
            self._instanceDebugPrint(
                "  filePathArg is True. Generating default filename based on mode and caller context.")
            fileNameBase = ""
            if isSimpleLogMode:
                fileNameBase = constants.DEFAULT_SIMPLE_LOG_FILENAME_BASE
                self._instanceDebugPrint(
                    f"    SimpleLogMode: Using default base filename '{fileNameBase}'")
            elif callerClassName and validation.isNonEmptyString(callerClassName):
                fileNameBase = self._sanitizeForFilename(callerClassName)
                self._instanceDebugPrint(
                    f"    FullMode: Using sanitized ClassName '{callerClassName}' -> '{fileNameBase}'")
            elif callerFuncName and validation.isNonEmptyString(
                    callerFuncName) and callerFuncName != constants.PLACEHOLDER_NA:
                fileNameBase = self._sanitizeForFilename(callerFuncName)
                self._instanceDebugPrint(
                    f"    FullMode (no class): Using sanitized FunctionName '{callerFuncName}' -> '{fileNameBase}'")
            else:  # Fallback if no good caller info
                fileNameBase = "dynamic_log_default"  # Fallback if no meaningful name
                self._instanceDebugPrint(
                    f"    FullMode (no class/func info): Using fallback base filename '{fileNameBase}'")

            preliminaryPath = defaultLogsSubDir / (fileNameBase + constants.LOG_FILE_EXTENSION)
        else:
            self._instanceDebugPrint(
                f"  filePathArg type '{type(filePathArg)}' or value '{filePathArg}' is not recognized or invalid. No file logging.")
            return None

        if preliminaryPath:
            self._instanceDebugPrint(f"  Preliminary path determined: '{preliminaryPath}'")
            try:
                # Resolve to get an absolute path and normalize (e.g., remove "..")
                # strict=False allows resolving even if the file/final dir doesn't exist yet
                finalAbsPath = preliminaryPath.resolve(strict=False)
                self._instanceDebugPrint(f"  Path after .resolve(strict=False): '{finalAbsPath}'")

                # Create parent directories if they don't exist
                finalAbsPath.parent.mkdir(parents=True, exist_ok=True)
                self._instanceDebugPrint(
                    f"    Successfully ensured parent directory '{finalAbsPath.parent}' exists.")

                # Final check for path validity (e.g. length, though OS usually handles this)
                # The main validation is ensuring the directory can be made.
                # Specific filename character validation was done by _sanitizeForFilename for simple names.
                # For absolute/relative paths, user is more responsible.
                self._instanceDebugPrint(f"  Returning final absolute path: '{finalAbsPath}'")
                return finalAbsPath
            except (OSError, SecurityException,
                    Exception) as e_path_finalize:  # Catch more specific OS errors if possible
                # SecurityException is from Java, not Python.
                # Python uses PermissionError, FileNotFoundError etc.
                resolvedPathStrAttempt = str(
                    preliminaryPath)  # Use the path before .resolve() if resolve itself failed
                try:
                    resolvedPathStrAttempt = str(preliminaryPath.resolve(strict=False))
                except Exception:  # If resolve fails, stick to original string
                    pass

                print(
                    f"ERROR_PathResolver ({id(self)}): Error during path finalization or parent directory creation for '{resolvedPathStrAttempt}'. Error: {e_path_finalize}",
                    file=sys.stderr, flush=True)
                self._instanceDebugPrint(
                    f"    Exception during path finalization: {e_path_finalize}. Cannot use this path.")
                return None

        self._instanceDebugPrint("  No preliminary path was determined. No file logging.")
        return None
