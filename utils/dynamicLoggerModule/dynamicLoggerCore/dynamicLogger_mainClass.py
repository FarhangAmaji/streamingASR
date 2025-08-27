# dynamicLoggerCore/dynamicLogger.py
import inspect
import logging
import shutil
import sys
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple, Union, Set

from utils.dynamicLoggerModule.dyLogUtils import constants
from utils.dynamicLoggerModule.dyLogUtils import validation
from utils.dynamicLoggerModule.dyLogUtils.projectTypes import FileWriteModeType
from utils.dynamicLoggerModule.dynamicLoggerCore.argumentProcessor import ArgumentProcessor
from utils.dynamicLoggerModule.dynamicLoggerCore.callerInspector import CallerInspector
from utils.dynamicLoggerModule.dynamicLoggerCore.configManager import ConfigManager
from utils.dynamicLoggerModule.dynamicLoggerCore.handlerManager import HandlerManager
from utils.dynamicLoggerModule.dynamicLoggerCore.pathResolver import PathResolver

INTERNAL_LOGGER_NAME_PREFIX = "__dynamicLogger_instance_"
_internalLoggerCounter = 0
_internalLoggerRLock = threading.RLock()


def getUniqueInternalLoggerName() -> str:
    global _internalLoggerCounter
    with _internalLoggerRLock:
        _internalLoggerCounter += 1
        return f"{INTERNAL_LOGGER_NAME_PREFIX}{_internalLoggerCounter}"


class DynamicLogger:
    CRITICAL = logging.CRITICAL
    FATAL = logging.FATAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    WARN = logging.WARN
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET

    def __init__(self,
                 configSet: Optional[Dict[str, Any]] = None,
                 indicatorName: Optional[str] = None,
                 printToConsole: Optional[bool] = None,
                 filePath: Optional[Union[Path, str, bool]] = None,
                 fileWriteMode: Optional[FileWriteModeType] = None,
                 exclude: Optional[bool] = None,
                 logFormat: Optional[Union[str, bool]] = None,
                 timestampFormat: Optional[Union[str, bool]] = None,
                 excInfo: Optional[Union[bool, Tuple, BaseException]] = None,
                 extra: Optional[Dict[str, Any]] = None,
                 logLevel: Optional[int] = None,
                 stackInfo: Optional[bool] = None,
                 highOrderOptions: Optional[Dict[str, Any]] = None,
                 wrappersToSkip: Optional[List[Callable[..., Any]]] = None,
                 debugPrint: bool = False,
                 **otherInitKwargs: Any):

        self._threadLock = threading.RLock()
        self._debugPrintActive = debugPrint

        self._wMode_remove_mainLogFolder_done: bool = False
        self._wMode_removed_log_files_this_session: Set[Path] = set()

        self._instanceDebugPrint(f"Initializing new DynamicLogger instance. Object ID: {id(self)}")

        determined_import_root_for_basepath: Optional[Path] = None
        try:
            frame_info_of_caller = inspect.stack(0)[1]
            instantiating_module_file = Path(frame_info_of_caller.filename).resolve()
            self._instanceDebugPrint(f"  Instantiation from file: '{instantiating_module_file}'")

            instantiating_module = inspect.getmodule(frame_info_of_caller.frame)

            if instantiating_module and hasattr(instantiating_module, '__file__') and \
                    hasattr(instantiating_module,
                            '__package__') and instantiating_module.__package__:
                module_path = Path(instantiating_module.__file__).resolve()
                package_name = instantiating_module.__package__
                top_level_package_name = package_name.split('.')[0]
                self._instanceDebugPrint(
                    f"  Instantiating module: '{instantiating_module.__name__}', package: '{package_name}', top-level: '{top_level_package_name}'")
                package_depth = len(package_name.split('.'))
                if package_depth > 0 and len(module_path.parents) > package_depth:
                    candidate_root = module_path.parents[package_depth]
                    if (candidate_root / top_level_package_name).is_dir():
                        determined_import_root_for_basepath = candidate_root.resolve()
                        self._instanceDebugPrint(
                            f"    Derived import root (package context): '{determined_import_root_for_basepath}'")
                    else:
                        self._instanceDebugPrint(
                            f"    Candidate root '{candidate_root}' does not seem to contain top-level package '{top_level_package_name}'.")
                else:
                    self._instanceDebugPrint(
                        f"    Could not determine package root based on package depth. Module path: {module_path}, Package depth: {package_depth}")
            elif instantiating_module and hasattr(instantiating_module, '__file__'):
                determined_import_root_for_basepath = Path(
                    instantiating_module.__file__).resolve().parent
                self._instanceDebugPrint(
                    f"  Instantiating module is a top-level script. Using its directory as import root: '{determined_import_root_for_basepath}'")
            else:
                determined_import_root_for_basepath = instantiating_module_file.parent
                self._instanceDebugPrint(
                    f"  Could not get instantiating module info fully. Using directory of calling file: '{determined_import_root_for_basepath}'")

        except IndexError:
            self._instanceDebugPrint(
                "  Could not determine instantiation site frame via inspect.stack()[1].")
        except Exception as e_basepath_det:
            self._instanceDebugPrint(
                f"  Error determining import root for basePath: {e_basepath_det}")

        if determined_import_root_for_basepath is None or not determined_import_root_for_basepath.is_dir():
            self._instanceDebugPrint(
                f"  Determined import root '{determined_import_root_for_basepath}' is invalid or None. PathResolver will use its fallbacks.")

        initDirectArgsCollected: Dict[str, Any] = {}
        initConfgiSetArgsFromInit: Dict[str, Any] = {}

        if configSet is not None:
            if not isinstance(configSet, dict):
                print(
                    f"Warning_DynamicLogger ({id(self)}): '{constants.ARG_CONFIG_SET}' in __init__ was not a dict. Ignored.",
                    file=sys.stderr, flush=True)
            else:
                unknownCsKeys = validation.isValidConfigSetDict(configSet,
                                                                constants.ALL_MAIN_LOG_ARGS_KEYS)
                if unknownCsKeys:
                    for uKey in unknownCsKeys:
                        print(
                            f"Warning_DynamicLogger ({id(self)}): Unknown key '{uKey}' in __init__ configSet. Ignored.",
                            file=sys.stderr, flush=True)
                    initConfgiSetArgsFromInit = {
                        k: v for k, v in configSet.items() if
                        k in constants.ALL_MAIN_LOG_ARGS_KEYS and k != constants.ARG_CONFIG_SET
                    }
                else:
                    initConfgiSetArgsFromInit = configSet.copy()

        instanceDefaultIndicatorName = indicatorName

        potentialDirectArgs = {
            constants.ARG_PRINT_TO_CONSOLE: printToConsole,
            constants.ARG_FILE_PATH: filePath,
            constants.ARG_FILE_WRITE_MODE: fileWriteMode,
            constants.ARG_EXCLUDE: exclude,
            constants.ARG_LOG_FORMAT: logFormat,
            constants.ARG_TIMESTAMP_FORMAT: timestampFormat,
            constants.ARG_EXC_INFO: excInfo,
            constants.ARG_EXTRA: extra,
            constants.ARG_LOG_LEVEL: logLevel,
            constants.ARG_STACK_INFO: stackInfo,
        }
        for key, value in potentialDirectArgs.items():
            if value is not None:
                initDirectArgsCollected[key] = value

        for key, value in otherInitKwargs.items():
            if key in constants.ALL_MAIN_LOG_ARGS_KEYS and key not in initDirectArgsCollected and key != constants.ARG_CONFIG_SET and key != constants.ARG_INDICATOR_NAME:
                initDirectArgsCollected[key] = value
            elif key == constants.ARG_INDICATOR_NAME and instanceDefaultIndicatorName is None:
                instanceDefaultIndicatorName = value
            elif key == constants.ARG_CONFIG_SET and not initConfgiSetArgsFromInit:
                if isinstance(value, dict):
                    unknownCsKeys_kw = validation.isValidConfigSetDict(value,
                                                                       constants.ALL_MAIN_LOG_ARGS_KEYS)
                    if unknownCsKeys_kw:
                        for uKey_kw in unknownCsKeys_kw: print(
                            f"Warning_DynamicLogger ({id(self)}): Unknown key '{uKey_kw}' in __init__ kwargs configSet. Ignored.",
                            file=sys.stderr, flush=True)
                        initConfgiSetArgsFromInit = {k_kw: v_kw for k_kw, v_kw in value.items() if
                                                     k_kw in constants.ALL_MAIN_LOG_ARGS_KEYS and k_kw != constants.ARG_CONFIG_SET}
                    else:
                        initConfgiSetArgsFromInit = value.copy()
                else:
                    print(
                        f"Warning_DynamicLogger ({id(self)}): '{constants.ARG_CONFIG_SET}' in __init__ **kwargs was not a dict. Ignored.",
                        file=sys.stderr, flush=True)
            elif key not in constants.ALL_MAIN_LOG_ARGS_KEYS and key not in [
                constants.ARG_CONFIG_SET, constants.ARG_INDICATOR_NAME]:
                print(
                    f"Warning_DynamicLogger ({id(self)}): Unknown keyword argument '{key}' in __init__. Ignored.",
                    file=sys.stderr, flush=True)

        if instanceDefaultIndicatorName is not None and not validation.isNonEmptyString(
                instanceDefaultIndicatorName):
            print(
                f"Warning_DynamicLogger ({id(self)}): Initial '{constants.ARG_INDICATOR_NAME}' ('{instanceDefaultIndicatorName}') was invalid. Set to None.",
                file=sys.stderr, flush=True)
            instanceDefaultIndicatorName = None
        self._instanceDebugPrint(f"Instance indicatorName set to: {instanceDefaultIndicatorName}")

        self._instanceDebugPrint(f"Init direct args collected: {initDirectArgsCollected}")
        self._instanceDebugPrint(f"Init configSet args from init: {initConfgiSetArgsFromInit}")

        self._configManager = ConfigManager(
            initIndicatorName=instanceDefaultIndicatorName,
            initDirectArgs=initDirectArgsCollected,
            initConfgiSetArgs=initConfgiSetArgsFromInit,
            highOrderOptions=highOrderOptions,
            debugPrint=self._debugPrintActive
        )
        self._callerInspector = CallerInspector(wrappersToSkip, debugPrint=self._debugPrintActive)
        self._pathResolver = PathResolver(
            explicitBasePath=determined_import_root_for_basepath,
            debugPrint=self._debugPrintActive
        )
        self._argumentProcessor = ArgumentProcessor(self._configManager,
                                                    debugPrint=self._debugPrintActive)
        self._handlerManager = HandlerManager(self._threadLock, debugPrint=self._debugPrintActive)

        self._internalStdLogger = logging.getLogger(getUniqueInternalLoggerName())
        self._internalStdLogger.propagate = False
        self._internalStdLogger.setLevel(logging.DEBUG)
        self._instanceDebugPrint(
            f"Internal std logger '{self._internalStdLogger.name}' created and set to DEBUG level.")

    def _instanceDebugPrint(self, message: str):
        if self._debugPrintActive:
            print(f"DEBUG_DynamicLogger ({id(self)}): {message}", file=sys.stderr, flush=True)

    def _getEffectiveIndicatorNameForHoLookup(self,
                                              inlineIndicatorName: Optional[str],
                                              inlineConfigSetIndicatorName: Optional[str]) -> \
            Optional[str]:
        hoLookupIndicatorName = inlineIndicatorName
        if hoLookupIndicatorName is None:
            hoLookupIndicatorName = inlineConfigSetIndicatorName
        if hoLookupIndicatorName is None:
            hoLookupIndicatorName = self._configManager.instanceIndicatorName

        if hoLookupIndicatorName is not None and not validation.isNonEmptyString(
                hoLookupIndicatorName):
            return None
        return hoLookupIndicatorName

    def log(self,
            initialMessageLevel: int,
            message: str,
            *messageArgsTuple: Any,
            configSet: Optional[Dict[str, Any]] = None,
            indicatorName: Optional[str] = None,
            printToConsole: Optional[bool] = None,
            filePath: Optional[Union[Path, str, bool]] = None,
            fileWriteMode: Optional[FileWriteModeType] = None,
            exclude: Optional[bool] = None,
            logFormat: Optional[Union[str, bool]] = None,
            timestampFormat: Optional[Union[str, bool]] = None,
            excInfo: Optional[Union[bool, Tuple, BaseException]] = None,
            extra: Optional[Dict[str, Any]] = None,
            logLevel: Optional[int] = None,
            simpleLog: Optional[bool] = None,
            stackInfo: Optional[bool] = None,
            messageArgs: Optional[Tuple[Any, ...]] = None,
            **otherLogKwargs: Any) -> None:

        self._instanceDebugPrint(
            f"--- Log call started: msg='{message[:50]}...', level={initialMessageLevel} ---")

        allPassedKwargs = {}
        lcls = locals()
        explicitly_passed_named_args_in_log_signature = [
            (constants.ARG_CONFIG_SET, 'configSet'),
            (constants.ARG_INDICATOR_NAME, 'indicatorName'),
            (constants.ARG_PRINT_TO_CONSOLE, 'printToConsole'),
            (constants.ARG_FILE_PATH, 'filePath'),
            (constants.ARG_FILE_WRITE_MODE, 'fileWriteMode'),
            (constants.ARG_EXCLUDE, 'exclude'), (constants.ARG_LOG_FORMAT, 'logFormat'),
            (constants.ARG_TIMESTAMP_FORMAT, 'timestampFormat'),
            (constants.ARG_EXC_INFO, 'excInfo'),
            (constants.ARG_EXTRA, 'extra'), (constants.ARG_LOG_LEVEL, 'logLevel'),
            (constants.ARG_SIMPLE_LOG, 'simpleLog'), (constants.ARG_STACK_INFO, 'stackInfo'),
            (constants.ARG_MESSAGE_ARGS, 'messageArgs')
        ]
        for arg_const, param_name_in_sig in explicitly_passed_named_args_in_log_signature:
            param_value = lcls[param_name_in_sig]
            if param_name_in_sig == 'simpleLog':
                if param_value is not None:
                    allPassedKwargs[arg_const] = param_value
            elif param_value is not None:
                allPassedKwargs[arg_const] = param_value

        allPassedKwargs.update(otherLogKwargs)
        self._instanceDebugPrint(f"  All effectively passed kwargs to log(): {allPassedKwargs}")

        with self._threadLock:
            self._instanceDebugPrint("Lock acquired for log call.")

            isSimpleLogModeEffective = simpleLog is True

            if not isSimpleLogModeEffective and isinstance(configSet, dict) and \
                    configSet.get(constants.ARG_SIMPLE_LOG) is True:
                print(
                    f"Warning_DynamicLogger ({id(self)}): '{constants.ARG_SIMPLE_LOG}=True' found in inline configSet "
                    f"but not passed directly as simpleLog=True. "
                    f"'{constants.ARG_SIMPLE_LOG}' from configSet will be ignored for mode determination.",
                    file=sys.stderr, flush=True)

            self._instanceDebugPrint(
                f"Effective simpleLog mode for this call: {isSimpleLogModeEffective}")

            inlineDirectArgsCollected = {
                k: v for k, v in allPassedKwargs.items()
                if k in constants.ALL_MAIN_LOG_ARGS_KEYS and k != constants.ARG_CONFIG_SET
            }
            if constants.ARG_SIMPLE_LOG in allPassedKwargs:
                inlineDirectArgsCollected[constants.ARG_SIMPLE_LOG] = allPassedKwargs[
                    constants.ARG_SIMPLE_LOG]

            self._instanceDebugPrint(
                f"Collected inlineDirectArgs (L5): {inlineDirectArgsCollected}")

            inlineConfigSetFromParam = allPassedKwargs.get(constants.ARG_CONFIG_SET)
            if not isinstance(inlineConfigSetFromParam, dict):
                if inlineConfigSetFromParam is not None:
                    print(
                        f"Warning_DynamicLogger ({id(self)}): Inline '{constants.ARG_CONFIG_SET}' was not a dict. Ignored.",
                        file=sys.stderr, flush=True)
                inlineConfigSetFromParam = None
            else:
                unknownCsKeys = validation.isValidConfigSetDict(inlineConfigSetFromParam,
                                                                constants.ALL_MAIN_LOG_ARGS_KEYS)
                if unknownCsKeys:
                    for uKey in unknownCsKeys: print(
                        f"Warning_DynamicLogger ({id(self)}): Unknown key '{uKey}' in inline configSet. Ignored.",
                        file=sys.stderr, flush=True)
                    inlineConfigSetFromParam = {
                        k_cs: v_cs for k_cs, v_cs in inlineConfigSetFromParam.items() if
                        k_cs in constants.ALL_MAIN_LOG_ARGS_KEYS and k_cs != constants.ARG_CONFIG_SET
                    }
            self._instanceDebugPrint(f"Inline configSet (L6): {inlineConfigSetFromParam}")

            hoLookupIndicator = self._getEffectiveIndicatorNameForHoLookup(
                inlineDirectArgsCollected.get(constants.ARG_INDICATOR_NAME),
                inlineConfigSetFromParam.get(
                    constants.ARG_INDICATOR_NAME) if inlineConfigSetFromParam else None
            )
            self._instanceDebugPrint(f"Effective indicator for HO lookup: {hoLookupIndicator}")

            callerDetails: Optional[Dict[str, Any]] = None
            funcMethIndicatorForHoLookup: Optional[str] = None
            if not isSimpleLogModeEffective:
                self._instanceDebugPrint("Full mode: determining caller details.")
                callerDetails = self._callerInspector.determineCallerDetails()
                funcMethIndicatorForHoLookup = callerDetails.get(
                    "funcMethIndicator")
                self._instanceDebugPrint(f"CallerDetails: {callerDetails}")
            else:
                self._instanceDebugPrint("Simple mode: skipping caller detail inspection.")

            self._instanceDebugPrint("Calling ArgumentProcessor.resolveEffectiveArgs...")
            finalResolvedOptions = self._argumentProcessor.resolveEffectiveArgs(
                initialMessageLevel=initialMessageLevel,
                inlineDirectArgs=inlineDirectArgsCollected,
                inlineConfigSetFromKwargs=inlineConfigSetFromParam,
                effectiveIndicatorNameForHoLookup=hoLookupIndicator,
                funcMethIndicatorForHoLookup=funcMethIndicatorForHoLookup,
                isSimpleLogMode=isSimpleLogModeEffective
            )
            self._instanceDebugPrint(
                f"ArgumentProcessor returned finalResolvedOptions. Excerpt: "
                f"printToConsole: {finalResolvedOptions.get(constants.ARG_PRINT_TO_CONSOLE)}, "
                f"filePath: {str(finalResolvedOptions.get(constants.ARG_FILE_PATH))[:50]}, "
                f"fileWriteMode: {finalResolvedOptions.get(constants.ARG_FILE_WRITE_MODE)}")

            if finalResolvedOptions.get(constants.ARG_EXCLUDE, False):
                self._instanceDebugPrint("Exclude is True. Aborting log call.")
                return

            # --- fileWriteMode 'w' logic ---
            effectiveFileWriteMode = finalResolvedOptions.get(constants.ARG_FILE_WRITE_MODE)
            # This will store the path resolved by PathResolver for the current call, if applicable
            currentCallResolvedPath: Optional[Path] = None

            if effectiveFileWriteMode == 'w' and finalResolvedOptions.get(
                    constants.ARG_FILE_PATH) is not False:
                self._instanceDebugPrint(f"Effective fileWriteMode is 'w'. Processing deletions.")
                mainLogFolderPath = (
                        self._pathResolver.basePath / constants.DEFAULT_LOGS_DIR_NAME).resolve()

                # Resolve current call's path first to determine if it targets mainLogFolder or an external file
                callerClassNameForPath = callerDetails.get(
                    "className") if callerDetails and not isSimpleLogModeEffective else None
                callerFuncNameForPath = callerDetails.get(
                    "funcName") if callerDetails and not isSimpleLogModeEffective else None
                currentCallResolvedPath = self._pathResolver.resolveFilePath(
                    filePathArg=finalResolvedOptions.get(constants.ARG_FILE_PATH),
                    callerClassName=callerClassNameForPath,
                    callerFuncName=callerFuncNameForPath,
                    isSimpleLogMode=isSimpleLogModeEffective
                )
                self._instanceDebugPrint(
                    f"  Path for current call (for 'w' mode checks): {currentCallResolvedPath}")

                if currentCallResolvedPath:  # Only proceed if there's an actual file path
                    # 1. Main Log Folder Deletion (once per instance session, if targeted)
                    if not self._wMode_remove_mainLogFolder_done:
                        # Check if the current log call's target file is *within* the mainLogFolderPath.
                        if str(currentCallResolvedPath.resolve()).startswith(
                                str(mainLogFolderPath)):
                            self._instanceDebugPrint(
                                f"  Current 'w' call targets mainLogFolder. Flag _wMode_remove_mainLogFolder_done is False. "
                                f"Attempting to remove/recreate mainLogFolder: {mainLogFolderPath}")
                            try:
                                if mainLogFolderPath.exists():
                                    shutil.rmtree(mainLogFolderPath)
                                    self._instanceDebugPrint(
                                        f"    Successfully removed mainLogFolder: {mainLogFolderPath}")
                                else:
                                    self._instanceDebugPrint(
                                        f"    MainLogFolder {mainLogFolderPath} does not exist. Skipping rmtree.")
                                mainLogFolderPath.mkdir(parents=True, exist_ok=True)
                                self._instanceDebugPrint(
                                    f"    Successfully ensured mainLogFolder exists: {mainLogFolderPath}")
                                self._wMode_remove_mainLogFolder_done = True
                                self._instanceDebugPrint(
                                    f"    Set _wMode_remove_mainLogFolder_done to True.")
                            except Exception as e_rm_main_log_folder:
                                error_msg = f"ERROR_DynamicLogger ({id(self)}): Could not remove/recreate mainLogFolder '{mainLogFolderPath}': {e_rm_main_log_folder}"
                                print(error_msg, file=sys.stderr, flush=True)
                                self._instanceDebugPrint(f"    {error_msg}")
                        else:
                            self._instanceDebugPrint(
                                f"  Current 'w' mode call targets external path ({currentCallResolvedPath}). MainLogFolder not cleared at this step.")
                    else:  # _wMode_remove_mainLogFolder_done is True
                        self._instanceDebugPrint(
                            f"  MainLogFolder already processed for 'w' mode this session (_wMode_remove_mainLogFolder_done is True).")

                    # 2. Individual File Deletion (for files outside mainLogFolder, once per file per session)
                    resolvedAbsFilePathForWMode = currentCallResolvedPath.resolve()
                    isOutsideMainLogFolder = not str(resolvedAbsFilePathForWMode).startswith(
                        str(mainLogFolderPath))

                    if isOutsideMainLogFolder:
                        self._instanceDebugPrint(
                            f"  Resolved file path '{resolvedAbsFilePathForWMode}' is outside mainLogFolder '{mainLogFolderPath}'.")
                        if resolvedAbsFilePathForWMode not in self._wMode_removed_log_files_this_session:
                            self._instanceDebugPrint(
                                f"    Path '{resolvedAbsFilePathForWMode}' not in _wMode_removed_log_files_this_session set.")
                            if resolvedAbsFilePathForWMode.exists() and resolvedAbsFilePathForWMode.is_file():
                                try:
                                    resolvedAbsFilePathForWMode.unlink()
                                    self._instanceDebugPrint(
                                        f"      Successfully unlinked existing file: {resolvedAbsFilePathForWMode}")
                                except Exception as e_unlink_individual:
                                    error_msg_unlink = f"ERROR_DynamicLogger ({id(self)}): Could not unlink file '{resolvedAbsFilePathForWMode}' for 'w' mode: {e_unlink_individual}"
                                    print(error_msg_unlink, file=sys.stderr, flush=True)
                                    self._instanceDebugPrint(f"      {error_msg_unlink}")
                            else:
                                self._instanceDebugPrint(
                                    f"      File '{resolvedAbsFilePathForWMode}' does not exist or is not a file. No unlinking needed.")
                            self._wMode_removed_log_files_this_session.add(
                                resolvedAbsFilePathForWMode)
                            self._instanceDebugPrint(
                                f"      Added '{resolvedAbsFilePathForWMode}' to _wMode_removed_log_files_this_session set.")
                        else:
                            self._instanceDebugPrint(
                                f"    Path '{resolvedAbsFilePathForWMode}' already in _wMode_removed_log_files_this_session set. No action.")
                    # If inside mainLogFolder, its handling was covered by the mainLogFolder deletion logic (if applicable)
                else:  # No specific file path for this call (e.g. filePath=False was resolved, or PathResolver failed)
                    self._instanceDebugPrint(
                        f"  No resolved file path from PathResolver for 'w' mode file check (filePath was likely resolved to False or resolution failed).")
            else:
                self._instanceDebugPrint(
                    f"Effective fileWriteMode is '{effectiveFileWriteMode}' or filePath is False. Skipping 'w' mode deletions.")
            # --- End of fileWriteMode 'w' logic ---

            finalEffectiveMessageLevel = finalResolvedOptions['_finalEffectiveMessageLevel']
            self._instanceDebugPrint(
                f"Final effective message level for record: {finalEffectiveMessageLevel}")

            actualMessageArgsToUse = messageArgsTuple
            if not messageArgsTuple and finalResolvedOptions.get(
                    constants.ARG_MESSAGE_ARGS) is not None:
                msgArgsVal = finalResolvedOptions[constants.ARG_MESSAGE_ARGS]
                if not isinstance(msgArgsVal, tuple):
                    actualMessageArgsToUse = tuple(msgArgsVal) if isinstance(msgArgsVal,
                                                                             list) else (
                        msgArgsVal,)
                else:
                    actualMessageArgsToUse = msgArgsVal
            logMessageContent = message
            self._instanceDebugPrint(
                f"Log message content: '{logMessageContent}', Args for %: {actualMessageArgsToUse}")

            # If currentCallResolvedPath was not determined during 'w' mode processing
            # (e.g., fileWriteMode was 'a' or filePath was False earlier), resolve it now.
            if currentCallResolvedPath is None and finalResolvedOptions.get(
                    constants.ARG_FILE_PATH) is not False:
                self._instanceDebugPrint(
                    "Calling PathResolver.resolveFilePath (standard path resolution).")
                callerClassNameForPath = callerDetails.get(
                    "className") if callerDetails and not isSimpleLogModeEffective else None
                callerFuncNameForPath = callerDetails.get(
                    "funcName") if callerDetails and not isSimpleLogModeEffective else None
                currentCallResolvedPath = self._pathResolver.resolveFilePath(
                    filePathArg=finalResolvedOptions.get(constants.ARG_FILE_PATH),
                    callerClassName=callerClassNameForPath,
                    callerFuncName=callerFuncNameForPath,
                    isSimpleLogMode=isSimpleLogModeEffective
                )
                self._instanceDebugPrint(
                    f"PathResolver returned (standard path): {currentCallResolvedPath}")

            attributesForRecord = {}
            attributesForRecord[constants.RECORD_ATTR_IS_SIMPLE_LOG_MODE] = isSimpleLogModeEffective
            attributesForRecord[
                constants.RECORD_ATTR_LOG_FORMAT_FROM_ARG] = finalResolvedOptions.get(
                constants.ARG_LOG_FORMAT)
            attributesForRecord[
                constants.RECORD_ATTR_TIMESTAMP_FORMAT_FROM_ARG] = finalResolvedOptions.get(
                constants.ARG_TIMESTAMP_FORMAT)

            if isSimpleLogModeEffective:
                attributesForRecord["indicatorName"] = constants.PLACEHOLDER_NA
                attributesForRecord["callerInfo"] = constants.PLACEHOLDER_NA
            else:
                resolvedFmtIndicatorName = finalResolvedOptions.get(constants.ARG_INDICATOR_NAME,
                                                                    constants.PLACEHOLDER_NA)
                if resolvedFmtIndicatorName is None: resolvedFmtIndicatorName = constants.PLACEHOLDER_NA
                attributesForRecord["indicatorName"] = resolvedFmtIndicatorName

                attributesForRecord["callerInfo"] = callerDetails.get("funcMethIndicator",
                                                                      constants.PLACEHOLDER_NA) if callerDetails else constants.PLACEHOLDER_NA

            attributesForRecord[constants.RECORD_ATTR_RESOLVED_INDICATOR_NAME] = \
                attributesForRecord["indicatorName"]
            attributesForRecord[constants.RECORD_ATTR_RESOLVED_CALLER_INFO] = attributesForRecord[
                "callerInfo"]

            self._instanceDebugPrint(
                f"Attributes for record (to be merged into extra): {attributesForRecord}")

            finalExtraForRecord = finalResolvedOptions.get(constants.ARG_EXTRA, {}).copy()
            finalExtraForRecord.update(attributesForRecord)
            self._instanceDebugPrint(f"Final 'extra' for LogRecord: {finalExtraForRecord}")

            self._instanceDebugPrint("Calling HandlerManager.getActiveHandlers...")
            activeHandlers = self._handlerManager.getActiveHandlers(
                finalResolvedOptions=finalResolvedOptions,
                finalEffectiveMessageLevel=finalEffectiveMessageLevel,
                resolvedFilePath=currentCallResolvedPath,
                recordForFormattingHints=None
            )
            self._instanceDebugPrint(
                f"HandlerManager returned {len(activeHandlers)} active handlers: {activeHandlers}")

            if not activeHandlers:
                self._instanceDebugPrint("No active handlers. Aborting log dispatch.")
                return

            if isSimpleLogModeEffective or not callerDetails:
                pathname, lineno, funcName = constants.PLACEHOLDER_NA, 0, constants.PLACEHOLDER_NA
            else:
                pathname = callerDetails.get("pathname", constants.PLACEHOLDER_NA)
                lineno = callerDetails.get("lineno", 0)
                funcName = callerDetails.get("funcName", constants.PLACEHOLDER_NA)

            excInfoForRecord = finalResolvedOptions.get(constants.ARG_EXC_INFO)
            if excInfoForRecord is True and sys.exc_info() == (None, None, None):
                excInfoForRecord = None

            sinfo = None
            if finalResolvedOptions.get(constants.ARG_STACK_INFO, False):
                if 'stack_info' not in finalExtraForRecord or finalExtraForRecord['stack_info']:
                    finalExtraForRecord['stack_info'] = True

            self._instanceDebugPrint(
                f"Making LogRecord: name='{self._internalStdLogger.name}', level={finalEffectiveMessageLevel}, msg='{logMessageContent[:50]}...'")
            logRecord = self._internalStdLogger.makeRecord(
                name=self._internalStdLogger.name,
                level=finalEffectiveMessageLevel,
                fn=pathname,
                lno=lineno,
                msg=logMessageContent,
                args=actualMessageArgsToUse,
                exc_info=excInfoForRecord,
                func=funcName,
                extra=finalExtraForRecord
            )

            self._instanceDebugPrint(
                f"Dispatching LogRecord (level {logRecord.levelname}/{logRecord.levelno}, msg '{logRecord.getMessage()[:50]}...') to handlers.")
            for handler in activeHandlers:
                self._instanceDebugPrint(f"  Checking handler: {handler} (level {handler.level})")
                if logRecord.levelno >= handler.level:
                    self._instanceDebugPrint(
                        f"    Record level {logRecord.levelno} >= handler level {handler.level}. Calling handler.handle().")
                    try:
                        handler.handle(logRecord)
                        self._instanceDebugPrint(f"    Handler {handler} processed record.")
                    except Exception as e_handle:
                        print(
                            f"ERROR_DynamicLogger ({id(self)}): Exception during handler.handle() for {handler}: {e_handle}",
                            file=sys.stderr, flush=True)
                        self._instanceDebugPrint(f"    Error in handler {handler}: {e_handle}")
                else:
                    self._instanceDebugPrint(
                        f"    Record level {logRecord.levelno} < handler level {handler.level}. Skipping handler.")
            self._instanceDebugPrint("Log dispatch finished.")
        self._instanceDebugPrint("Lock released for log call.")
        self._instanceDebugPrint(f"--- Log call finished: msg='{message[:50]}...' ---")

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.log(constants.LOG_LEVEL_DEBUG, message, *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.log(constants.LOG_LEVEL_INFO, message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.log(constants.LOG_LEVEL_WARNING, message, *args, **kwargs)

    warn = warning

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.log(constants.LOG_LEVEL_ERROR, message, *args, **kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.log(constants.LOG_LEVEL_CRITICAL, message, *args, **kwargs)

    fatal = critical

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        current_exc_info = kwargs.get(constants.ARG_EXC_INFO)
        if current_exc_info is None:
            kwargs[constants.ARG_EXC_INFO] = True
        elif current_exc_info is False:
            pass
        self.log(constants.LOG_LEVEL_ERROR, message, *args, **kwargs)

    def updateHighOrderOptions(self, newHighOrderOptions: Dict[str, Any]) -> None:
        self._instanceDebugPrint(f"Updating highOrderOptions with: {newHighOrderOptions}")
        self._configManager.updateHighOrderOptions(newHighOrderOptions)

    def shutdown(self) -> None:
        self._instanceDebugPrint("Shutdown called for DynamicLogger instance.")
        with self._threadLock:
            self._handlerManager.shutdown()
        self._instanceDebugPrint("DynamicLogger instance shutdown complete.")
