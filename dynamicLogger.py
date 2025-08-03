# dynamicLogger.py
"""
==============================================================================
 DynamicLogger Class - Flexible and Configurable Logging
==============================================================================

For a comprehensive understanding of how to use this logger, its features,
and the priority system, please refer to the main documentation and examples
in `useDynamicLogger.py`.

------------------------------------------------------------------------------
Core Design Philosophy:
------------------------------------------------------------------------------
This logger is designed to provide a highly adaptable logging experience through
a layered configuration system. The key components are:

1.  Log Configuration Sets (`logConfigSets`):
    - Define various predefined "styles" or "profiles" for logging.
    - Each set can have multiple handlers (console, file) with their own
      formats, levels, and output destinations.
    - A `default` config set is used as a fallback.

2.  High-Order Options (`highOrderOptions`):
    - Allow for global or targeted overrides of logging behavior.
    - These options can be triggered by:
        - `indicatorName`: A specific string passed in a log call. This has
          the highest precedence within high-order options.
        - `funcMethIndicator`: An automatically determined string like
          "ClassName.methodName" or "functionName", allowing behavior
          customization for specific parts of the codebase.
    - They can control almost all aspects of a log message, including its
      destination, format, level, and even force a specific `logConfigSet`.

3.  Inline Options (Per-Call Arguments):
    - Provide the finest-grained control by allowing overrides directly in
      the logging call (e.g., `logger.info(msg, inlinePrintToConsole=False)`).

4.  Priority System:
    - The logger resolves the final settings for a log message by evaluating
      options in a strict priority:
      High-Order (IndicatorName) > High-Order (FuncMethIndicator) >
      Inline Argument > Log Config Set Handler Setting > Built-in Defaults.
    - Crucially, a `None` value from a higher priority level does *not*
      override a viable (non-None) value from a lower priority level. It
      signifies "no opinion," allowing the next level to take effect.

5.  Dynamic Behavior:
    - Caller information (`callerInfo`) is automatically captured.
    - Log file paths can be auto-generated relative to where the logger
      instance is created if not explicitly defined.
    - Message levels can be dynamically overridden, and a suppression
      mechanism prevents logging if an override significantly raises the
      message's effective severity beyond its original intent.
"""

import copy  # For deepcopy
import inspect
import logging
import os
import sys
import threading
from pathlib import Path


# --- Helper Functions ---
def getCallerInfo(depth=3):
    """
    Retrieves the class name (if applicable) and function/method name of the caller.
    This is crucial for `funcMethIndicator`-based High-Order Options and for
    populating `%(callerInfo)s` in log formats.

    Args:
        depth (int): How many frames to go up the stack to find the true caller.
                     Default is 3, which typically points to the user's code when
                     called via convenience methods like logger.info().

    Returns:
        tuple: (className, funcName)
               - className (str or None): Name of the class if caller is a method.
               - funcName (str or None): Name of the function or method.
                                         Can be an error string if inspection fails.
    """
    try:
        stack = inspect.stack()
        # Basic validation for stack and depth
        if not stack or depth <= 0:
            return None, f"<invalid_depth_{depth}>"

        effectiveDepth = min(depth, len(stack) - 1)  # Ensure depth isn't out of bounds

        # Handle edge cases for stack depth
        if effectiveDepth < 0: return None, "<stack_too_shallow_negative_depth>"
        if len(stack) <= effectiveDepth:  # If stack isn't deep enough for requested depth
            callerFuncName = stack[-1].function if stack else "<unknown_stack_empty>"
            return None, f"<stack_too_shallow_{len(stack)}_Req_{depth}_Eff_{effectiveDepth}_Last_{callerFuncName}>"

        frame = stack[effectiveDepth]
        callerFrame = frame.frame
        funcName = frame.function
        className = None

        # Determine if the caller is a method within a class
        if 'self' in callerFrame.f_locals:
            instance = callerFrame.f_locals['self']
            # Critical: Avoid identifying DynamicLogger's own methods as the caller
            if not isinstance(instance, DynamicLogger):
                if hasattr(instance, '__class__'):
                    className = instance.__class__.__name__
        elif 'cls' in callerFrame.f_locals:  # For class methods
            klass = callerFrame.f_locals['cls']
            if klass is not DynamicLogger and isinstance(klass, type):
                className = klass.__name__
        return className, funcName
    except Exception as e:
        # Fallback if any error occurs during inspection
        stackLength = len(inspect.stack()) if 'inspect' in globals() and hasattr(inspect,
                                                                                 'stack') else 'N/A'
        print(f"ERROR getting caller info (depth {depth}, stack_len {stackLength}): {e}",
              file=sys.stderr)
        return None, "<error_in_getCallerInfo>"
    finally:
        # Explicitly delete frame objects to help with garbage collection and prevent cycles
        if 'frame' in locals(): del frame
        if 'callerFrame' in locals(): del callerFrame
        if 'stack' in locals(): del stack


# --- Dynamic Logger Class ---
class DynamicLogger:
    """
    A highly configurable logger that supports dynamic behavior modification
    through log configuration sets, high-order options, and inline parameters.
    """
    defaultConfigSetName = 'default'  # Name of the fallback configuration set

    # Default log configurations. Users can provide their own or extend these.
    # All keys adhere to camelCase.
    defaultLogConfigSets = {
        defaultConfigSetName: {
            'logLevel': logging.DEBUG,  # Overall minimum level this config set supports
            'handlers': [
                {
                    'handlerType': 'console',
                    'logLevel': logging.INFO,  # Console handler only shows INFO and above
                    'logFormat': '%(levelname)-8s %(message)s',
                    'timestampFormat': None  # No timestamp for console by default
                },
                {
                    'handlerType': 'file',
                    'logLevel': logging.DEBUG,  # File handler logs DEBUG and above
                    'filePath': 'logs/default_app_log.log',  # Default path for this file handler
                    'logFormat': '%(asctime)s | %(levelname)-8s | [%(callerInfo)s:%(lineno)d] | %(message)s',
                    'timestampFormat': '%Y-%m-%d %H:%M:%S,%f'[:-3]
                    # Default timestamp format with milliseconds
                }
            ]
        },
        'simpleConsole': {  # A minimal console-only configuration
            'logLevel': logging.DEBUG,
            'handlers': [{'handlerType': 'console', 'logLevel': logging.DEBUG,
                          'logFormat': '%(levelname)s: %(message)s', 'timestampFormat': None}]
        },
        'detailedFileOnly': {  # For verbose file logging without console output
            'logLevel': logging.DEBUG,
            'handlers': [{
                'handlerType': 'file', 'logLevel': logging.DEBUG,
                'filePath': 'logs/detailed_log.log',
                'logFormat': '%(asctime)s|%(levelname)s|%(process)d|%(threadName)s|%(callerInfo)s:%(lineno)d|%(indicatorName)s|%(message)s',
                'timestampFormat': '%Y%m%d_%H%M%S_%f'  # Timestamp format for high-frequency logs
            }]
        },
        'audit': {  # Configuration for audit trails
            'logLevel': logging.INFO,
            'handlers': [
                {'handlerType': 'file', 'logLevel': logging.INFO,
                 'filePath': 'logs/audit_trail.log', 'logFormat': '%(asctime)s|AUDIT|%(message)s',
                 'timestampFormat': '%Y-%m-%d %H:%M:%S,%f'[:-3]},
                {'handlerType': 'console', 'logLevel': logging.WARNING,
                 'logFormat': 'AUDIT ALERT [%(levelname)s]: %(message)s', 'timestampFormat': None}
                # Audit alerts for important console messages
            ]
        },
        'errorLog': {  # Dedicated configuration for error logging
            'logLevel': logging.ERROR,
            'handlers': [
                {'handlerType': 'console', 'logLevel': logging.ERROR,
                 'logFormat': 'ERROR: %(message)s', 'timestampFormat': None},
                {'handlerType': 'file', 'logLevel': logging.ERROR, 'filePath': 'logs/errors.log',
                 'logFormat': '%(asctime)s | ERROR | [%(callerInfo)s:%(lineno)d] | %(message)s',
                 'timestampFormat': '%Y-%m-%d %H:%M:%S'}
            ]
        },
        'fileOnlyAutoPath': {  # Demonstrates auto-filePath generation
            'logLevel': logging.DEBUG,
            'handlers': [{
                'handlerType': 'file', 'logLevel': logging.DEBUG, 'filePath': None,
                # filePath: None triggers auto-generation
                'logFormat': 'AUTO-PATH [%(levelname)s] %(callerInfo)s: %(message)s',
                'timestampFormat': '%H:%M:%S'
            }]
        }
    }

    def __init__(self, name="DynamicLog", logConfigSets=None, highOrderOptions=None):
        """
        Initializes the DynamicLogger instance.

        Args:
            name (str): The name for this logger instance. This name is used by the
                        underlying Python logging system and can be part of auto-generated
                        log file names.
            logConfigSets (dict, optional): A dictionary of log configuration sets.
                                            If None, `defaultLogConfigSets` is used.
            highOrderOptions (dict, optional): A dictionary of high-order options
                                               for global or targeted overrides.
        """
        self.loggerName = name
        if logConfigSets is not None:
            self.logConfigSets = copy.deepcopy(logConfigSets)  # Ensure user's dict isn't modified
        else:
            self.logConfigSets = copy.deepcopy(self.defaultLogConfigSets)

        # Ensure the 'default' config set is always present and valid
        if self.defaultConfigSetName not in self.logConfigSets:
            self.logConfigSets[self.defaultConfigSetName] = copy.deepcopy(
                self.defaultLogConfigSets[self.defaultConfigSetName]
            )

        self.highOrderOptions = highOrderOptions if highOrderOptions is not None else {}

        # Determine basePath: The directory of the script that instantiated DynamicLogger.
        # This is crucial for resolving relative filePaths and auto-generating log paths.
        try:
            callerFrameInfo = inspect.stack()[1]  # Frame of the code that called __init__
            callerFile = callerFrameInfo.filename
            self.basePath = Path(os.path.dirname(os.path.abspath(callerFile)))
        except Exception as e:
            self.basePath = Path.cwd()  # Fallback to current working directory
            print(
                f"DynamicLogger Warning: Could not determine caller path for '{self.loggerName}', basePath set to CWD: {self.basePath} ({e})",
                file=sys.stderr)

        # Get or create the underlying Python logger
        self.logger = logging.getLogger(self.loggerName)
        # Set overall logger level to DEBUG. Actual filtering is done by handlers
        # based on their configured levels and the message's effective level.
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # Prevent messages from going to parent loggers

        self._handlerCache = {}  # Cache for handler instances to improve performance
        self._handlerLock = threading.Lock()  # Thread-safety for cache access

        # Clear any pre-existing handlers for this logger name.
        # This is important for environments where the logger might be re-initialized
        # (e.g., interactive sessions, test suites).
        if self.logger.hasHandlers():
            for handler in self.logger.handlers[:]:  # Iterate over a copy
                try:
                    handler.close()
                except Exception:
                    pass  # Ignore errors during cleanup
                self.logger.removeHandler(handler)

    def updateHighOrderOptions(self, highOrderOptions):
        """
        Allows runtime updates to the high-order options.
        Args:
            highOrderOptions (dict): The new set of high-order options.
        """
        self.highOrderOptions = highOrderOptions if highOrderOptions is not None else {}

    def _resolveFilePath(self, filePathToResolve):
        """
        Resolves a given filePath string or Path object to an absolute Path.
        If `filePathToResolve` is relative, it's resolved against `self.basePath`.
        Also ensures the parent directory for the log file exists.

        Args:
            filePathToResolve (str or Path): The file path to resolve.

        Returns:
            Path or None: The absolute Path object if successful, None otherwise.
        """
        if not filePathToResolve: return None
        pathObj = Path(filePathToResolve)
        # If path is already absolute, use it. Otherwise, combine with basePath.
        absolutePath = pathObj if pathObj.is_absolute() else self.basePath / pathObj
        try:
            # Ensure parent directory exists, creating it if necessary.
            absolutePath.parent.mkdir(parents=True, exist_ok=True)
            return absolutePath
        except OSError as e:  # Handles file system errors during directory creation
            print(
                f"ERROR (_resolveFilePath): Could not create directory for log path '{absolutePath}': {e}",
                file=sys.stderr)
            return None
        except Exception as eGeneral:  # Catch-all for other unexpected errors
            print(
                f"ERROR (_resolveFilePath): Unexpected issue resolving file path '{absolutePath}': {eGeneral}",
                file=sys.stderr)
            return None

    def _getHandlerInstance(self, handlerConfigKey):
        """
        Retrieves or creates a logging handler instance based on a configuration key.
        Handlers are cached using their base configuration to avoid redundant creation.
        The cache key includes the handler type, resolved destination path (for file handlers),
        base format, base timestamp format, and base level from the `logConfigSet`.

        Args:
            handlerConfigKey (tuple): A tuple uniquely identifying the handler's base config.
                                      (handlerType, destPathOrConsole, baseFmt, baseTsFmt, baseLevel)

        Returns:
            logging.Handler or None: The handler instance if successful, None otherwise.
        """
        with self._handlerLock:  # Ensure thread-safe access to the cache
            if handlerConfigKey in self._handlerCache:
                return self._handlerCache[handlerConfigKey]

            handlerType, destPathOrConsole, baseFmt, baseTsFmt, baseLevel = handlerConfigKey
            try:
                formatter = logging.Formatter(fmt=baseFmt, datefmt=baseTsFmt)
                handler = None
                if handlerType == 'console':
                    handler = logging.StreamHandler(sys.stdout)  # Console logs to standard output
                elif handlerType == 'file':
                    # destPathOrConsole for file handlers must be a resolved absolute path string
                    if not destPathOrConsole or destPathOrConsole == 'console':  # Should be caught earlier by filePath logic
                        print(
                            f"ERROR (_getHandlerInstance): File handler type but invalid destination '{destPathOrConsole}' for key {handlerConfigKey}",
                            file=sys.stderr)
                        return None
                    # File opened in append mode ('a') with UTF-8 encoding
                    handler = logging.FileHandler(str(destPathOrConsole), mode='a',
                                                  encoding='utf-8')

                if handler:
                    handler.setLevel(baseLevel)  # Handler's own minimum level
                    handler.setFormatter(formatter)
                    self._handlerCache[handlerConfigKey] = handler  # Store in cache
                    return handler
                else:  # Should not be reached if handlerType is validated before calling
                    print(
                        f"ERROR (_getHandlerInstance): Unsupported handler type '{handlerType}' for key {handlerConfigKey}",
                        file=sys.stderr)
                    return None
            except Exception as e:
                print(
                    f"ERROR (_getHandlerInstance): creating handler for key {handlerConfigKey}: {e}",
                    file=sys.stderr)
                return None

    def _getCallerAndIndicator(self, depthAdjust=0):
        """
        Determines the caller's information for `callerInfo` and `funcMethIndicator`.
        `depthAdjust` is used because this method is called from `log()`, which itself
        might be called from a convenience method (e.g., `logger.info()`).

        Args:
            depthAdjust (int): Adjusts how far up the stack to look.
                               0: called from convenience method (e.g., logger.info -> log -> this)
                              -1: called directly from user code (e.g., logger.log -> this)
        Returns:
            tuple: (callerInfoStr, funcMethIndicator, actualFuncNameFromUser)
        """
        # Base depth: UserCode -> Convenience -> log -> _getCallerAndIndicator -> getCallerInfo (depth=4)
        finalDepth = 4 + depthAdjust
        className, funcName = getCallerInfo(depth=finalDepth)

        # `callerInfoStr` is for the `%(callerInfo)s` format variable.
        callerInfoStr = f"{className}.{funcName}" if className else funcName
        # `funcMethIndicator` is used to match keys in `highOrderOptions`.
        # It should be a clean name, not an error string from getCallerInfo.
        funcMethIndicator = callerInfoStr if funcName and not (
                    "<" in funcName and ">" in funcName) else None
        # `actualFuncNameFromUser` is for the `LogRecord.funcName` attribute.
        actualFuncNameFromUser = funcName

        # Handle cases where getCallerInfo returned an error string
        if funcName is None or ("<" in funcName and ">" in funcName):
            callerInfoStr = funcName if funcName else "<unknown_caller_funcName_None>"  # For debugging logger errors
            actualFuncNameFromUser = funcName if funcName else "<unknown_funcName_None>"
        return callerInfoStr, funcMethIndicator, actualFuncNameFromUser

    def _getActiveHighOrderOptions(self, indicatorName, funcMethIndicator):
        """
        Retrieves the relevant high-order options dictionary.
        `indicatorName` (from log call) has precedence over `funcMethIndicator` (auto-detected).
        """
        if indicatorName and indicatorName in self.highOrderOptions:
            return self.highOrderOptions[indicatorName]
        if funcMethIndicator and funcMethIndicator in self.highOrderOptions:
            return self.highOrderOptions[funcMethIndicator]
        return {}  # No matching HO rule

    def _resolveConfigValue(self, optionName, highOrderOpts, inlineValue, handlerConfigFromSet,
                            defaultValue=None):
        """
        Resolves a configuration value based on the defined priority:
        High-Order > Inline > HandlerConfigFromSet > DefaultValue.
        Returns the first *non-None* (viable) value found.
        """
        hoValue = highOrderOpts.get(optionName)
        if hoValue is not None: return hoValue  # HO wins if value is not None

        if inlineValue is not None: return inlineValue  # Inline wins if value is not None

        if isinstance(handlerConfigFromSet, dict):  # Check if handler config is a dict
            configSetValue = handlerConfigFromSet.get(optionName)
            if configSetValue is not None: return configSetValue  # ConfigSet handler value wins

        return defaultValue  # Fallback to provided default

    def _prepareAndDispatchRecord(self, finalEffectiveMessageLevel, message, handlersToUseDetails,
                                  callerInfoStr, indicatorName, funcNameFromGetCaller,
                                  excInfo, stackInfo, extra, depthAdjust=0):
        """
        Creates a LogRecord with the `finalEffectiveMessageLevel` and dispatches it
        to the selected handlers, applying any per-call formatting overrides.
        """
        # Extra information for the LogRecord, including custom `callerInfo` and `indicatorName`.
        logExtra = {'callerInfo': callerInfoStr or "N/A", 'indicatorName': indicatorName or "N/A",
                    **(extra or {})}

        # Determine line number and filename of the user's logging call for the LogRecord.
        logLineNo = 0
        logFilePathName = "<unknown_file_path>"  # Corresponds to LogRecord.pathname (filename part)
        try:
            # Adjust depth to find the user's frame that initiated the log call.
            # Stack: UserCode -> (Convenience) -> log -> _prepareAndDispatchRecord -> currentframe
            # To UserCode from currentframe: 3 if via convenience, 2 if direct log() call.
            finalLinenoDepth = 3 + depthAdjust
            frame = inspect.currentframe()
            targetFrame = frame
            for _ in range(finalLinenoDepth):  # Go up the stack
                if targetFrame and targetFrame.f_back:
                    targetFrame = targetFrame.f_back
                else:
                    targetFrame = None; break  # Stack not deep enough
            if targetFrame:
                frameInfo = inspect.getframeinfo(targetFrame)
                logLineNo = frameInfo.lineno
                logFilePathName = Path(frameInfo.filename).name  # Just the filename part
            del targetFrame;
            del frame  # Clean up frames
        except Exception:  # Fallback if frame inspection fails
            logFilePathName = "<inspect_error_lineno_filename>"

        # Create the LogRecord.
        # Critically, `level=finalEffectiveMessageLevel` ensures the record reflects any overrides.
        # `name=self.loggerName` uses the logger's instance name, not the standard Python logger name.
        record = self.logger.makeRecord(
            name=self.loggerName,
            level=finalEffectiveMessageLevel,
            fn=logFilePathName, lno=logLineNo, msg=message, args=[],
            exc_info=excInfo, func=funcNameFromGetCaller or "<unknown_func>",
            # func is the function name
            extra=logExtra, sinfo=stackInfo
        )

        processedHandlers = set()  # To avoid processing a handler multiple times if config is unusual
        for handlerInstance, effectiveFormat, effectiveTsFormat in handlersToUseDetails:
            if handlerInstance in processedHandlers: continue

            originalFormatter = handlerInstance.formatter  # Store original formatter
            formatterChanged = False
            try:
                # Temporarily change formatter if per-call format or timestampFormat is different.
                currentFmt = getattr(originalFormatter, '_fmt',
                                     None)  # Access internal _fmt; somewhat fragile
                currentTsFmt = originalFormatter.datefmt
                if (effectiveFormat != currentFmt or effectiveTsFormat != currentTsFmt):
                    tempFormatter = logging.Formatter(fmt=effectiveFormat,
                                                      datefmt=effectiveTsFormat)
                    handlerInstance.setFormatter(tempFormatter)
                    formatterChanged = True

                handlerInstance.handle(record)  # Dispatch the record through the handler
            except Exception as handlerError:  # Catch errors during the .handle() call
                print(
                    f"ERROR (_prepareAndDispatchRecord): handling by {handlerInstance}: {handlerError}",
                    file=sys.stderr)
                import traceback;
                traceback.print_exc(file=sys.stderr)
            finally:
                # Restore the original formatter if it was temporarily changed.
                if formatterChanged: handlerInstance.setFormatter(originalFormatter)
                processedHandlers.add(handlerInstance)

    # The main logging method. All convenience methods (debug, info, etc.) call this.
    def log(self, initialMessageLevel, message,
            # Configuration and identification arguments:
            inlineConfigSetName=None, indicatorName=None,
            # Destination control arguments:
            inlinePrintToConsole=None, inlineWriteToFile=None, inlineFilePath=None,
            # Formatting control arguments:
            inlineIncludeTimestamp=None, inlineTimestampFormat=None, inlineLogFormat=None,
            # Behavior control arguments:
            inlineExclude=None, inlineOverrideLevel=None,
            # Standard logging arguments:
            excInfo=False, stackInfo=False, extra=None,
            # Internal argument for call stack depth adjustment:
            depthAdjust=0):  # 0 for convenience methods, -1 for direct log() call by user
        try:
            # --- 1. Get Caller Information and Active High-Order Options ---
            callerInfoStr, funcMethIndicator, actualFuncNameFromUser = self._getCallerAndIndicator(
                depthAdjust=depthAdjust)
            activeHoOpts = self._getActiveHighOrderOptions(indicatorName, funcMethIndicator)

            # --- 2. Exclusion Check (HO > Inline) ---
            # If 'exclude' is True from HO or Inline, the message is completely dropped.
            hoExclude = activeHoOpts.get('exclude')
            finalExclude = hoExclude if hoExclude is not None else (
                inlineExclude if inlineExclude is not None else False)
            if finalExclude: return  # Message excluded, stop processing.

            # --- 3. Determine Effective Message Level and Apply Suppression Logic ---
            # Priority for level override: HO (`level`) > Inline (`inlineOverrideLevel`) > Initial (`initialMessageLevel`)
            hoLevelOverride = activeHoOpts.get('level')
            levelFromHoOrInline = hoLevelOverride if hoLevelOverride is not None else inlineOverrideLevel

            finalEffectiveMessageLevel = levelFromHoOrInline if levelFromHoOrInline is not None else initialMessageLevel

            # Validate the determined level (e.g., ensure it's a valid integer)
            if not isinstance(finalEffectiveMessageLevel, int) or finalEffectiveMessageLevel < 0:
                print(
                    f"Warning (log): Invalid finalEffectiveMessageLevel '{finalEffectiveMessageLevel}'. Defaulting to initial {initialMessageLevel}.",
                    file=sys.stderr)
                finalEffectiveMessageLevel = initialMessageLevel

            # Suppression Rule: If an override (HO or Inline) specified a level,
            # AND the original message level was numerically lower than this override,
            # then the message is suppressed. This prevents, e.g., a DEBUG message from
            # being "promoted" to INFO if an override sets level=INFO, unless it was already INFO or higher.
            if levelFromHoOrInline is not None and initialMessageLevel < levelFromHoOrInline:
                return  # Suppress message.

            # Final check: Is the logger instance itself configured to handle this effective level?
            # This is usually true if self.logger.setLevel(logging.DEBUG) is used in __init__.
            if not self.logger.isEnabledFor(finalEffectiveMessageLevel): return

            # --- 4. Resolve Log Configuration Set ---
            # Priority: HO (`configSetName`) > Inline (`inlineConfigSetName`) > Default (`self.defaultConfigSetName`)
            hoForcedSetName = activeHoOpts.get('configSetName')
            resolvedConfigSetName = hoForcedSetName if hoForcedSetName is not None else \
                (
                    inlineConfigSetName if inlineConfigSetName is not None else self.defaultConfigSetName)

            currentConfigSet = self.logConfigSets.get(resolvedConfigSetName)
            # Fallback to default if the resolved set name is not found or invalid
            if not currentConfigSet:
                print(
                    f"Warning (log): ConfigSet '{resolvedConfigSetName}' not found. Using '{self.defaultConfigSetName}'.",
                    file=sys.stderr)
                resolvedConfigSetName = self.defaultConfigSetName
                currentConfigSet = self.logConfigSets.get(self.defaultConfigSetName,
                                                          {})  # Get default or empty dict

            # If no handlers in the chosen config set, cannot log.
            if not currentConfigSet or not currentConfigSet.get('handlers'): return

            # --- 5. Iterate Through Handlers in the Chosen Config Set ---
            handlersToDispatchToDetails = []  # List to store (handlerInstance, effectiveFormat, effectiveTsFormat)
            for handlerIndex, handlerConfigFromSet in enumerate(
                    currentConfigSet.get('handlers', [])):
                handlerType = handlerConfigFromSet.get('handlerType')
                handlerBaseLevel = handlerConfigFromSet.get('logLevel',
                                                            logging.DEBUG)  # Handler's own capability

                # Message's effective level must meet or exceed the handler's base level.
                if finalEffectiveMessageLevel < handlerBaseLevel:
                    continue  # Skip this handler for this message.

                # --- Resolve all specific options for this handler instance for THIS log call ---
                # Using _resolveConfigValue for each option ensures correct priority.

                # printToConsole: Default True for 'console' type, None otherwise (meaning no strong opinion)
                effPrintToConsole = self._resolveConfigValue('printToConsole', activeHoOpts,
                                                             inlinePrintToConsole,
                                                             handlerConfigFromSet,
                                                             defaultValue=(
                                                                 True if handlerType == 'console' else None))

                # writeToFile: Can be bool (True/False) or a path string/Path object.
                # Default True for 'file' type to indicate an attempt to write.
                effWriteToFileConfig = self._resolveConfigValue('writeToFile', activeHoOpts,
                                                                inlineWriteToFile,
                                                                handlerConfigFromSet,
                                                                defaultValue=(
                                                                    True if handlerType == 'file' else None))

                effFilePath = None  # The actual file path string to be used for this call
                absoluteResolvedFilePathStr = None  # The absolute path string for cache key and FileHandler
                determinedWriteToFileFlag = False  # Master flag: should this log go to a file?

                if handlerType == 'file':
                    # Determine if writing is intended based on effWriteToFileConfig
                    if isinstance(effWriteToFileConfig, (str, Path)):  # Path explicitly provided
                        determinedWriteToFileFlag = True
                        effFilePath = str(effWriteToFileConfig)  # This path will be used
                    elif effWriteToFileConfig is True:  # Explicitly True (use path from lower prio or auto-gen)
                        determinedWriteToFileFlag = True
                        # filePath itself also follows priority: HO > Inline > HandlerConfigSet
                        effFilePath = self._resolveConfigValue('filePath', activeHoOpts,
                                                               inlineFilePath, handlerConfigFromSet,
                                                               defaultValue=None)
                    elif effWriteToFileConfig is False:  # Explicitly False
                        determinedWriteToFileFlag = False
                    # If effWriteToFileConfig was None, determinedWriteToFileFlag remains False, but
                    # for file handlers, the defaultValue=True for effWriteToFileConfig means it will be True.

                    if determinedWriteToFileFlag:  # Only proceed if writing to a file is intended
                        if effFilePath is None:  # No filePath specified from HO, Inline, or ConfigSet
                            # Auto-generate a default file path.
                            # Pattern: logs/{loggerName}_{configSetName}_h{handlerIndex}_auto.log
                            # This path is relative to self.basePath.
                            defaultLogDir = "logs"  # Default subdirectory for auto-generated logs
                            # Use self.loggerName (from __init__) for unique logger instance files.
                            defaultFileName = f"{self.loggerName}_{resolvedConfigSetName}_h{handlerIndex}_auto.log"
                            effFilePath = str(Path(defaultLogDir) / defaultFileName)

                        if effFilePath:  # If a path string was determined (explicit or auto-generated)
                            resolvedPathObj = self._resolveFilePath(
                                effFilePath)  # Resolve to absolute, create dirs
                            if resolvedPathObj:
                                absoluteResolvedFilePathStr = str(resolvedPathObj)
                            else:  # Path resolution failed (e.g., permissions)
                                determinedWriteToFileFlag = False  # Cannot write
                                print(
                                    f"Warning (log): Failed to resolve filePath '{effFilePath}' for file handler. Disabling write for this call.",
                                    file=sys.stderr)
                        else:  # Should be rare if auto-generation works, but a fallback.
                            determinedWriteToFileFlag = False  # No path, cannot write
                            print(
                                f"Warning (log): No filePath determined for file handler. Disabling write for this call.",
                                file=sys.stderr)

                # Formatting options
                effIncludeTimestamp = self._resolveConfigValue('includeTimestamp', activeHoOpts,
                                                               inlineIncludeTimestamp,
                                                               handlerConfigFromSet,
                                                               defaultValue=None)
                effTimestampFormat = self._resolveConfigValue('timestampFormat', activeHoOpts,
                                                              inlineTimestampFormat,
                                                              handlerConfigFromSet,
                                                              defaultValue=handlerConfigFromSet.get(
                                                                  'timestampFormat'))
                effLogFormat = self._resolveConfigValue('logFormat', activeHoOpts, inlineLogFormat,
                                                        handlerConfigFromSet,
                                                        defaultValue=handlerConfigFromSet.get(
                                                            'logFormat', '%(message)s'))

                # Harmonize timestampFormat based on includeTimestamp
                if effIncludeTimestamp is False:  # Explicitly exclude timestamp
                    effTimestampFormat = None
                elif effIncludeTimestamp is True:  # Explicitly include timestamp
                    if effTimestampFormat is None:  # If no format is given, try handler's, then hard default
                        effTimestampFormat = handlerConfigFromSet.get('timestampFormat')
                        if effTimestampFormat is None:
                            effTimestampFormat = '%Y-%m-%d %H:%M:%S,%f'[:-3]  # Sensible default
                elif effIncludeTimestamp is None:  # Not specified by HO/Inline/Set
                    # If effTimestampFormat has a value (from HO/Inline/Set), timestamp will be included.
                    # If effTimestampFormat is also None, it means no timestamp unless the logFormat implies it
                    # or the base handler config had a timestampFormat (which would be in effTimestampFormat already).
                    # The standard logging.Formatter handles datefmt=None by not including asctime.
                    pass

                    # --- Final Decision to Process This Handler for This Log Call ---
                shouldProcessThisHandler = False
                cacheDestKey = None  # Part of the handler cache key; 'console' or absolute file path

                if handlerType == 'console':
                    cacheDestKey = 'console'
                    if effPrintToConsole is True:  # If True (explicitly or by default for console type)
                        shouldProcessThisHandler = True
                elif handlerType == 'file':
                    cacheDestKey = absoluteResolvedFilePathStr  # Must be the resolved absolute path
                    if determinedWriteToFileFlag and absoluteResolvedFilePathStr:  # Must intend to write AND have a valid path
                        shouldProcessThisHandler = True

                if shouldProcessThisHandler:
                    # Construct cache key based on the handler's *original* configuration from the logConfigSet,
                    # plus the uniquely resolved destination for file handlers.
                    # This ensures that if the same base handler config is used for different dynamic file paths,
                    # new handler instances are created.
                    handlerCacheKey = (
                        handlerType,
                        cacheDestKey,  # 'console' or the absolute resolved file path string
                        handlerConfigFromSet.get('logFormat', '%(message)s'),
                        # Base format from config
                        handlerConfigFromSet.get('timestampFormat'),  # Base ts format from config
                        handlerBaseLevel  # Base level from config
                    )
                    handlerInstance = self._getHandlerInstance(handlerCacheKey)
                    if handlerInstance:
                        # Add to list for dispatching, along with the effective formats for THIS call.
                        handlersToDispatchToDetails.append(
                            (handlerInstance, effLogFormat, effTimestampFormat))

            # --- 6. Dispatch the Record if any handlers were selected ---
            if handlersToDispatchToDetails:
                self._prepareAndDispatchRecord(
                    finalEffectiveMessageLevel, message, handlersToDispatchToDetails,
                    callerInfoStr, indicatorName, actualFuncNameFromUser,
                    excInfo, stackInfo, extra, depthAdjust=depthAdjust
                )
        except Exception as logError:  # Catch-all for unexpected errors within the log() method itself
            print(
                f"CRITICAL LOGGING SYSTEM ERROR in DynamicLogger.log() ({type(logError).__name__}): {logError}",
                file=sys.stderr)
            import traceback;
            traceback.print_exc(file=sys.stderr)

    # --- Convenience Methods (debug, info, warning, error, critical) ---
    # These simply call the main `log` method with the appropriate `initialMessageLevel`
    # and `depthAdjust=0` because they add one level to the call stack before reaching `log`.

    def debug(self, message, **kwargs):
        self.log(logging.DEBUG, message, depthAdjust=0, **kwargs)

    def info(self, message, **kwargs):
        self.log(logging.INFO, message, depthAdjust=0, **kwargs)

    def warning(self, message, **kwargs):
        self.log(logging.WARNING, message, depthAdjust=0, **kwargs)

    def error(self, message, excInfo=True,
              **kwargs):  # excInfo defaults to True for error and critical
        self.log(logging.ERROR, message, excInfo=excInfo, depthAdjust=0, **kwargs)

    def critical(self, message, excInfo=True, **kwargs):
        self.log(logging.CRITICAL, message, excInfo=excInfo, depthAdjust=0, **kwargs)