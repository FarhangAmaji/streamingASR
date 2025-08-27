## DynamicLogger: Project Description

### Level 1: Project Overview (Big Picture)

**What is DynamicLogger?**

DynamicLogger is an advanced and adaptable Python logging framework designed to offer highly granular and dynamic control over logging behavior. It extends Python's standard `logging` module, providing a flexible system for developers to manage how, when, and where log messages are recorded, even at runtime without application restarts.

**Core Purpose & Vision:**

The primary vision behind DynamicLogger is to empower developers with a sophisticated tool for contextual logging, especially in complex applications and during debugging in production-like environments. It aims to allow fine-tuning of log output (destination, format, level, file writing mode) for specific code sections or individual log calls dynamically, minimizing the need to modify the core application code for temporary logging adjustments.

**Key Implementation Approach:**

*   **Platform & Environment:** DynamicLogger is a pure Python library, intended to run on any platform that supports Python (typically 3.8+ due to modern typing, `pathlib` usage, and `shutil` features). It has no external dependencies beyond the Python standard library.
*   **Foundation:** It builds upon the standard Python `logging` module, leveraging core components like `LogRecord`, `Handler`, and `Formatter`.
*   **Dynamic Configuration Core:** The central mechanism is a **Prioritized Configuration Chain**. Log call behavior is determined by resolving a set of predefined arguments (`All Main Log Args`) through a strict hierarchy of configuration sources. These sources include High-Order Options (global overrides matched by indicators), inline arguments passed directly to log functions, and initialization-time settings for the logger instance.
*   **Contextual Awareness:** The logger automatically detects information about the calling code (e.g., class and method name via `funcMethIndicator`) to facilitate rule matching for High-Order Options and for use in log message formatting or default file naming.
*   **Modularity:** The codebase is structured into multiple Python files organized into `dynamicLoggerCore` and `utils` directories. Each module has a focused responsibility, such as argument processing (`ArgumentProcessor`), path resolution (`PathResolver`), or handler management (`HandlerManager`), promoting maintainability and testability.

**High-Level Features for Users:**

*   **Runtime Log Control:** Change logging behavior for specific parts of your code or specific log calls without application restarts using High-Order Options, triggered by `indicatorName` or automatically detected `funcMethIndicator`.
*   **Flexible Output:** Easily direct logs to the console (`sys.stdout`), specific files, or both. File paths can be absolute, relative to a determined project `basePath`, or simple names resulting in files within a default `logs` subdirectory.
*   **Customizable File Writing Modes:** Choose between append (`'a'`) or a custom overwrite (`'w'`, default) mode for file logging. The `'w'` mode provides a "fresh start" by clearing the main log directory or specific external log files once per logger session when first encountered.
*   **Customizable Formatting:** Control log message structure using standard Python logging format strings. A detailed default format is provided, which can be overridden or simplified (even to just the raw message via `logFormat=False`). Timestamps can be customized or omitted.
*   **Automatic Caller Info:** Log messages can automatically include the calling function/method name (`funcMethIndicator` available as `%(callerInfo)s`) and line number (`%(lineno)d`).
*   **Granular Overrides:** Globally configured or instance-default settings can be overridden for individual log calls using inline keyword arguments.
*   **"Simple Log" Mode:** An optional, performance-oriented mode for specific log calls (`simpleLog=True` passed inline). This mode bypasses High-Order Option resolution and detailed caller info detection, offering a potentially faster logging path for high-throughput scenarios.
*   **Standard Logging Integration:** Works seamlessly with Python's `logging` ecosystem, using standard `LogRecord`, `Handler`, and `Formatter` objects internally.

---

### Level 2: Detailed Features, Behaviors, and Edge Case Handling (For Developers & Advanced Users)

This level delves into the specifics of how DynamicLogger works, its expected behaviors, and how it handles various situations.

**I. Key Definitions (Crucial for Understanding DynamicLogger):**

*   **`All Log Funcs`**: Refers to the primary logging methods available on a `DynamicLogger` instance: `log(level, message, *args, **kwargs)`, and convenience methods `debug()`, `info()`, `warning()`, `error()`, `critical()`, and `exception(message, *args, **kwargs)` (which implies `excInfo=True`).
*   **`All Main Log Args`**: A specific set of keyword arguments that control logging behavior. These arguments have **unified names** across all configuration points. The set includes:
    1.  `configSet`: A Python dictionary used to bundle other log arguments, offering a way to group reusable settings.
    2.  `indicatorName`: A user-defined string (cannot be empty if provided) to tag log calls or logger instances. It's primarily used for matching High-Order Option rules and can be included in log formats.
    3.  `printToConsole`: A boolean. If `True`, the log message is output to `sys.stdout`.
    4.  `filePath`: Specifies the file output path. It can be a `pathlib.Path` object, a string (absolute, relative to `basePath`, or a "simple name"), `True` (for default file logging behavior), or `False` (to disable file logging for the call).
    5.  `fileWriteMode`: A string, either `'w'` (default) or `'a'`. Controls how file logs are written. See detailed behavior below.
    6.  `exclude`: A boolean. If `True`, the log message is completely suppressed, and no further processing occurs for that log call (short-circuit).
    7.  `logFormat`: Defines the log message structure.
        *   `str`: A Python `logging` format string.
        *   `True` (default): Uses a predefined detailed template (see below).
        *   `False`: Outputs only the raw formatted message content, overriding any timestamp settings.
    8.  `timestampFormat`: Controls the timestamp prepended to the log message.
        *   `str`: A `strftime` compatible format string.
        *   `True` (default): Uses a default timestamp format (`"%Y-%m-%d %H:%M:%S"`).
        *   `False`: No timestamp is prepended to the log message.
    9.  `excInfo`: Standard Python `logging` behavior for including exception traceback information. Can be `True`, `False`, an exception tuple, or an exception instance.
    10. `extra`: A dictionary of user-defined key-value pairs to be added as attributes to the `LogRecord`.
    11. `logLevel`: An integer (e.g., `logging.INFO`) that, if specified, overrides the log level implied by the specific log function called (e.g., `logger.debug()`). This sets the level of the `LogRecord` itself.
    12. `simpleLog`: A boolean. If `True` **and passed directly inline** to one of `All Log Funcs`, it activates a simplified, performance-optimized logging path for that specific call.
    13. `stackInfo`: A boolean. If `True`, general call stack information (not just for exceptions) is added to the log record (typically populating `record.stack_info`).
    14. `messageArgs`: A tuple. If positional `*args` are not provided to `All Log Funcs`, this tuple is used for `%`-style string formatting with the main `message` string.
*   **`Args Entry Points`**: The distinct places where a user can provide values for `All Main Log Args`:
    1.  **High-Order Options (HOs)**: Centrally defined override rules. An HO rule body is a dictionary containing direct log arguments and/or an optional `configSet` key. Rules are matched by `indicatorName` or `funcMethIndicator`.
    2.  **Inline Arguments**: Arguments passed directly to one of `All Log Funcs` via named keyword arguments.
    3.  **DynamicLogger Initialization (Init)**: Arguments passed to the `DynamicLogger` constructor (`__init__`), either directly as named parameters or via a `configSet` parameter.
*   **`Priority Chain`**: The strict order for resolving `All Main Log Args` (highest to lowest):
    1.  HO triggered by `indicatorName` (Direct args from the HO rule body).
    2.  HO triggered by `indicatorName` (Args from a `configSet` key within that HO rule body).
    3.  HO triggered by `funcMethIndicator` (Direct args from the HO rule body).
    4.  HO triggered by `funcMethIndicator` (Args from a `configSet` key within that HO rule body).
    5.  Inline Arguments (Directly passed as named parameters to `All Log Funcs`).
    6.  Inline Arguments (Args from a `configSet` dictionary passed inline to `All Log Funcs`).
    7.  DynamicLogger Initialization (Directly passed as named parameters to `__init__`).
    8.  DynamicLogger Initialization (Args from a `configSet` dictionary passed to `__init__`).
    9.  `Final Defaults` (Ultimate hardcoded fallback values).
*   **`configSet` (as an argument type)**: A standard Python dictionary. It can contain any of `All Main Log Args` as keys (except for `configSet` itself). It acts as a reusable bundle of settings.
*   **`funcMethIndicator`**: An automatically detected string representing the caller of the logger (e.g., `ClassName.methodName` or `functionName`). This is used for matching High-Order Option rules and for default file path generation when `filePath=True`. It's available in format strings as `%(callerInfo)s`.
*   **`onLevelDefault`**: For each of `All Main Log Args`, this is a predefined value (often `None`) that signifies "this configuration level does not have an explicit setting for this argument; its opinion is to pass through and let lower priority levels decide." If a configuration level *explicitly sets* an argument to its `onLevelDefault` value, it's an active decision to use that passthrough behavior.
*   **`finalDefault`**: For each of `All Main Log Args`, this is the ultimate fallback value used if no higher-priority source in the `Priority Chain` provides an active setting for that argument.
*   **`basePath`**: An automatically determined absolute path to a directory, primarily based on the instantiation site of the `DynamicLogger` instance, with fallbacks to the DynamicLogger library root, main script directory, or CWD. It serves as the base for resolving relative `filePath` arguments and for placing the `mainLogFolder`.
*   **`mainLogFolder`**: A directory typically named `logs` (configurable via `constants.DEFAULT_LOGS_DIR_NAME`) located directly under the logger's `basePath`. This is the default location for many log files.

**II. Core Argument Behaviors & Edge Cases (`All Main Log Args`):**

1.  **`configSet` (`dict` | `None`)**
    *   **`onLevelDefault`**: `None`, **`finalDefault`**: `None`.
    *   **Behavior**: (As before)

2.  **`indicatorName` (`str` | `None`)**
    *   **`onLevelDefault`**: `None`, **`finalDefault`**: `None`.
    *   **Behavior**: (As before)
    *   **Formatting**: (As before)

3.  **`printToConsole` (`bool` | `None`)**
    *   **`onLevelDefault`**: `None`, **`finalDefault`**: `True`.
    *   **Behavior**: (As before)

4.  **`filePath` (`pathlib.Path` | `str` | `bool` | `None`)**
    *   **`onLevelDefault`**: `None`, **`finalDefault`**: `True`.
    *   **Behavior**: (As before)
    *   **Path Handling**: (As before)

5.  **`fileWriteMode` (`str` \[`'w'`, `'a'`] | `None`)**
    *   **`onLevelDefault`**: `None`, **`finalDefault`**: `'w'`.
    *   **Behavior**: Controls how file logs are written. The underlying `logging.FileHandler` always opens files in append mode (`'a'`). DynamicLogger's `fileWriteMode` adds behavior *before* the handler writes.
        *   `'w'` (Write/Overwrite - Default):
            *   **Main Log Folder (`basePath/logs/`):** If a log call resolves to `fileWriteMode='w'` and targets a file within the `mainLogFolder`, and this folder hasn't been cleared yet in the current `DynamicLogger` instance's session:
                1.  The entire `mainLogFolder` is removed and then recreated as an empty directory.
                2.  An internal flag (`_wMode_remove_mainLogFolder_done`) is set to `True` to prevent this folder from being cleared again by subsequent `'w'` mode calls for this logger instance.
            *   **Individual External Files (outside `mainLogFolder`):** If a log call resolves to `fileWriteMode='w'` and targets a specific file *outside* the `mainLogFolder`, and that specific file hasn't been cleared yet in the current session for this logger instance:
                1.  The target log file is deleted if it exists.
                2.  The file's path is added to an internal set (`_wMode_removed_log_files_this_session`) to prevent it from being deleted again by subsequent `'w'` mode calls to the *same file path* for this logger instance.
            *   If `filePath` resolves to `False`, no deletion actions occur for `'w'` mode.
        *   `'a'` (Append): No files or folders are deleted by DynamicLogger. Log messages are simply appended to the target file (which will be created if it doesn't exist by the underlying `FileHandler`).
    *   **Validation**: If an invalid string is provided, a warning is issued, and it defaults to `'w'`.

6.  **`exclude` (`bool` | `None`)**
    *   **`onLevelDefault`**: `None`, **`finalDefault`**: `False`.
    *   **Behavior**: (As before)

7.  **`logFormat` (`str` | `bool` | `None`)**
    *   **`onLevelDefault`**: `None`, **`finalDefault`**: `True`.
    *   **Behavior**: (As before)

8.  **`timestampFormat` (`str` | `bool` | `None`)**
    *   **`onLevelDefault`**: `None`, **`finalDefault`**: `True`.
    *   **Behavior**: (As before)

9.  **`excInfo` (`bool` | `tuple` | `BaseException` | `None`)**
    *   **`onLevelDefault`**: `None`, **`finalDefault`**: `False`.
    *   **Behavior**: (As before)

10. **`extra` (`dict` | `None`)**
    *   **`onLevelDefault`**: `None`, **`finalDefault`**: `{}` (a new empty dictionary).
    *   **Behavior**: (As before)

11. **`logLevel` (`int` | `None`)**
    *   **`onLevelDefault`**: `None`, **`finalDefault`**: `None`.
    *   **Behavior**: (As before)

12. **`simpleLog` (`bool` | `None`)**
    *   **`onLevelDefault`**: `None`, **`finalDefault`**: `False`.
    *   **Behavior**: (As before)

13. **`stackInfo` (`bool` | `None`)**
    *   **`onLevelDefault`**: `None`, **`finalDefault`**: `False`.
    *   **Behavior**: (As before)

14. **`messageArgs` (`tuple` | `None`)**
    *   **`onLevelDefault`**: `None`, **`finalDefault`**: `()` (empty tuple).
    *   **Behavior**: (As before)

**III. Handling of Invalid Inputs & System Errors:**

*   **Invalid Keys in `configSet` or HOs**: (As before)
*   **Empty `indicatorName` String**: (As before)
*   **`indicatorName` or `simpleLog` in HO Rules**: (As before)
*   **Misplaced `simpleLog=True`**: (As before)
*   **Invalid `logLevel` Value**: (As before)
*   **Invalid `logFormat` or `timestampFormat` Values**: (As before)
*   **Invalid `fileWriteMode` Value**: A warning is issued, and the system falls back to the default (`'w'`).
*   **File Path Errors / File/Folder Deletion Errors for `fileWriteMode='w'`**: `PathResolver` attempts to handle path creation errors gracefully. If `DynamicLogger` encounters errors during the deletion operations for `fileWriteMode='w'` (e.g., permissions issues when trying to delete `mainLogFolder` or an individual file), a warning is printed to `stderr`, and the logger continues (the "overwrite" effect might be incomplete for that specific operation, but logging to the file will proceed in append mode).
*   **Message Formatting `TypeError`**: (As before)
*   **Internal System Errors**: (As before)

---

### Level 3: Implementation Highlights & Code Structure (For Developers)

This level provides insights into the internal architecture and design choices of DynamicLogger.

**I. Code Organization (Folders & Key Modules):**

The project is organized into two main directories under the `dynamicLogger` project root:

*   **`dynamicLoggerCore/`**: Contains the core classes and logic for the logger's operation.
    *   **`dynamicLogger.py` (`DynamicLogger` class)**:
        *   This is the primary public interface. Users instantiate this class.
        *   It handles the `__init__` method, processing initial configuration arguments.
        *   Initializes internal state flags for `fileWriteMode='w'` behavior (`_wMode_remove_mainLogFolder_done`, `_wMode_removed_log_files_this_session`).
        *   It exposes the `All Log Funcs` (`debug`, `info`, `log`, etc.). The core `log()` method is the central dispatcher.
        *   **Key Responsibilities in `log()`:**
            1.  Manages overall thread safety for a log call.
            2.  Determines if `simpleLog` mode is active.
            3.  Collects all inline arguments.
            4.  If not in `simpleLog` mode, invokes `CallerInspector`.
            5.  Invokes `ArgumentProcessor` to resolve effective values for `All Main Log Args` (including `fileWriteMode`).
            6.  Checks for `exclude=True`.
            7.  **Implements `fileWriteMode='w'` logic:**
                *   If effective `fileWriteMode` is `'w'` and `filePath` is not `False`:
                    *   Determines the `mainLogFolder` path.
                    *   Resolves the `currentCallResolvedPath` for the current log call.
                    *   If `currentCallResolvedPath` targets the `mainLogFolder` and `_wMode_remove_mainLogFolder_done` is `False`, it removes and recreates `mainLogFolder` and sets the flag.
                    *   If `currentCallResolvedPath` targets a file *outside* `mainLogFolder` and this file is not in `_wMode_removed_log_files_this_session`, it deletes the file and adds it to the set.
            8.  Determines `finalEffectiveMessageLevel`.
            9.  Invokes `PathResolver` (if path not already resolved during 'w' mode checks) to get the final `resolvedFilePath`.
            10. Prepares attributes for the `LogRecord`.
            11. Invokes `HandlerManager` to get `activeHandlers`.
            12. If active handlers exist, creates and dispatches the `LogRecord`.
        *   Manages `highOrderOptions` updates.
        *   Provides a `shutdown()` method.
    *   **`argumentProcessor.py` (`ArgumentProcessor` class)**:
        *   (As before, now also resolves `fileWriteMode` and validates its resolved value).
    *   **`configManager.py` (`ConfigManager` class)**:
        *   (As before).
    *   **`callerInspector.py` (`CallerInspector` class)**:
        *   (As before).
    *   **`handlerManager.py` (`HandlerManager` class)**:
        *   (As before). Critically, `logging.FileHandler` instances are always created with `mode='a'` (append), as DynamicLogger's `'w'` mode is implemented via pre-emptive deletion.
    *   **`pathResolver.py` (`PathResolver` class)**:
        *   (As before). `DynamicLogger` uses its `basePath` property to determine the `mainLogFolder`.
    *   **`customFormatter.py` (`DynamicFormatter` class)**:
        *   (As before).
*   **`utils/`**:
    *   **`constants.py`**: (Updated with `ARG_FILE_WRITE_MODE` and its defaults).
    *   **`validation.py`**: (Updated with `isValidFileWriteMode` function).
    *   **`projectTypes.py`**: (Updated with `FileWriteModeType`).

**II. Key Implementation Philosophy and Design Choices:**

*   **Explicit API:** (As before, `fileWriteMode` is now an explicit parameter).
*   **Argument Resolution Process:** (As before).
*   **Immutability of Defaults:** (As before).
*   **File Write Mode (`'w'`) Implementation:** The custom `'w'` mode is implemented by `DynamicLogger` itself performing one-time deletion operations on directories or files *before* `HandlerManager` creates/uses its always-append-mode `FileHandler`s. This keeps `HandlerManager` simpler regarding file modes. Internal flags (`_wMode_remove_mainLogFolder_done`, `_wMode_removed_log_files_this_session`) manage the "once per session" aspect of these deletions.
*   **Separation of Formatting Concerns:** (As before).
*   **Contextual Data on `LogRecord`:** (As before).
*   **Robust `basePath` Determination:** (As before, `DynamicLogger`'s `__init__` now uses `inspect` to determine a path based on its instantiation site and passes this to `PathResolver` as an `explicitBasePath`).
*   **Thread Safety:** (As before, the new file/folder deletion logic in `DynamicLogger.log()` is also protected by the instance-level `_threadLock`).