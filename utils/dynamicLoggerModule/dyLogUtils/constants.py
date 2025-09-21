# dyLogUtils/constants.py
import logging

# --- Core Argument Keys (used as keys in configuration dictionaries) ---
ARG_CONFIG_SET = "configSet"  # A dict bundling other log arguments.
ARG_INDICATOR_NAME = "indicatorName"  # User-defined string for HO matching and formatting.
ARG_PRINT_TO_CONSOLE = "printToConsole"  # Boolean: True to log to sys.stdout.
ARG_FILE_PATH = "filePath"  # Path/str/bool: Specifies file output.
ARG_EXCLUDE = "exclude"  # Boolean: True to suppress the log message completely.
ARG_LOG_FORMAT = "logFormat"  # str/bool: Defines log message structure (content part).
ARG_TIMESTAMP_FORMAT = "timestampFormat"  # str/bool: Controls timestamp prepending.
ARG_EXC_INFO = "excInfo"  # bool/tuple/exception: Include exception traceback.
ARG_EXTRA = "extra"  # Dict: User-defined key-value pairs for LogRecord.
ARG_LOG_LEVEL = "logLevel"  # int: Overrides log level from the called method.
ARG_SIMPLE_LOG = "simpleLog"  # Boolean: Activates simplified logging path (inline only).
ARG_STACK_INFO = "stackInfo"  # Boolean: Include general call stack information.
ARG_MESSAGE_ARGS = "messageArgs"  # Tuple: Args for %-formatting if *args not used.
ARG_FILE_WRITE_MODE = "fileWriteMode"  # str: 'w' for custom overwrite, 'a' for append. NEW

# Frozen set of all valid main log argument keys for validation purposes.
ALL_MAIN_LOG_ARGS_KEYS = frozenset([
    ARG_CONFIG_SET, ARG_INDICATOR_NAME, ARG_PRINT_TO_CONSOLE, ARG_FILE_PATH,
    ARG_EXCLUDE, ARG_LOG_FORMAT, ARG_TIMESTAMP_FORMAT, ARG_EXC_INFO,
    ARG_EXTRA, ARG_LOG_LEVEL, ARG_SIMPLE_LOG, ARG_STACK_INFO, ARG_MESSAGE_ARGS,
    ARG_FILE_WRITE_MODE,  # NEW
])

# --- Default Values for Configuration Levels (`onLevelDefault`) ---
# These values signify "no opinion" or "passthrough" at a specific configuration level.
# If a config level (e.g., an HO rule) explicitly sets an arg to its `onLevelDefault`,
# it means that level wants to defer the decision for that arg to lower priority levels.
ON_LEVEL_DEFAULTS = {
    ARG_CONFIG_SET: None,  # No bundled config by default at any level.
    ARG_INDICATOR_NAME: None,  # No indicator name specified by default.
    ARG_PRINT_TO_CONSOLE: None,  # No opinion on console printing.
    ARG_FILE_PATH: None,  # No opinion on file path.
    ARG_EXCLUDE: None,  # No opinion on excluding.
    ARG_LOG_FORMAT: None,  # No opinion on log format string.
    ARG_TIMESTAMP_FORMAT: None,  # No opinion on timestamp format.
    ARG_EXC_INFO: None,  # No opinion on exception info.
    ARG_EXTRA: None,  # No opinion on extra dict (merging happens separately).
    ARG_LOG_LEVEL: None,  # No opinion on log level override.
    ARG_SIMPLE_LOG: None,  # No opinion on simpleLog mode (mode is set before arg resolution).
    ARG_STACK_INFO: None,  # No opinion on stack info.
    ARG_MESSAGE_ARGS: None,  # No opinion on messageArgs tuple.
    ARG_FILE_WRITE_MODE: None,  # No opinion on file write mode. NEW
}

# --- Final Default Values (`finalDefault`) ---
# These are the ultimate fallback values if no higher-priority source provides an active setting.
FINAL_DEFAULTS = {
    ARG_CONFIG_SET: ON_LEVEL_DEFAULTS[ARG_CONFIG_SET],  # Should always be None if not set.
    ARG_INDICATOR_NAME: ON_LEVEL_DEFAULTS[ARG_INDICATOR_NAME],  # No indicator name by default.
    ARG_PRINT_TO_CONSOLE: True,  # Default to printing to console.
    ARG_FILE_PATH: True,  # Default to trying file logging (specifics depend on context).
    ARG_EXCLUDE: False,  # Default to not excluding messages.
    ARG_LOG_FORMAT: True,  # Special: True means use DEFAULT_LOG_FORMAT_TEMPLATE_CONTENT_ONLY.
    ARG_TIMESTAMP_FORMAT: True,
    # Special: True means prepend time using DEFAULT_TIMESTAMP_FORMAT_STRING.
    ARG_EXC_INFO: False,  # Default to not including exception info.
    ARG_EXTRA: {},  # Default to an empty dictionary for extra attributes (new instance each time).
    ARG_LOG_LEVEL: ON_LEVEL_DEFAULTS[ARG_LOG_LEVEL],  # No log level override by default.
    ARG_SIMPLE_LOG: False,  # Default to not using simpleLog mode.
    ARG_STACK_INFO: False,  # Default to not including stack info.
    ARG_MESSAGE_ARGS: (),  # Default to an empty tuple for messageArgs.
    ARG_FILE_WRITE_MODE: 'a',  # Default to custom write mode. NEW
}

# --- Default Format Strings, Placeholders, and Separators ---

# Default template for the CONTENT part of the log message (timestamp is prepended separately).
# `{indicatorSegment}` is a placeholder within this template that DynamicFormatter fills.
DEFAULT_LOG_FORMAT_TEMPLATE_CONTENT_ONLY = "%(levelname)-8s | {indicatorSegment}[%(callerInfo)s:%(lineno)d] | %(message)s"

# Segment to insert into the content template if an indicatorName is active.
# `%(indicatorName)s` will be substituted by the LogRecord's 'indicatorName' attribute.
DEFAULT_LOG_FORMAT_INDICATOR_SEGMENT = "indicatorName: %(indicatorName)s | "

# Default format string for `strftime` if `timestampFormat` is True.
DEFAULT_TIMESTAMP_FORMAT_STRING = "%Y-%m-%d %H:%M:%S.%f"

# Separator string placed between the prepended timestamp and the main log content.
DEFAULT_TIMESTAMP_SEPARATOR = " | "

# Placeholder string for unavailable information (e.g., callerInfo in simpleLog mode).
PLACEHOLDER_NA = "N/A"

# --- File System Related Constants ---
DEFAULT_LOGS_DIR_NAME = "logs"  # Default subdirectory name for log files, relative to basePath.
DEFAULT_SIMPLE_LOG_FILENAME_BASE = "appMain"  # Default base filename if filePath=True in simpleLog mode.
LOG_FILE_EXTENSION = ".log"  # Default extension for log files.

# --- High-Order Option Structure Keys (Internal) ---
# These were from a previous design iteration and are not directly used if an HO rule body
# is just a flat dictionary of log arguments plus an optional 'configSet' key.
# Kept for reference if a more structured HO body was ever reconsidered.
# HO_DIRECT_ARGS_KEY = "directArgs" # No longer primary way to structure HOs
# HO_CONFIG_SET_KEY = "configSet"   # This IS a valid key within an HO rule body dict.


# --- LogRecord Attribute Keys (Internal, for communication with DynamicFormatter) ---
# These keys are used to attach resolved argument values or state flags to the LogRecord's
# 'extra' dictionary, so DynamicFormatter can access them.
RECORD_ATTR_RESOLVED_INDICATOR_NAME = "resolvedIndicatorName"  # Resolved indicatorName for formatter (may differ from what triggered HO)
RECORD_ATTR_RESOLVED_CALLER_INFO = "resolvedCallerInfo"  # Resolved funcMethIndicator for formatter
RECORD_ATTR_IS_SIMPLE_LOG_MODE = "isSimpleLogMode"  # Boolean: True if log call is in simpleLog mode.
RECORD_ATTR_LOG_FORMAT_FROM_ARG = "logFormatFromArg"  # The resolved value of the 'logFormat' argument.
RECORD_ATTR_TIMESTAMP_FORMAT_FROM_ARG = "timestampFormatFromArg"  # The resolved value of the 'timestampFormat' argument.

# --- Standard Logging Levels (convenience aliases) ---
LOG_LEVEL_DEBUG = logging.DEBUG
LOG_LEVEL_INFO = logging.INFO
LOG_LEVEL_WARNING = logging.WARNING
LOG_LEVEL_ERROR = logging.ERROR
LOG_LEVEL_CRITICAL = logging.CRITICAL
LOG_LEVEL_NOTSET = logging.NOTSET
