# dynamicLoggerCore/customFormatter.py
import logging
import sys
import time
import datetime  # <--- ADDED: This is required to handle %f format codes
from typing import Optional

from utils.dynamicLoggerModule.dyLogUtils import constants


# CUSTOM_FORMATTER_DEBUG_MODE = True # Replaced by instance-level debugPrint

# def _debugPrint(message: str): # Replaced by instance method _instanceDebugPrint
#     if CUSTOM_FORMATTER_DEBUG_MODE:
#         print(f"DEBUG_CustomFormatter: {message}", file=sys.stderr, flush=True)


class DynamicFormatter(logging.Formatter):
    def __init__(self,
                 fmt: str = None,  # Default fmt is effectively "%(message)s" now due to super call
                 datefmt: str = None,
                 style: str = '%',
                 validate: bool = True, *, defaults=None,
                 # Added by Python 3.10, keep for compatibility
                 debugPrint: bool = False):  # Added debugPrint

        # Initialize the base Formatter.
        # We primarily format %(message)s here and handle timestamp prefixing ourselves.
        # The `fmt` passed to super() will be dynamically changed or used for the content part.
        super().__init__(fmt="%(message)s", datefmt=datefmt, style=style)
        # `validate` and `defaults` are part of logging.Formatter.__init__ signature in newer Pythons.

        self._debugPrintActive = debugPrint  # Store debugPrint state
        # Store the initial style format string if needed for reset, though we override it per call.
        self._initial_style_fmt_on_init = self._style._fmt  # The fmt string for the content part
        self._initial_datefmt_on_init = self.datefmt  # The date format string

        self._instanceDebugPrint(
            f"Initialized. Initial content_fmt='{self._initial_style_fmt_on_init}', "
            f"initial datefmt='{self._initial_datefmt_on_init}', style='{style}'")

    def _instanceDebugPrint(self, message: str):
        """Prints debug messages for this DynamicFormatter instance if debugPrint is enabled."""
        if self._debugPrintActive:
            print(f"DEBUG_CustomFormatter ({id(self)}): {message}", file=sys.stderr, flush=True)

    # -----------------------------------------------------------------------------------
    # --- vvv THIS METHOD HAS BEEN REPLACED TO FIX THE ValueError AND ADD 4-DIGIT ms vvv ---
    # -----------------------------------------------------------------------------------
    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        """
        Overrides the default formatTime to support sub-second precision (%f)
        and custom 4-digit truncation, which time.strftime() cannot do.
        """
        # KEY CHANGE: Create a datetime object from the record's creation time.
        # This is necessary because datetime.strftime() supports %f, while time.strftime() does not.
        dt = datetime.datetime.fromtimestamp(record.created)

        if datefmt:
            # Check if the user wants sub-second precision via %f.
            if "%f" in datefmt:
                # Format with 6-digit microseconds first.
                s = dt.strftime(datefmt)
                # Truncate to the desired 4-digit precision (ten-thousandths of a second).
                # This slices a string like "...123456" to "...1234".
                return s[:-2]
            else:
                # If %f is not in the format string, time.strftime would have worked,
                # but using datetime.strftime is consistent and safe.
                return dt.strftime(datefmt)
        else:
            # Fallback to standard library's default time formatting if no datefmt provided.
            # This logic is adapted from the original logging.Formatter to use our 'dt' object.
            t = dt.strftime(self.default_time_format)  # e.g., "%H:%M:%S"
            msecs = int(dt.microsecond / 1000)
            if self.default_msec_format:  # e.g., "%s,%03d"
                return self.default_msec_format % (t, msecs)
            return t
    # -----------------------------------------------------------------------------------
    # --- ^^^ END OF MODIFIED SECTION ^^^ ---
    # -----------------------------------------------------------------------------------

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats the LogRecord into a final string.
        Dynamically constructs the output based on attributes attached to the record
        (e.g., logFormatFromArg, timestampFormatFromArg, indicatorName, callerInfo)
        which are set by DynamicLogger.log() via the 'extra' dictionary.
        """
        self._instanceDebugPrint(f"--- format() called for record.msg (raw): '{record.msg}' ---")

        # Retrieve formatting control arguments from the record's attributes.
        # These were set by DynamicLogger.log() from the resolved arguments.
        # Fallback to constants' FINAL_DEFAULTS if attributes are missing (should not happen in normal flow).
        logFormatArg = getattr(record, constants.RECORD_ATTR_LOG_FORMAT_FROM_ARG,
                               constants.FINAL_DEFAULTS[constants.ARG_LOG_FORMAT])
        timestampFormatArg = getattr(record, constants.RECORD_ATTR_TIMESTAMP_FORMAT_FROM_ARG,
                                     constants.FINAL_DEFAULTS[constants.ARG_TIMESTAMP_FORMAT])

        # Retrieve contextual info also attached to the record.
        # These are used for %(key)s substitution in format strings.
        record_indicatorName = getattr(record, "indicatorName", constants.PLACEHOLDER_NA)
        record_callerInfo = getattr(record, "callerInfo",
                                    constants.PLACEHOLDER_NA)  # e.g., Class.method or func
        # record.levelname, record.lineno, record.message are standard.

        self._instanceDebugPrint(
            f"  Record attributes: logFormatArg='{logFormatArg}', timestampFormatArg='{timestampFormatArg}'")
        self._instanceDebugPrint(
            f"  Record attributes for formatting: record.indicatorName='{record_indicatorName}', "
            f"record.callerInfo='{record_callerInfo}', record.lineno='{record.lineno}'")

        # Store original formatter state to restore it later, ensuring formatter is stateless between calls.
        original_content_fmt_str = self._style._fmt
        original_datefmt_str = self.datefmt  # Standard datefmt used by asctime

        base_log_content_str = ""

        # 1. Determine the format string for the "content" part of the log message.
        effective_content_fmt_str = ""
        if logFormatArg is False:
            # Raw message only, no adornments from format string. Timestamp is also suppressed.
            effective_content_fmt_str = "%(message)s"  # Simplest way to get just the message
            self._instanceDebugPrint(
                f"  logFormatArg is False. Content format set to: '%(message)s'")
        elif logFormatArg is True or logFormatArg is None:  # None implies default behavior if not overridden
            # Use the default template. Conditionally include the indicator segment.
            indicatorSegmentStr = ""
            # Only include indicator segment if indicatorName is meaningful (not placeholder or empty)
            # and this is not simpleLog mode (where indicatorName on record might be N/A).
            # The record.indicatorName should already be PLACEHOLDER_NA if in simpleLog mode.
            if record_indicatorName and record_indicatorName != constants.PLACEHOLDER_NA:
                indicatorSegmentStr = constants.DEFAULT_LOG_FORMAT_INDICATOR_SEGMENT
            effective_content_fmt_str = constants.DEFAULT_LOG_FORMAT_TEMPLATE_CONTENT_ONLY.format(
                indicatorSegment=indicatorSegmentStr
            )
            self._instanceDebugPrint(
                f"  logFormatArg is True/None. Default content format template used. IndicatorSegment: '{indicatorSegmentStr}'. "
                f"Effective content format: '{effective_content_fmt_str}'")
        elif isinstance(logFormatArg, str) and logFormatArg:  # User-provided format string
            effective_content_fmt_str = logFormatArg
            self._instanceDebugPrint(
                f"  logFormatArg is a string. Custom content format: '{effective_content_fmt_str}'")
        else:  # Invalid logFormatArg value, fallback to just message
            print(
                f"Warning_CustomFormatter ({id(self)}): Invalid logFormatArg '{logFormatArg}' (type {type(logFormatArg)}). Defaulting to message only for content.",
                file=sys.stderr, flush=True)
            self._instanceDebugPrint(
                f"  logFormatArg is invalid ('{logFormatArg}'). Fallback content format: '%(message)s'")
            effective_content_fmt_str = "%(message)s"

        # 2. Format the "content" part using the determined format string.
        # Temporarily set the formatter's style and date format for this operation.
        self._style._fmt = effective_content_fmt_str
        self.datefmt = None  # Ensure %(asctime)s in custom content_fmt_str does not get formatted here.
        # Timestamp is handled separately and prepended.

        try:
            # self._instanceDebugPrint( # Very verbose
            #     f"  Record dictionary before super().format for base content: {record.__dict__}")
            # super().format(record) will call record.getMessage() to format `record.msg % record.args`
            # and then apply `self._style._fmt` to the record.
            base_log_content_str = super().format(
                record)  # This uses the temporarily set self._style._fmt
            self._instanceDebugPrint(f"  Base log content after formatting with "
                                     f"'{effective_content_fmt_str[:100]}...': '{base_log_content_str[:300]}'")
        except Exception as e_fmt_base:
            # If formatting the base content fails, log an error and use a fallback message.
            # This can happen if user-supplied format string is invalid or record attributes are missing.
            print(
                f"ERROR_CustomFormatter ({id(self)}): Exception during base content formatting: {e_fmt_base}. "
                f"Record msg: '{record.msg}', Effective content fmt: '{effective_content_fmt_str}'",
                file=sys.stderr, flush=True)
            self._instanceDebugPrint(f"    Exception during base content formatting: {e_fmt_base}")
            base_log_content_str = f"[BASE CONTENT FORMATTING ERROR: {record.getMessage()}]"  # getMessage() is safer

        # 3. Determine and prepend the timestamp if needed.
        # Timestamp is NOT prepended if logFormatArg was False (raw message mode).
        timestamp_prefix_str = ""
        if logFormatArg is False:
            self._instanceDebugPrint("  logFormatArg is False. Timestamp is fully suppressed.")
            # final_output_str will just be base_log_content_str (which is record.getMessage())
        else:
            # logFormat is True or a string, so consider timestampFormatArg.
            if timestampFormatArg is True:  # Default timestamp format
                ts_str = self.formatTime(record, constants.DEFAULT_TIMESTAMP_FORMAT_STRING)
                timestamp_prefix_str = f"{ts_str}{constants.DEFAULT_TIMESTAMP_SEPARATOR}"
                self._instanceDebugPrint(
                    f"  timestampFormatArg is True. Default timestamp: '{ts_str}'")
            elif isinstance(timestampFormatArg,
                            str) and timestampFormatArg:  # Custom timestamp format
                try:
                    ts_str = self.formatTime(record, timestampFormatArg)
                    timestamp_prefix_str = f"{ts_str}{constants.DEFAULT_TIMESTAMP_SEPARATOR}"
                    self._instanceDebugPrint(
                        f"  timestampFormatArg is str ('{timestampFormatArg}'). Custom timestamp: '{ts_str}'")
                except Exception as e_ts_fmt:
                    # If custom timestamp format is invalid.
                    print(
                        f"ERROR_CustomFormatter ({id(self)}): Invalid custom timestampFormat string '{timestampFormatArg}': {e_ts_fmt}. No timestamp will be prepended.",
                        file=sys.stderr, flush=True)
                    self._instanceDebugPrint(
                        f"    Error formatting custom timestamp: {e_ts_fmt}. No timestamp.")
                    timestamp_prefix_str = ""  # Fallback to no timestamp
            elif timestampFormatArg is False:  # Explicitly no timestamp
                self._instanceDebugPrint(
                    "  timestampFormatArg is False. No timestamp will be prepended.")
                timestamp_prefix_str = ""
            # else: timestampFormatArg is None or invalid - effectively treated as no timestamp by omission

        # 4. Combine timestamp prefix and base content.
        final_output_str = f"{timestamp_prefix_str}{base_log_content_str}"
        self._instanceDebugPrint(f"  Final combined output string: '{final_output_str[:300]}'")

        # 5. Restore original formatter state.
        self._style._fmt = original_content_fmt_str
        self.datefmt = original_datefmt_str
        # self._instanceDebugPrint( # Can be verbose
        #     f"  Formatter style._fmt RESTORED to: '{self._style._fmt}', datefmt RESTORED to: '{self.datefmt}'")

        return final_output_str