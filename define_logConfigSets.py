# define_logConfigSets.py
import logging

# ==============================================================================
# Default Log Configuration Sets for DynamicLogger
# ==============================================================================
# These configurations will be used by DynamicLogger.
# Users can define their own sets or modify these.
# All keys should adhere to camelCase.
#
# For `filePath`:
# - Can be a string (relative or absolute path).
# - Relative paths are resolved against DynamicLogger's `basePath`.
# - `None` triggers auto-generation of the log file path.
#   Pattern: logs/{loggerName}_{configSetName}_h{handlerIndex}_auto.log
#
# For `logFormat`:
# - `% (callerInfo)s` can be used to log ClassName.methodName or functionName.
# - `% (indicatorName)s` can be used to log the indicatorName passed in a log call.
# ==============================================================================

defaultLogConfigSets = {
    'default': {
        'logLevel': logging.DEBUG,  # Overall minimum level this config set supports
        'handlers': [
            {
                'handlerType': 'console',
                'logLevel': logging.INFO,
                'logFormat': '[%(levelname)-8s] %(message)s (Console)',
                'timestampFormat': None, # No timestamp for console by default in this set
            },
            {
                'handlerType': 'file',
                'logLevel': logging.DEBUG,
                'filePath': 'logs/app_default.log', # Main application log
                'logFormat': '%(asctime)s | %(levelname)-8s | [%(callerInfo)s:%(lineno)d] | %(message)s',
                'timestampFormat': '%Y-%m-%d %H:%M:%S,%f'[:-3],
            }
        ]
    },
    'verboseConsole': {
        'logLevel': logging.DEBUG,
        'handlers': [
            {
                'handlerType': 'console',
                'logLevel': logging.DEBUG, # Show all debug messages on console
                'logFormat': '[%(levelname)-8s] [%(callerInfo)s:%(lineno)d] %(message)s (VerboseConsole)',
                'timestampFormat': '%H:%M:%S',
            }
            # No file handler in this set by default
        ]
    },
    'fileOnlyDebug': {
        'logLevel': logging.DEBUG,
        'handlers': [
            {
                'handlerType': 'file',
                'logLevel': logging.DEBUG,
                'filePath': 'logs/app_debug_details.log',
                'logFormat': '%(asctime)s | %(levelname)-8s | [%(callerInfo)s:%(lineno)d] | %(message)s (FileOnlyDebug)',
                'timestampFormat': '%Y-%m-%d %H:%M:%S,%f'[:-3],
            }
            # No console handler in this set
        ]
    },
    'criticalErrors': { # For very important errors
        'logLevel': logging.CRITICAL,
        'handlers': [
            {
                'handlerType': 'console',
                'logLevel': logging.CRITICAL,
                'logFormat': '!!! CRITICAL !!! [%(callerInfo)s:%(lineno)d] %(message)s',
                'timestampFormat': None,
            },
            {
                'handlerType': 'file',
                'logLevel': logging.CRITICAL,
                'filePath': 'logs/critical_errors.log',
                'logFormat': '%(asctime)s | CRITICAL | [%(callerInfo)s:%(lineno)d] | %(message)s',
                'timestampFormat': '%Y-%m-%d %H:%M:%S,%f'[:-3],
            }
        ]
    },
    'wslSubprocessOutput': { # Special config for WSL server output
        'logLevel': logging.DEBUG, # Catches all levels for this specific purpose
        'handlers': [
            {
                'handlerType': 'console',
                'logLevel': logging.ERROR, # Only show WSL errors on console by default
                'logFormat': 'WSL_SERVER_CONSOLE: %(message)s',
                'timestampFormat': None,
            },
            {
                'handlerType': 'file',
                'logLevel': logging.DEBUG, # Log all WSL output to a dedicated file
                'filePath': 'logs/wsl_server_output.log',
                'logFormat': '%(asctime)s | WSL_SERVER_FILE | %(message)s', # Simple format for potentially large output
                'timestampFormat': '%Y-%m-%d %H:%M:%S',
            }
        ]
    },
    'appSetupInfo': { # For the initial application setup block
        'logLevel': logging.INFO,
        'handlers': [
            {
                'handlerType': 'console',
                'logLevel': logging.INFO,
                'logFormat': '%(message)s', # Clean format, just the message
                'timestampFormat': None,
            },
            {
                'handlerType': 'file',
                'logLevel': logging.INFO,
                'filePath': 'logs/app_setup.log', # Dedicated file for setup info
                'logFormat': '%(asctime)s | %(message)s',
                'timestampFormat': '%Y-%m-%d %H:%M:%S',
            }
        ]
    }
    # Add more predefined logConfigSets as needed for different scenarios.
    # For example, a 'performanceMetrics' set, or specific sets for different modules if desired.
}
