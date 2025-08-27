## DynamicLogger: Most common Configuration Guide





### 1. Dynamic and Flexible Logging Options



The core strength of DynamicLogger is its ability to change logging behavior dynamically and from multiple control points. This design eliminates the need for a single, monolithic setup file and application restarts, which are common in traditional logging frameworks. Instead, you can fine-tune every log call's output, format, and destination on the fly, giving you precise control for debugging or performance-sensitive scenarios. This flexibility is managed by a smart **Prioritized Configuration Chain**, ensuring that the most specific setting always takes precedence.

The `simpleLog` and `exclude` options are performance-critical features. The `simpleLog` option is a streamlined logging mode that bypasses expensive inspections for a significant increase in speed. Both of these features are explained in detail in the `log()` function reference (Section 5).

**Example:**

Python

```
from utils.dyLog_highOrderOptions import highOrderOptions, globalDebugPrint
from dynamicLoggerCore.dynamicLogger_mainClass import DynamicLogger

# Create uniLogger for general logs
uniLogger = DynamicLogger(
    highOrderOptions=highOrderOptions,
    simpleLog=not globalDebugPrint,
    timestampFormat="%H:%M:%S.%f"
)

# Create uniDebugLogger for debug logs
uniDebugLogger = DynamicLogger(
    highOrderOptions=highOrderOptions,
    simpleLog=not globalDebugPrint,
    exclude=not globalDebugPrint,
    logLevel=DynamicLogger.DEBUG,
    timestampFormat="%H:%M:%S.%f",
)

# --- Basic Usage ---
# Console only
uniLogger.info("System started", printToConsole=True, filePath=False)
# File only
uniLogger.error("Critical failure", filePath="critical.log", printToConsole=False)
# Custom format & file
uniLogger.debug("Debugging...", logFormat="%(levelname)s >> %(message)s", filePath="debug.log")
```

**Important Note:** The creation of the `uniLogger` and `uniDebugLogger` instances is a foundational and recommended pattern. This approach helps you keep general application messages separate from technical debugging details. You are encouraged to take inspiration from this idea and create other specialized instances for your project's specific needs, such as a `db_logger` for your database or an `audit_logger` for security events.

#### The Roles and Differences of `uniLogger` and `uniDebugLogger`



These two logger instances are designed for completely different purposes. This separation helps you manage your logs more effectively and keep general messages separate from technical details.

- **Key Insight:** `uniLogger` is for general and important messages like errors and warnings that should always be active. In contrast, `uniDebugLogger` is for technical and debug details, which are **disabled by default**. This separation allows you to easily turn all debug logs on or off with a simple configuration change at runtime.

- **Example of Runtime Activation:**

  Python

  ```
  # By default, this line produces no output.
  uniDebugLogger.debug("This is a debug message.")
  
  # But with a small change in High-Order Options, debug logs can be enabled for a specific function.
  high_order_options = {
      "BankAccount.withdraw": {"exclude": False}
  }
  uniDebugLogger.updateHighOrderOptions(high_order_options)
  
  # Now, this line will produce output when called from within the BankAccount.withdraw method.
  class BankAccount:
      def withdraw(self, amount):
          uniDebugLogger.debug(f"Attempting to withdraw {amount}")
          # ...
  ```

------



### 2. High-Order Options (HOs): External Configuration Control



**High-Order Options (HOs)** are a central control hub for configuring logging behavior, allowing you to define logging rules for specific parts of your codebase without modifying the code itself. HOs are a dictionary where each key matches a function name, a class method, or a custom tag. When a log call is made, DynamicLogger checks if the caller matches a key in the HOs and applies those settings with the highest priority.



#### Strategic Applications:



- **Development Phase:** Configure HOs to temporarily enable `DEBUG` logs for a specific module.
- **Production Troubleshooting:** Quickly add an HO rule for a problematic function to activate detailed logging.
- **Performance Tuning:** Use `{"exclude": true}` to completely suppress logs from a high-frequency function.



#### Runtime Updates:



DynamicLogger allows you to change the rules of HOs at runtime, without needing to restart the application, through the `updateHighOrderOptions` function.

**Example:**

Python

```
# dyLog_highOrderOptions.py
highOrderOptions = {
    "funcName": {
        "exclude": False,
        "configSet": { "filePath": "db.log", "printToConsole": False }
    },
    "PaymentService.processTransaction": {
        "configSet": { "filePath": "payments.log", "printToConsole": True }
    }
}
```

Python

```
from utils.loggerInstance import uniLogger

uniLogger.info("Query started", indicatorName="db")

class PaymentService:
    def processTransaction(self, tx):
        uniLogger.debug("Processing %s", messageArgs=(tx.id,))
```

------



### 3. ConfigSets: Reusable Configuration Templates



A **ConfigSet** is a reusable bundle of logging options, defined as a standard Python dictionary. This powerful tool is designed to help you write cleaner code by eliminating redundancy. Essentially, you can think of a `ConfigSet` as a "logging template."

**Example:**

Python

```
dbConfigSet = {
    "filePath": "db_activity.log",
    "printToConsole": False,
    "logLevel": 20  # INFO
}

# Used Inline
uniLogger.info("Connection opened", configSet=dbConfigSet)

# Used during Initialization
from dynamicLoggerCore.dynamicLogger_mainClass import DynamicLogger
dbLogger = DynamicLogger(configSet=dbConfigSet)
dbLogger.warning("High latency detected")

# Used in a High-Order Option
highOrderOptions = {
    "db": { "configSet": dbConfigSet }
}
```

------



### 4. The Configuration Priority Chain



DynamicLogger resolves the final logging options by following a strict 7-level priority chain. The first non-null value found for a given option is the one that is used.



#### The 7-Level Priority Hierarchy:



1. **HO Direct Arguments**
2. **HO ConfigSet**
3. **Inline Direct Arguments**
4. **Inline ConfigSet**
5. **Init Direct Arguments**
6. **Init ConfigSet**
7. **Framework Defaults**

**Example:**

Python

```
# HO wins
highOrderOptions = { "db": { "filePath": "db_ho.log" } }

# Init defaults
loggerInit = DynamicLogger(filePath="init_default.log")

# HO overrides both Inline and Init settings for this call
loggerInit.info("Hello inline", filePath="inline_override.log", indicatorName="db")
```

------



### 5. Comprehensive `log()` Function Options Reference





#### Core Parameters:



- **`indicatorName`** (string): A label to tag a log call, primarily for matching HO rules.
- **`printToConsole`** (boolean): If `True`, the log message is sent to the console.
- **`filePath`** (flexible type): Defines the destination for the log file.
- **`fileWriteMode`** (string): Controls how logs are written to a file (`'w'` for smart-overwrite, `'a'` for append).



```
# The first time this line runs in a session, the log file is cleared.
uniLogger.info("Starting a fresh session", fileWriteMode="w", filePath="session.log")

# Any subsequent logs in the same session will be appended, not overwritten.
uniLogger.debug("Debug info for ongoing task", filePath="session.log")
```

#### Performance-Critical Parameters:



- **`exclude`** (boolean): If `True`, the log message is completely suppressed with near-zero performance cost.
- **`simpleLog`** (boolean): If `True` and passed directly inline, it activates a simplified, high-performance logging path.



#### Formatting Parameters:



- **`logFormat`** (flexible type): Defines the structure of the log message content.
- **`timestampFormat`** (flexible type): Controls the timestamp formatting.



#### Advanced Parameters:



- **`excInfo`** (flexible type): Includes exception traceback information.
- **`extra`** (dictionary): Adds custom key-value pairs to the log record.
- **`logLevel`** (integer): Overrides the default log level of the called function.
- **`stackInfo`** (boolean): Adds general stack call information.
- **`messageArgs`** (tuple): Provides arguments for `%`-style string formatting.



#### Usage Examples:



Python

```
# indicatorName
uniLogger.info("Tagged log", indicatorName="network")
# printToConsole
uniLogger.info("Console only", printToConsole=True, filePath=False)
# filePath variations
uniLogger.info("Auto file", filePath=True)
# fileWriteMode
uniLogger.info("Fresh session", filePath="session.log", fileWriteMode="w")
# exclude
uniLogger.debug("Won't be logged", exclude=True)
# simpleLog
uniLogger.info("Fast log", simpleLog=True)
# logFormat
uniLogger.info("Formatted", logFormat="%(levelname)s - %(message)s")
# excInfo
try:
    1/0
except:
    uniLogger.error("Division failed", excInfo=True)
```

------



### 6. Smart Project Path Resolution (`basePath`) — Advanced



One hidden but powerful feature is automatic project-root discovery. DynamicLogger determines a stable `basePath` (project root) even when your code runs from different working directories, resolving the default logs directory to something like `<project_root>/logs`.



#### Why it matters



- **Predictable destinations:** Log files consistently land in the same place, avoiding scattered files.
- **Safer deployments:** CI/CD or scheduled tasks won’t accidentally write logs to unexpected system folders.
- **Portable configs:** You can specify `filePath` as a name (e.g., `"app.log"`) or a relative path (e.g., `"sub/worker.log"`), and it will be resolved correctly.

**Practical example:**

Python

```
# Always lands under <project_root>/logs
uniLogger.info("Booting service...", filePath=True)

# Name resolved under the logs folder
uniLogger.warning("Slow query", filePath="db/perf.log")

# Absolute path stays absolute
uniLogger.error("Disk full!", filePath="/var/log/my_app/errors.log")
```