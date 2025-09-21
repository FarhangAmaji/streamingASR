from utils.dyLog_highOrderOptions import highOrderOptions, globalDebugPrint
from utils.dynamicLoggerModule.dynamicLoggerCore.dynamicLogger_mainClass import DynamicLogger

uniLogger = DynamicLogger(
    highOrderOptions=highOrderOptions,
    simpleLog=not globalDebugPrint,
    timestampFormat="%H:%M:%S.%f"
)
uniDebugLogger = DynamicLogger(
    highOrderOptions=highOrderOptions,
    simpleLog=not globalDebugPrint,
    exclude=True,
    logLevel=DynamicLogger.DEBUG,
    timestampFormat="%H:%M:%S.%f",
)