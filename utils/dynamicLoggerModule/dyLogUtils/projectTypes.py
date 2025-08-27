# dyLogUtils/projectTypes.py
from pathlib import Path
from typing import Union, Dict, Any, List, Tuple, Callable, Literal  # Added Literal

# --- Type Aliases for Clarity ---

# Represents a configuration dictionary that can be part of the priority chain.
# Keys are from ALL_MAIN_LOG_ARGS_KEYS.
ConfigLayerDict = Dict[str, Any]

# Represents a configSet dictionary.
ConfigSet = Dict[str, Any]

# For High-Order Options
# Key is indicatorName or funcMethIndicator (str)
# Value is a dict: {"directArgs": ConfigLayerDict, "configSet": ConfigSet} # Old structure
# New structure: Value is a ConfigLayerDict (which can include a 'configSet' key whose value is a ConfigSet)
HighOrderOptionsRuleBody = ConfigLayerDict  # Updated for simpler HO structure.
HighOrderOptionsDict = Dict[str, HighOrderOptionsRuleBody]

# For arguments passed directly to log functions or __init__
DirectArgsDict = Dict[str, Any]

# For the list of dictionaries forming the priority sources
PrioritySourcesList = List[ConfigLayerDict]

# For the final resolved options after processing the priority chain
FinalResolvedOptionsDict = Dict[str, Any]

# Path-like types that filePath argument can accept
FilePathType = Union[Path, str, bool, None]

# Caller details structure returned by CallerInspector
CallerDetails = Dict[str, Union[str, int, None]]  # Keys: "funcMethIndicator", "pathname", etc.

# Wrappers to skip during stack inspection
WrapperFunction = Callable[..., Any]
WrappersToSkipList = List[WrapperFunction]

# Message arguments for string formatting
MessageArgsTuple = Tuple[Any, ...]

# Standard logging levels
LogLevelInt = int

# Exception info type
ExcInfoType = Union[None, bool, Tuple[type, BaseException, Any], BaseException]

# File write mode type
FileWriteModeType = Literal['w', 'a']
