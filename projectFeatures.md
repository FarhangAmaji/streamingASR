*   **Real-Time Speech-to-Text Engine (Python, Multi-ASR, Cross-Platform with WSL focus)**
    *   **Core Functionality & User Experience**
        *   Real-time Audio Transcription
            *   Captures audio from microphone (`sounddevice`)
            *   Supports multiple ASR model backends
        *   File-Based Transcription (`useFileTranscriber.py`, `FileTranscriber` in `tasks.py`)
            *   Processes pre-recorded audio files
            *   Uses configured ASR handler for transcription
            *   Outputs to console or specified text file
        *   Hotkey Control (`systemInteractions.py` via `keyboard` library)
            *   Toggle Recording (Start/Stop audio capture)
            *   Toggle Text Output (Enable/Disable typing/clipboard output)
            *   Force Transcription (Immediately process current audio buffer)
                *   User-configurable hotkey (e.g., `ctrl+,`)
                *   Cooldown mechanism (0.5s) to prevent rapid re-triggering
        *   Flexible Text Output (`systemInteractions.py`)
            *   Simulated Typing (Windows native via `pyautogui`)
                *   Modes: Letter-by-letter, Word-by-word, Whole block
                *   Interruptible by Ctrl key or output disable during "word" mode
            *   Clipboard (Windows via `clip.exe`, accessible from WSL)
            *   Globally enable/disable via configuration
        *   Audio Notifications (`systemInteractions.py` via `pygame`)
            *   Optional sound feedback for key events (e.g., recording on/off, model unloaded)
            *   Configurable enablement for all notifications and specific "enable" sounds
    *   **ASR Model Management & Integration (`modelHandlers.py`, `mainManager.py`)**
        *   Abstract ASR Handler Interface (`AbstractAsrModelHandler`)
            *   Defines contract for `loadModel`, `unloadModel`, `transcribeAudioSegment`, `isModelLoaded`, `getDevice`
        *   Local Whisper Model Support (`WhisperModelHandler`)
            *   Integrates Hugging Face `transformers` for Whisper models (e.g., `tiny.en`, `large-v3`)
            *   Automatic device selection (CUDA GPU or CPU, with config override for CPU)
            *   CUDA memory management (`_cudaClean`, `_monitorMemory`)
            *   Model warm-up for reduced first-inference latency
            *   Handles `trust_remote_code` and `torch_dtype` for model loading
            *   Robust error handling during model load (OOM, dependencies, etc.)
        *   Remote NVIDIA NeMo Model Support (`RemoteNemoClientHandler`, `wslNemoServer.py`)
            *   Client for `wslNemoServer.py` (Flask application)
            *   Communicates via HTTP requests (`requests` library) for status, load/unload, transcribe
            *   Handles multilingual NeMo models (e.g., Canary) by passing `target_lang`
            *   Robust error handling for server communication (connection, timeout, HTTP errors)
            *   Server Reachability and Model Status Tracking
        *   Automatic WSL NeMo Server Lifecycle Management (Windows hosts in `mainManager.py`)
            *   Launches `wslNemoServer.py` in specified WSL distribution (`subprocess.Popen`)
                *   Robust script path detection (main script dir, CWD fallback)
                *   Windows path to WSL path conversion (`utils.convertWindowsPathToWsl`)
                *   Configurable `sudo` usage for server script
            *   Monitors server reachability (`_waitForServerReachable`) via `/status` endpoint
            *   Captures and logs server output on error (`_logWslProcessOutputOnError`)
                *   Detects common errors (sudo password, Python traceback, address in use)
            *   Terminates server process on application exit (`_terminateWslServer`)
        *   Automatic Model Unloading (`ModelLifecycleManager` in `managers.py`)
            *   Unloads ASR model (local or remote) after configurable period of inactivity (`model_unloadTimeout`)
            *   Reloads model if recording starts and model is not loaded
    *   **Audio Processing & Transcription Logic (`audioProcesses.py`, `tasks.py`)**
        *   Audio Input Stream Management (`AudioHandler`)
            *   Device selection and configuration (`_setupDeviceInfo`)
            *   Non-blocking audio chunk queuing from `sounddevice` callback
            *   Stream start/stop and queue clearing
        *   Real-Time Audio Buffering & Triggering (`RealTimeAudioProcessor`)
            *   Accumulates audio chunks into a `Transcription Window`
            *   Audio chunk pre-processing (float32 conversion, mono averaging)
            *   Transcription Trigger Modes:
                *   `dictationMode`: Triggers after speech followed by configurable silence duration and loudness threshold
                *   `constantIntervalMode`: Triggers at fixed time intervals
            *   `getAudioBufferCopyAndClear()`: Method for `Force Transcription` hotkey, provides buffer and resets state.
            *   Buffer clearing also resets dictation state (speaking flag, silence timer)
        *   Transcription Output Handling & Filtering (`TranscriptionOutputHandler` in `tasks.py`)
            *   Calculates segment loudness for filtering decisions
            *   Silence/Low Content Skipping:
                *   Minimum loud duration check (`minLoudDurationForTranscription`)
                *   Average loudness threshold (`silenceSkip_threshold`)
                *   Overrides based on loudness in leading/trailing sections of the segment
            *   False Positive Word Filtering:
                *   Filters common misrecognized words (e.g., "you", "thank you") if segment loudness is below a threshold
                *   Uses normalized text for comparison
            *   Banned Word Removal (case-insensitive regex replacement)
            *   Trailing dot removal
            *   Updates "last valid transcription time" only for actual, non-filtered output
    *   **Configuration & State Management (`managers.py`)**
        *   Centralized Configuration (`ConfigurationManager`)
            *   Initialized with user settings from main script
            *   Provides `get`/`set` access, `getAll` for a copy
            *   Robustly determines `scriptDir`
        *   Dynamic State Tracking (`StateManager`)
            *   Manages `isRecordingActive`, `outputEnabled`, `isProgramActive`
            *   Tracks key timestamps for timeouts: `programStartTime`, `lastActivityTime`, `recordingStartTime`, `lastValidTranscriptionTime`
            *   Implements timeout checks: `checkRecordingTimeout`, `checkIdleTimeout`, `checkProgramTimeout`
    *   **Advanced Logging System (`dynamicLogger.py`, `utils.py`, `define_logConfigSets.py`)**
        *   `DynamicLogger` Class:
            *   Highly configurable with `Log Configuration Sets` and `High-Order Options`
            *   Supports multiple handlers (console, file) per log set
            *   Customizable log formats (including `%(callerInfo)s`, `%(indicatorName)s`) and timestamp formats
            *   Priority system for resolving log parameters (HO > Inline > ConfigSet > Default)
            *   Automatic caller info capture (`getCallerInfo`)
            *   Auto-generation of log file paths relative to `basePath`
            *   Thread-safe handler caching
        *   `define_logConfigSets.py`: Predefined log configurations (e.g., `default`, `verboseConsole`, `wslSubprocessOutput`)
        *   `utils.py`:
            *   Global `appDynamicLogger` instance and `configure_dynamic_logging` setup function
            *   Convenience wrappers (`logDebug`, `logInfo`, `logWarning`, `logError`, `logCritical`)
            *   Robust `exc_info` handling in `logError`/`logCritical` to prevent logging system errors
            *   Silencing/level adjustment for noisy third-party libraries
            *   Fallback to print-based logging if `DynamicLogger` initialization fails
    *   **Architectural & Development Considerations**
        *   Modular Design: Clearly separated components for audio, state, config, model interaction, system interaction, tasks, and logging.
        *   Thread Management (`mainManager.py`):
            *   Dedicated threads for keyboard monitoring, model lifecycle, and transcription processing
            *   `threadWrapper` for common exception handling in threads
            *   Graceful thread shutdown using `stateManager.shouldProgramContinue()` and joining with timeouts
        *   Error Handling:
            *   Extensive use of `try-except` blocks throughout the application
            *   Specific exception handling (e.g., `PortAudioError`, `FileNotFoundError`, `requests.exceptions`)
            *   Critical errors log with `logCritical` and may stop the application or disable features
            *   Detailed error messages and hints provided in logs
        *   WSL Path Conversion (`utils.convertWindowsPathToWsl`)
            *   Handles drive letters and basic UNC paths
            *   Attempts to resolve relative paths before conversion
            *   Uses conditional logging to avoid errors if logger is not yet available