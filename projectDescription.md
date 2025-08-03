## Real-Time Speech-to-Text Engine: Project Description

---

### Level 1: Big Picture Overview

**What is this project?**
This project is a modular, real-time (and file-based) Speech-to-Text (ASR) application. It's designed to capture audio input, transcribe it using a selection of ASR models, and then output the resulting text, typically by simulating keyboard input or copying to the clipboard. Its core strength lies in its flexibility, allowing users to choose between different ASR technologies (like local Whisper or remote NVIDIA NeMo models) and configure various aspects of its behavior, including transcription modes, hotkeys, and output filtering.

**Rough Implementation & Key Technologies:**
The application is primarily written in Python. Key components and libraries include:
*   **Audio Input:** `sounddevice` library for capturing raw audio from microphones.
*   **ASR Model Interaction:**
    *   Hugging Face `transformers` library for running local Whisper models.
    *   `requests` library for client-side communication with a remote NeMo ASR server.
    *   `Flask` for the `wslNemoServer.py`, which hosts NeMo models.
    *   `torch` (PyTorch) as the backend for both Whisper and NeMo models.
*   **System Interaction:**
    *   `keyboard` library for global hotkey detection.
    *   `pygame` (optional) for audio notifications.
    *   `pyautogui` (optional, Windows-only) for simulating keyboard typing.
    *   `subprocess` for managing the WSL NeMo server process and `clip.exe` interaction.
*   **Core Architecture:** A modular design featuring distinct managers for configuration, state, audio processing, model handling, system interactions, and logging.

**Platform & Development Environment:**
The application is developed in Python and designed to be cross-platform, but full functionality, particularly the automated management of NVIDIA NeMo models, is optimized for a **Windows host interacting with a Linux environment (e.g., Ubuntu) running within Windows Subsystem for Linux (WSL)**. Local Whisper models can run on Windows, Linux, or macOS, provided Python and the necessary dependencies are installed. The tracebacks suggest development and testing within a Python environment, possibly managed by Anaconda.

---

### Level 2: Core Modules, Features, and Developer Considerations

This level delves into the primary components and their roles, catering to both users interested in features and developers needing to understand behavior and error handling.

**Custom Definitions:**

*   **`Transcription Window`**: The segment of accumulated audio data currently held within the `RealTimeAudioProcessor`'s internal buffer that is being considered for, or is ready to be sent for, transcription.
*   **`ASR Handler`**: An abstraction (`AbstractAsrModelHandler`) representing the interface to an ASR engine. Concrete implementations include `WhisperModelHandler` (for local Hugging Face Whisper models) and `RemoteNemoClientHandler` (for interacting with the `wslNemoServer.py`).
*   **`Log Configuration Set`**: A named dictionary defining a complete logging setup (multiple handlers like console/file, their individual log levels, message formats, and timestamp formats) used by the `DynamicLogger`.
*   **`High-Order Options (Logging)`**: Special configuration rules within `DynamicLogger` that can override logging behavior for messages originating from specific functions/methods (identified by `funcMethIndicator`) or tagged with a specific `indicatorName` in the log call.
*   **`Force Transcription`**: A user-initiated action, typically via a hotkey (e.g., `Ctrl + ,`), that immediately sends the current `Transcription Window` for ASR processing, bypassing normal trigger logic and clearing the buffer.
*   **`WSL NeMo Server`**: The `wslNemoServer.py` script, a Flask-based web application designed to run within a WSL environment, host NVIDIA NeMo ASR models, and expose them via HTTP endpoints for transcription.

**I. User-Facing Features & Functionality (High/Medium Importance)**

1.  **Versatile ASR Model Support:**
    *   **Local Whisper Models:** Users can select various OpenAI Whisper models (e.g., `tiny.en`, `medium`, `large-v3`) hosted by Hugging Face. These run directly on the user's machine.
    *   **Remote NVIDIA NeMo Models:** Supports advanced NeMo models (e.g., `Canary`, `Parakeet`) by offloading ASR to a dedicated server (`wslNemoServer.py`) running in WSL. This is ideal for powerful models that might be resource-intensive for direct Windows execution or have Linux-specific dependencies.
2.  **Real-time Transcription Modes:**
    *   **`dictationMode`**: Optimized for spoken dictation. It intelligently waits for a pause (silence) after speech before transcribing the `Transcription Window`.
    *   **`constantIntervalMode`**: Transcribes the `Transcription Window` at fixed, user-defined time intervals, suitable for continuous audio streams where distinct pauses might not occur.
3.  **Intuitive Hotkey Control:**
    *   **Toggle Recording:** Start/stop audio capture (e.g., `Win+Alt+L`).
    *   **Toggle Text Output:** Enable/disable the final text from being typed or copied (e.g., `Ctrl+Q`).
    *   **Force Transcription:** Immediately process the current `Transcription Window` (e.g., `Ctrl+,`).
4.  **Flexible Text Output:**
    *   **Simulated Typing (Windows):** Uses `pyautogui` to type out transcribed text character-by-character, word-by-word, or as a whole block, mimicking human input.
    *   **Clipboard (WSL/Windows):** For users running the main application in WSL or as an alternative on Windows, text can be copied to the system clipboard (uses `clip.exe` on Windows, accessible from WSL).
5.  **Automated WSL NeMo Server Management (Windows Hosts):**
    *   If a NeMo model is selected, the application can automatically launch the `wslNemoServer.py` script in the configured WSL distribution.
    *   It monitors server reachability and can terminate the server process on application exit.
6.  **Audio Feedback:** Optional sound notifications (requires `pygame`) for events like recording start/stop, output enable/disable, and model unloading, providing auditory cues for application state changes.
7.  **File Transcription Utility:** Includes functionality (`FileTranscriber`, `useFileTranscriber.py`) to transcribe pre-recorded audio files using any of the configured ASR Handlers.
8.  **Intelligent Filtering:**
    *   Automatic skipping of silent segments or those with very low audio content.
    *   Filtering of common false positive words (e.g., "you", "thank you") if detected in low-loudness audio.
    *   Removal of user-defined banned words.
9.  **Resource Management:** Automatic unloading of ASR models after a configurable period of inactivity to free up system resources (especially VRAM for local GPU models).

**II. Developer Deep Dive: Expected Behaviors & Error Handling**

This section focuses on how the system is designed to behave, particularly concerning error conditions and edge cases, crucial for developers maintaining or extending the codebase.

*   **Core Principle: Graceful Degradation & Robustness**
    *   The application attempts to handle missing optional dependencies (e.g., `pygame`, `pyautogui`, `sounddevice`) by disabling the corresponding features and logging warnings, rather than crashing.
    *   Critical dependency failures (e.g., `Flask` for the server, `transformers` for Whisper) will prevent startup or specific functionalities, with `logCritical` messages.

*   **Module-Specific Behaviors & Error Handling:**

    *   **`ConfigurationManager` (`managers.py`):**
        *   Robustly determines `scriptDir`. If `__main__.__file__` is unavailable (e.g., interactive session), it attempts fallbacks (directory of `managers.py`, then CWD), logging warnings.
        *   All configurations are accessed via `get()` with defaults, preventing `KeyError` if a setting is missing.

    *   **`StateManager` (`managers.py`):**
        *   Strictly requires a `ConfigurationManager` instance for initialization.
        *   Accurately tracks various timestamps (`programStartTime`, `lastActivityTime`, `recordingStartTime`, `lastValidTranscriptionTime`) crucial for timeout logic and model lifecycle.
        *   `updateLastValidTranscriptionTime()` is only called by `TranscriptionOutputHandler` when truly valid, non-filtered text is produced, ensuring the idle timeout for recording is based on meaningful output.

    *   **`AudioHandler` (`audioProcesses.py`):**
        *   **`sounddevice` Dependency:** If `sounddevice` fails to import or initialize, audio input is disabled, a `logCritical` message is issued, and the application may continue without audio features if designed to do so (though real-time transcription would be impossible).
        *   **Device Setup (`_setupDeviceInfo`):** Logs available audio devices. If a requested `deviceId` is invalid or not found, it defaults to system default. If `requestedChannels` exceed device capabilities, it logs a warning and uses the device's maximum. `actualSampleRate` and `actualChannels` are stored back in the config.
        *   **Audio Callback (`_audioCallback`):** Uses `put_nowait` for the `audioQueue` to prevent blocking the high-priority audio thread. If the queue is full (logged as a warning, indicating processing can't keep up), it drops the oldest chunk to make space for the new one. Catches `queue.Full` and other exceptions during queue operations.
        *   **Stream Management (`startStream`, `stopStream`):** Handles `sd.PortAudioError` (e.g., no microphone, device in use) and `ValueError` (e.g., incompatible parameters) during stream start, logging detailed errors and hints. Ensures streams are properly closed.

    *   **`RealTimeAudioProcessor` (`audioProcesses.py`):**
        *   **Audio Normalization:** Incoming chunks are consistently converted to `float32` and mono (by averaging channels if stereo). Errors during conversion are logged, and the chunk might be dropped.
        *   **Dictation Mode (`_updateDictationState`):**
            *   Calculates chunk loudness. `isCurrentlySpeaking` flag is central.
            *   `silenceStartTime` is only set *after* speech has been detected and then followed by silence, preventing triggers from leading silence.
            *   Resets `silenceStartTime` immediately if speech resumes.
        *   **Buffer Clearing (`clearBuffer`):** Crucially, this method also resets dictation state (`isCurrentlySpeaking`, `silenceStartTime`). This ensures that actions like disabling output, stopping recording, or a `Force Transcription` event correctly reset any ongoing "thought" about an utterance.
        *   **`getAudioBufferCopyAndClear()` (for Force Transcription):** Returns a *copy* of the buffer and then calls `clearBuffer()`. It also resets the `lastTranscriptionTriggerTime` for constant interval mode, treating the forced segment as a complete unit.

    *   **`SystemInteractionHandler` (`systemInteractions.py`):**
        *   **Optional Dependencies (`pygame`, `pyautogui`):** Initialization attempts these imports. Failures result in the respective features (audio notifications, PyAutoGUI typing) being disabled, with warnings logged.
        *   **Hotkey Monitoring (`monitorKeyboardShortcuts`):**
            *   Thread will exit and signal program stop if the `keyboard` library cannot be imported or accessed (e.g., permission errors on Linux, logged as `logCritical`). Provides hints for resolution.
            *   Handles `forceTranscriptionKey` parsing: gracefully manages single keys or "modifier+key" strings. If the key is invalid or unparsable by the `keyboard` library, errors from `keyboard.is_pressed()` are caught within the loop, logged, and the monitor attempts to continue for other valid hotkeys.
            *   **Hotkey Cooldown:** `lastForceTranscriptionTime` and `forceTranscriptionCooldown` (0.5s) prevent the "Force Transcription" action from being triggered multiple times in rapid succession from a single key press/hold.
            *   `_waitForKeyRelease()`: Prevents a single long key press from repeatedly toggling an action. It has its own timeout to avoid getting stuck.
        *   **Text Output (`typeText`):**
            *   Method (`pyautogui`, `clipboard`, `none`) is determined at init based on OS, config, and `pyautogui` availability / `clip.exe` presence.
            *   PyAutoGUI typing can be interrupted if Ctrl is pressed or if `outputEnabled` state changes mid-typing (checked per word in "word" mode).
            *   Clipboard operations via `clip.exe` catch `FileNotFoundError` (if `clip.exe` path becomes invalid) and `subprocess.CalledProcessError`, logging `stderr` from `clip.exe`.

    *   **`TranscriptionOutputHandler` (`tasks.py`):**
        *   **Loudness Calculation:** Uses `_calculateSegmentLoudness` on the *entire* received audio segment for filtering decisions. Handles non-float audio data by attempting normalization.
        *   **Silence/Low Content Filtering (`_shouldSkipTranscriptionDueToSilenceOrLowContent`):**
            *   Requires valid `audioData`.
            *   A multi-step check:
                1.  `minLoudDurationForTranscription`: Skips if the total duration of "loud" samples (above `dictationMode_silenceLoudnessThreshold`) is less than this value.
                2.  `silenceSkip_threshold`: If average segment loudness is above this, the segment is *not* skipped.
                3.  Overrides: If average loudness is low, but the start (`skipSilence_beforeNSecSilence`) OR end (`skipSilence_afterNSecSilence`) of the segment contains sufficiently loud audio, the segment is *not* skipped. This catches short utterances in generally quiet segments.
        *   **False Positive Filtering (`_isFalsePositive`):** Compares normalized (lowercase, no punctuation, collapsed spaces) transcribed text against a normalized list of `commonFalseDetectedWords`. Only filters if `segmentLoudness` is below `loudnessThresholdOf_commonFalseDetectedWords`.
        *   **Banned Words:** Uses case-insensitive `re.sub` for replacement. Extra spaces resulting from removal are cleaned up.
        *   **State Update:** Crucially, `stateManager.updateLastValidTranscriptionTime()` is only called if the `finalText` is non-empty *after all filtering*, ensuring idle timeouts are based on actual useful output.

    *   **ASR Handlers (`modelHandlers.py`):**
        *   **`AbstractAsrModelHandler`:** Defines the contract. `isModelLoaded()` is key. `cleanup()` ensures models are unloaded.
        *   **`WhisperModelHandler` (Local):**
            *   Critical dependency check for `transformers` and `torch` at init.
            *   `_determineDevice()`: Selects CUDA if available, else CPU. Configurable CPU override.
            *   `_cudaClean()`: `gc.collect()` then `torch.cuda.empty_cache()`.
            *   **Model Loading (`loadModel`):** Highly robust. Logs memory pre/post. Cleans CUDA before load. Handles `trust_remote_code` (configurable, defaults True), `torch_dtype` (fp16 on CUDA for efficiency). Logs detailed errors and hints on failure (model name, internet, dependencies, OOM, `trust_remote_code` issues). Sets `modelLoaded` state. Calls `_warmUpModel()` on success.
            *   `_warmUpModel()`: Transcribes a short silent clip to compile CUDA kernels/optimize, reducing latency of the first *actual* transcription.
            *   **Transcription (`transcribeAudioSegment`):** Expects `float32` audio. Warns if input `sampleRate` is not 16kHz (pipeline should resample). Uses `torch.no_grad()`. Returns `None` on critical failure, `""` for empty/silent successful transcription. Attempts CUDA cleanup on CUDA-related transcription errors.
        *   **`RemoteNemoClientHandler` (Remote):**
            *   Critical if `wslServerUrl` is not configured.
            *   `_makeServerRequest()`: Central request logic. Handles `requests.exceptions` (ConnectionError, Timeout, HTTPError), JSON decoding errors. Updates `self.serverReachable` and `self.modelLoaded` (e.g., on connection failure). Logs detailed hints for connection issues (WSL server, firewalls, IP).
            *   `checkServerStatus()`: Throttled by default. Updates `modelLoaded` and `serverReachable` based on `/status` endpoint response.
            *   `loadModel()`/`unloadModel()`: POST to `/load` or `/unload`. Interprets server status in response (e.g., "loading", "loaded", "unloaded", "already_unloaded").
            *   **Transcription (`transcribeAudioSegment`):** Sends audio as `multipart/form-data` to `/transcribe` with `sample_rate` and `target_lang` as query params. Expects float32 audio. Handles server errors or non-JSON/missing 'transcription' key in response. Returns `None` on critical failure.
            *   `cleanup()`: Optionally tells the server to unload its model based on `unloadRemoteModelOnExit` config.

    *   **`ModelLifecycleManager` (`managers.py`):**
        *   Runs in a background thread.
        *   **Unload Logic:** If `model_unloadTimeout > 0`, not recording, and model *is* loaded, checks `stateManager.timeSinceLastActivity()`. If timeout exceeded, calls `asrModelHandler.unloadModel()`. Plays notification on success.
        *   **Load Logic:** If recording and model *is not* loaded, calls `asrModelHandler.loadModel()`. Retries on failure after a short pause. Updates `lastActivityTime` after any load attempt.

    *   **`MainManager` (`mainManager.py`):**
        *   **WSL Server Launch (`_launchWslServer`, `_prepareWslLaunchCommand`):**
            *   Command preparation is robust: finds `wslNemoServer.py` relative to the main script or CWD. Converts Windows path to WSL path using `utils.convertWindowsPathToWsl`. Handles `wslUseSudo`.
            *   Launch uses `subprocess.Popen` with `CREATE_NO_WINDOW` (Windows) and redirects `stderr` to `stdout` to capture all server script output. `bufsize=1` for line buffering.
            *   Catches `FileNotFoundError` (e.g., `wsl.exe` not in PATH) and `PermissionError`.
        *   **WSL Server Reachability (`_waitForServerReachable`):** Polls `/status`. `checkProcessFirst=True` ensures quick detection of server script crashes before network polling timeouts.
        *   **WSL Output Logging (`_logWslProcessOutputOnError`):** Called on server error/exit. Uses `communicate()` to get remaining output. Specifically checks for common error strings like "sudo: a password is required", "Traceback", "Address already in use", logging specific hints.
        *   **WSL Termination (`_terminateWslServer`):** Attempts graceful `terminate()`, waits, then `kill()` if necessary. Logs output after termination attempts.
        *   **Thread Management (`_startBackgroundThreads`, `_cleanup`):** `threadWrapper` provides common exception handling for threads; critical threads failing (Keyboard, Transcription) will signal `stateManager.stopProgram()`. `_cleanup` ensures sentinels are sent and threads are joined with timeouts.
        *   **Initial Setup (`_runInitialSetup`):** Order of operations is important: WSL server reachability before model load attempt. Critical failure during local Whisper load aborts application start.
        *   **Main Loop (`run`):**
            *   `_runCheckTimeoutsAndGlobalState`: Handles program/recording/idle timeouts.
            *   `_runManageAudioStreamLifecycle`: Starts/stops `AudioHandler.stream` based on `StateManager.isRecording()`.
            *   `_runProcessAudioChunks`: Pulls from `AudioHandler.audioQueue` and pushes to `RealTimeAudioProcessor.audioBuffer`.
            *   `_runQueueTranscriptionRequest`: Calls `RealTimeAudioProcessor.checkTranscriptionTrigger()` and queues data if output enabled and trigger met.
        *   **`forceTranscribeCurrentBuffer()`:** New method. Directly calls `realTimeProcessor.getAudioBufferCopyAndClear()`, queues the result if valid and output is enabled, and updates `lastActivityTime`.

    *   **`DynamicLogger` & `utils.py` Logging:**
        *   `configure_dynamic_logging()`: Central setup point.
        *   **`utils.logError`/`logCritical` `exc_info` handling:** The logic was refined to ensure that if `exc_info=True` is passed from the call site, `DynamicLogger` is ultimately given `True` (so it can call `sys.exc_info()` itself) or an actual exception tuple if one was passed directly. This prevents the `TypeError: 'bool' object is not subscriptable` by ensuring the `LogRecord`'s `exc_info` attribute is correctly populated as a tuple by the time formatters process it.
        *   Fallback to `PrintLoggerFallback` in `utils.py` if `DynamicLogger` fails to initialize, ensuring some form of logging is always available.
        *   Silencing of third-party libraries is done by getting their standard Python logger instance and setting its level.

    *   **`utils.convertWindowsPathToWsl`:**
        *   Handles drive letters (e.g., `C:\...` -> `/mnt/c/...`).
        *   Attempts basic UNC path conversion (e.g., `\\server\share` -> `//server/share`), warning that this depends on WSL mount config.
        *   Requires absolute paths; attempts to `resolve()` relative paths but will fail if it remains relative.
        *   Uses conditional logging helpers (`log_warning_func`, etc.) to avoid errors if `appDynamicLogger` isn't initialized when it's called (e.g., very early by `MainManager._prepareWslLaunchCommand`).

    *   **`wslNemoServer.py` (Server-Side Specifics):**
        *   **Immediate Flask Start:** `app = Flask(__name__)` and `app.run()` are executed early. Model loading happens in a background thread if `--load_on_start` is used. This ensures the server is responsive to `/status` or `/load` requests even if NeMo model loading is slow or fails.
        *   **Dynamic NeMo Import:** `from nemo.collections.asr.models import ASRModel` is called inside `NemoServerModelHandler.loadModel()` and defensively in `transcribeAudioData()`. This allows the Flask server to start and function (e.g., report status) even if NeMo is not installed correctly or has import issues, which would then be reported as a load error.
        *   **`loadInProgress` flag:** Prevents multiple concurrent load attempts on the server.
        *   **Error Reporting:** `/status` endpoint reports "error" and includes `loadError` message if loading failed. `/transcribe` returns 503 if model not loaded or loading.
        *   **Thread Safety (Implicit):** Flask handles requests in separate threads. The `NemoServerModelHandler` methods (load, unload, transcribe) are designed to be called from these threads. Global state like `self.model`, `self.modelLoaded`, `self.loadInProgress` should ideally have locks if complex concurrent modifications were expected, but current usage (e.g., `loadInProgress` check, background load thread) mitigates most direct conflicts. Transcription itself with a loaded NeMo model is generally thread-safe for inference.

**III. Developer Overview: Implementation "How-To"**

This section provides a more narrativestyle overview of how key features are implemented for developers new to the project.

*   **Core Application Flow (`useRealtimeTranscription.py` -> `MainManager`):**
    1.  `useRealtimeTranscription.py` sets up user configuration (`userSettings`) and initializes `DynamicLogger` via `utils.configure_dynamic_logging()`.
    2.  It instantiates `SpeechToTextOrchestrator` from `mainManager.py`, passing `userSettings`.
    3.  `SpeechToTextOrchestrator.__init__`:
        *   Creates `ConfigurationManager`.
        *   Initializes core components: `StateManager`, `SystemInteractionHandler` (which sets up hotkey listeners, Pygame, PyAutoGUI), `AudioHandler` (sets up `sounddevice`), `RealTimeAudioProcessor`, `TranscriptionOutputHandler`.
        *   `_initializeAsrHandler()`: Based on `modelName` in config, instantiates either `WhisperModelHandler` or `RemoteNemoClientHandler`. If NeMo, `_prepareWslLaunchCommand()` builds the `wsl.exe` command.
        *   Creates `ModelLifecycleManager`.
    4.  `SpeechToTextOrchestrator.run()`:
        *   `_runInitialSetup()`:
            *   If NeMo, calls `_launchWslServer()` which uses `subprocess.Popen` to start `wslNemoServer.py` and then `_waitForServerReachable()` to poll its `/status`.
            *   Calls `asrModelHandler.loadModel()` (which for remote, sends a `/load` request if needed; for local, loads the model into VRAM/RAM).
            *   `_startBackgroundThreads()`: Launches threads for keyboard monitoring, model lifecycle management, and the transcription worker.
            *   Starts audio stream via `audioHandler.startStream()` if `isRecordingActive` is true.
        *   **Main Loop:** Continuously:
            *   Checks timeouts and global program state (`_runCheckTimeoutsAndGlobalState`).
            *   Manages audio stream on/off based on `StateManager` (`_runManageAudioStreamLifecycle`).
            *   If recording, gets audio chunks from `AudioHandler` and feeds them to `RealTimeAudioProcessor` (`_runProcessAudioChunks`).
            *   `RealTimeAudioProcessor.checkTranscriptionTrigger()` determines if a segment is ready (based on dictation/interval mode). If so, the audio segment is put onto the `transcriptionRequestQueue` (`_runQueueTranscriptionRequest`).
            *   The `_transcriptionWorkerLoop` (separate thread) gets segments from this queue, calls `asrModelHandler.transcribeAudioSegment()`, and passes the result to `outputHandler.processTranscriptionResult()`.
            *   `TranscriptionOutputHandler` filters and formats the text, then calls `systemInteractionHandler.typeText()` if output is enabled.
        *   `_cleanup()`: Stops threads, audio stream, WSL server (if launched), and ASR handler.

*   **Adding a New Hotkey (e.g., "Force Transcription"):**
    1.  **Config (`useRealtimeTranscription.py`):** Add ` "forceTranscriptionKey": "ctrl+,"` to `userSettings`.
    2.  **SystemInteractionHandler:**
        *   In `__init__`: Add `self.lastForceTranscriptionTime = 0.0`.
        *   In `monitorKeyboardShortcuts`:
            *   Get `forceTranscriptionKeyConfig = self.config.get('forceTranscriptionKey')`.
            *   Parse it (e.g., into `modifierKey='ctrl'`, `mainKey=','`).
            *   Add `if forceMainKey and keyboard.is_pressed(modifierKey or True) and keyboard.is_pressed(mainKey):`.
            *   Inside, check `if (time.time() - self.lastForceTranscriptionTime) > self.forceTranscriptionCooldown:`.
            *   Call `orchestrator.forceTranscribeCurrentBuffer()`.
            *   Update `self.lastForceTranscriptionTime`.
            *   Call `self._waitForKeyRelease(mainKey)`.
    3.  **MainManager (`SpeechToTextOrchestrator`):**
        *   Add `def forceTranscribeCurrentBuffer(self):`.
        *   Inside, call `audio_data = self.realTimeProcessor.getAudioBufferCopyAndClear()`.
        *   If `audio_data` is valid and output is enabled, get `sampleRate`, put `(audio_data, sampleRate)` on `self.transcriptionRequestQueue`. Update `self.stateManager.updateLastActivityTime()`.
    4.  **RealTimeAudioProcessor:**
        *   Add `def getAudioBufferCopyAndClear(self):`.
        *   Inside, copy `self.audioBuffer`, call `self.clearBuffer()`, reset `self.lastTranscriptionTriggerTime`, and return the copy.

*   **`DynamicLogger` Implementation:**
    *   `DynamicLogger` is instantiated once (globally in `utils.py` as `appDynamicLogger` via `configure_dynamic_logging`).
    *   `logConfigSets` (like `default` or `verboseConsole` from `define_logConfigSets.py`) provide base styling for handlers.
    *   `highOrderOptions` allow targeted overrides (e.g., exclude logs from `MyClass.some_method`, or force a specific `configSetName` for logs with `indicatorName="AUDIT"`).
    *   `log()` method is central:
        1.  Gets caller info (`_getCallerAndIndicator`) for `funcMethIndicator` and `%(callerInfo)s`.
        2.  Checks `exclude` rules (HO > Inline).
        3.  Resolves effective message level (HO > Inline > Initial) and applies suppression if original level was lower than an override.
        4.  Resolves `configSetName` (HO > Inline > Default).
        5.  Iterates handlers in the chosen set. For each handler:
            *   Resolves final `printToConsole`, `writeToFile`, `filePath`, `logFormat`, `timestampFormat` using `_resolveConfigValue` (HO > Inline > Handler Setting > Default).
            *   `filePath=None` in a config set triggers auto-path generation like `logs/{loggerName}_{configSetName}_h{handlerIndex}_auto.log`.
            *   Gets/creates cached handler instance via `_getHandlerInstance()` (key includes type, destination, base format/level).
        6.  `_prepareAndDispatchRecord()`: Creates `LogRecord` (with custom `callerInfo`, `indicatorName`), temporarily applies per-call format/timestamp to handler if different from its base, and calls `handler.handle(record)`.
