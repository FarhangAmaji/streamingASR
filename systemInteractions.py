# systemInteractions.py
import os
import platform
import shutil  # For finding clip.exe
import subprocess
import time
import traceback

import keyboard

# Import logging helpers from utils
from utils import logWarning, logDebug, logInfo, logError, logCritical


# Pygame and PyAutoGUI are imported conditionally later where needed


# ==================================
# System Interaction (Hotkeys, Notifications, Output)
# ==================================
class SystemInteractionHandler:
    """
    Manages interactions with keyboard for hotkeys, pygame for sound notifications,
    and handles text output via simulated typing (Windows native) or clipboard (WSL).
    """

    def __init__(self, config):
        self.config = config
        self.audioFiles = {}
        self.isMixerInitialized = False
        self._pyautoguiAvailable = False
        self._pyautoguiErrorMessage = ""
        self._setupPygame()
        self._setupPyautogui()
        self._setupAudioNotifications()
        self.textOutputMethod = "none"
        self.clipExePath = None
        self.isWslEnvironment = False
        self._determineTextOutputMethod()
        # Attributes needed for interaction with orchestrator/state within methods
        self.stateManager = None  # To be set by orchestrator or passed to methods
        # Attributes for hotkey cooldowns
        self.lastForceTranscriptionTime = 0.0  # Timestamp for the force transcription hotkey
        self.forceTranscriptionCooldown = 0.5  # Cooldown duration in seconds

    def _setupPygame(self):
        """Initializes pygame mixer."""
        try:
            import pygame
            pygame.mixer.init()
            self.isMixerInitialized = True
            logInfo("Pygame mixer initialized for audio notifications.")
        except ImportError:
            logWarning(
                "Pygame library not found (`pip install pygame`). Audio notifications disabled.")
            self.isMixerInitialized = False
        except pygame.error as e:
            logWarning(f"Failed to initialize pygame mixer: {e}. Audio notifications disabled.")
            self.isMixerInitialized = False
        except Exception as e:
            logCritical(f"Unexpected error during pygame mixer setup: {e}", exc_info=True)
            self.isMixerInitialized = False

    def _setupPyautogui(self):
        """Attempts to import and initialize PyAutoGUI if on Windows."""
        if platform.system() == "Windows":
            try:
                import pyautogui
                pyautogui.size()  # A simple call to check if it's functional
                self._pyautoguiAvailable = True
                logInfo("PyAutoGUI loaded successfully (for potential Windows native typing).")
            except ImportError:
                self._pyautoguiErrorMessage = "PyAutoGUI library not found. Install it (`pip install pyautogui`) to enable typing output on Windows."
                logWarning(self._pyautoguiErrorMessage)
                self._pyautoguiAvailable = False
            except Exception as e:  # Catches other errors like display not found
                self._pyautoguiErrorMessage = f"PyAutoGUI could not initialize on Windows (maybe no display?): {e}. Typing output will be disabled."
                logWarning(self._pyautoguiErrorMessage)
                self._pyautoguiAvailable = False
        else:
            self._pyautoguiAvailable = False  # Not Windows, so not available by this logic

    def _setupAudioNotifications(self):
        """Loads sound file paths if mixer is initialized."""
        if not self.isMixerInitialized:
            return
        soundMap = {
            "modelUnloaded": "modelUnloaded.mp3",
            "outputDisabled": "outputDisabled.mp3",
            "outputEnabled": "outputEnabled.mp3",
            "recordingOff": "recordingOff.mp3",
            "recordingOn": "recordingOn.mp3",
            "forceTranscribe": "forceTranscribe.mp3"  # Optional: new sound for this action
        }
        scriptDir = self.config.get('scriptDir')
        if not scriptDir:
            logError("Cannot load notification sounds: scriptDir not found in config.")
            return
        loadedCount = 0
        for name, filename in soundMap.items():
            path = scriptDir / filename
            if path.is_file():
                self.audioFiles[name] = str(path)
                loadedCount += 1
            else:
                if name != "forceTranscribe":  # Don't warn for optional new sound if missing
                    logWarning(f"Notification sound file not found: {path}")
                else:
                    logDebug(
                        f"Optional notification sound file 'forceTranscribe.mp3' not found at {path}. This sound will not play.")

        if loadedCount > 0:
            logInfo(f"Loaded {loadedCount} audio notification files.")
        else:
            logWarning("No audio notification files were loaded.")

    def _determineTextOutputMethod(self):
        """Determines the best available text output method based on OS and config."""
        outputEnabledByConfig = self.config.get('enableTypingOutput', True)
        osName = platform.system()
        # Check for WSL environment variables
        if "WSL_DISTRO_NAME" in os.environ or "WSL_INTEROP" in os.environ:
            self.isWslEnvironment = True
            logInfo("WSL environment detected.")
        elif osName == "Windows":
            logInfo("Windows Native environment detected.")
        else:
            logInfo(f"Non-Windows/Non-WSL environment detected ({osName}).")

        if outputEnabledByConfig:
            if osName == "Windows" and not self.isWslEnvironment:
                if self._pyautoguiAvailable:
                    self.textOutputMethod = "pyautogui"
                    logInfo("Text Output Method: PyAutoGUI (Windows Native Typing)")
                else:
                    logWarning(
                        f"PyAutoGUI is unavailable or failed to initialize ({self._pyautoguiErrorMessage}). Text output disabled.")
                    self.textOutputMethod = "none"
            elif self.isWslEnvironment:
                self.clipExePath = shutil.which('clip.exe')
                if self.clipExePath:
                    self.textOutputMethod = "clipboard"
                    logInfo(f"Text Output Method: Windows Clipboard via '{self.clipExePath}' (WSL)")
                else:
                    logWarning("Text output disabled in WSL: 'clip.exe' not found in PATH.")
                    self.textOutputMethod = "none"
            else:  # Other Linux, macOS, etc.
                logInfo(
                    f"Simulated text output (typing/clipboard) is not configured for this OS ({osName}).")
                self.textOutputMethod = "none"
        else:
            logInfo("Text output globally disabled by configuration ('enableTypingOutput': False).")
            self.textOutputMethod = "none"

    def playNotification(self, soundName, forcePlay=False):
        """Plays a notification sound if available and enabled."""
        if not forcePlay and not self.config.get('enableAudioNotifications', True):
            # logDebug(f"Skipping sound '{soundName}' - notifications disabled.")
            return
        if not forcePlay and soundName in ['recordingOn', 'outputEnabled',
                                           'forceTranscribe'] and not self.config.get(
            'playEnableSounds', False):
            # logDebug(f"Skipping enable sound '{soundName}'.")
            return

        if not self.isMixerInitialized or soundName not in self.audioFiles:
            # logDebug(f"Cannot play sound '{soundName}'. Mixer: {self.isMixerInitialized}, Sound exists: {soundName in self.audioFiles}")
            return
        import pygame  # Import should be safe here as we checked isMixerInitialized
        soundPath = self.audioFiles[soundName]
        try:
            sound = pygame.mixer.Sound(soundPath)
            sound.play()
            logDebug(f"Played notification sound: {soundName}")
        except Exception as e:
            logError(f"Error playing notification sound '{soundPath}': {e}")

    def monitorKeyboardShortcuts(self, orchestrator):
        """
        Runs in a thread to monitor global hotkeys. Calls methods on the orchestrator.
        Stops when orchestrator's state indicates program should stop.
        """
        logInfo("Starting keyboard shortcut monitor thread.")
        self.stateManager = orchestrator.stateManager  # Set for use in typeText if needed for modifier checks

        recordingToggleKey = self.config.get('recordingToggleKey')
        outputToggleKey = self.config.get('outputToggleKey')
        forceTranscriptionKeyConfig = self.config.get(
            'forceTranscriptionKey')  # Get the configured string e.g., "ctrl+,"

        # Parse the forceTranscriptionKeyConfig into modifier and main key if it's a combo
        forceModifierKey = None
        forceMainKey = None
        if forceTranscriptionKeyConfig and isinstance(forceTranscriptionKeyConfig, str):
            parts = forceTranscriptionKeyConfig.lower().split('+')
            if len(parts) > 1:  # It's a combo like "ctrl+,"
                forceModifierKey = parts[0].strip()
                forceMainKey = parts[-1].strip()  # Get the last part as the main key
                logDebug(
                    f"Parsed force transcription hotkey: Modifier='{forceModifierKey}', MainKey='{forceMainKey}'")
            else:  # Single key
                forceMainKey = parts[0].strip()
                logDebug(
                    f"Parsed force transcription hotkey: MainKey='{forceMainKey}' (no modifier)")
        else:
            logWarning(
                "Optional 'forceTranscriptionKey' not configured or invalid. This hotkey will be disabled.")

        if not recordingToggleKey or not outputToggleKey:
            logCritical(
                "Core hotkeys not configured ('recordingToggleKey' or 'outputToggleKey'). Keyboard monitor thread stopping.")
            orchestrator.stateManager.stopProgram()
            return

        try:
            _ = keyboard.is_pressed('shift')  # A simple test
            logInfo("Keyboard library access test successful.")

            while orchestrator.stateManager.shouldProgramContinue():
                try:
                    currentTime = time.time()

                    # Recording Toggle Hotkey
                    if recordingToggleKey and keyboard.is_pressed(recordingToggleKey):
                        logDebug(f"Hotkey '{recordingToggleKey}' pressed.")
                        orchestrator.toggleRecording()
                        self._waitForKeyRelease(recordingToggleKey)

                    # Output Toggle Hotkey
                    if outputToggleKey and keyboard.is_pressed(outputToggleKey):
                        logDebug(f"Hotkey '{outputToggleKey}' pressed.")
                        orchestrator.toggleOutput()
                        self._waitForKeyRelease(outputToggleKey)

                    # Force Transcription Hotkey (if configured and parsed)
                    if forceMainKey:  # Check if forceMainKey was successfully parsed
                        mainKeyPressed = keyboard.is_pressed(forceMainKey)
                        modifierPressed = keyboard.is_pressed(
                            forceModifierKey) if forceModifierKey else True  # True if no modifier

                        if mainKeyPressed and modifierPressed:
                            logDebug(
                                f"Hotkey '{forceTranscriptionKeyConfig}' (parsed: mod='{forceModifierKey}', key='{forceMainKey}') pressed.")
                            if (
                                    currentTime - self.lastForceTranscriptionTime) > self.forceTranscriptionCooldown:
                                logInfo("Force transcription hotkey activated.")
                                orchestrator.forceTranscribeCurrentBuffer()
                                self.playNotification("forceTranscribe")
                                self.lastForceTranscriptionTime = currentTime
                            else:
                                logDebug("Force transcription hotkey in cooldown.")
                            # Wait for the main action key release
                            self._waitForKeyRelease(forceMainKey)
                            # Optionally, also wait for modifier release if it causes issues,
                            # but usually waiting for the main key is sufficient.
                            # if forceModifierKey: self._waitForKeyRelease(forceModifierKey)

                    time.sleep(0.05)  # Short sleep to prevent high CPU usage
                except Exception as keyCheckError:
                    # This will catch errors from keyboard.is_pressed if an invalid key name from config is used
                    # (e.g., if other hotkeys were misconfigured)
                    logError(
                        f"Error checking key press: {keyCheckError}. Hotkeys may stop working.",
                        exc_info=True)  # Pass True to capture actual exception for logging
                    time.sleep(1)  # Avoid spamming logs if error repeats quickly

        except ImportError:
            logCritical(
                "Keyboard library not installed (`pip install keyboard`). Hotkeys disabled. Stopping program.",
                exc_info=True)
            orchestrator.stateManager.stopProgram()
        except Exception as e:
            logCritical(f"Unhandled exception in keyboard monitoring setup/loop: {e}",
                        exc_info=True)
            logError("Hint: If on Linux, ensure user is in 'input' group or run with sudo.")
            logError("Hint: If on Windows, try running as Administrator.")
            orchestrator.stateManager.stopProgram()
        finally:
            logInfo("Keyboard shortcut monitor thread stopping.")

    def _waitForKeyRelease(self, key):
        """
        Waits until the specified key is released to prevent rapid toggling.
        The 'key' argument should be a single key name that keyboard.is_pressed() understands.
        """
        startTime = time.time()
        timeout = 2.0  # seconds
        # Ensure key is not None or empty before proceeding, though callers should ensure this.
        if not key:
            logWarning("_waitForKeyRelease called with empty key.")
            return
        try:
            # Use a loop with a short sleep to check for release
            while keyboard.is_pressed(key):
                if time.time() - startTime > timeout:
                    logDebug(f"Timeout waiting for key release '{key}'.")
                    break  # Avoid getting stuck indefinitely
                time.sleep(0.05)
            logDebug(f"Hotkey '{key}' released.")
        except Exception as e:
            # keyboard library might raise errors here too if key is invalid
            logWarning(f"Error checking key release for '{key}': {e}")

    def isModifierKeyPressed(self, key):
        """Checks if a specific modifier key (e.g., 'ctrl', 'alt', 'shift') is pressed."""
        try:
            return keyboard.is_pressed(key)
        except Exception as e:
            logDebug(f"Could not check modifier key '{key}': {e}")
            return False

    def typeText(self, text):
        """
        Outputs text using the method determined during initialization
        (PyAutoGUI typing on Windows native, clipboard copy on WSL).
        Supports different typing modes ('letter', 'word', 'whole') for PyAutoGUI.
        Assumes the check for outputEnabled happened before calling this.
        """
        if not text:  # Don't try to output empty text
            logDebug("typeText called with empty string, skipping output.")
            return

        textToOutput = text  # Do not add space here, add it after typing/copying if needed

        if self.textOutputMethod == "pyautogui":
            if self._pyautoguiAvailable:
                typingMode = self.config.get('typingMode', 'letter')
                logDebug(f"Executing PyAutoGUI output with mode: '{typingMode}'")
                try:
                    import pyautogui
                    if typingMode == "letter":
                        pyautogui.write(textToOutput, interval=0.01)
                        pyautogui.write(" ", interval=0.01)
                        logDebug(f"Typed letter-by-letter: '{text[:50]}...'")
                    elif typingMode == "word":
                        wordInterval = 0.0
                        spaceInterval = 0.0
                        pauseBetweenWords = 0.05
                        words = textToOutput.split()
                        for i, word in enumerate(words):
                            # Check stateManager here directly as it's set in monitorKeyboardShortcuts
                            if self.stateManager and not self.stateManager.isOutputEnabled() or self.isModifierKeyPressed(
                                    "ctrl"):
                                logDebug(
                                    "Output disabled or CTRL pressed during word-by-word typing, stopping.")
                                break
                            pyautogui.write(word, interval=wordInterval)
                            pyautogui.write(' ', interval=spaceInterval)
                            time.sleep(pauseBetweenWords)
                        logDebug(f"Typed word-by-word: '{text[:50]}...'")
                    elif typingMode == "whole":
                        pyautogui.write(textToOutput, interval=0)
                        pyautogui.write(" ", interval=0)
                        logDebug(f"Typed whole text: '{text[:50]}...'")
                    else:
                        logWarning(f"Unknown typingMode '{typingMode}'. Defaulting to 'letter'.")
                        pyautogui.write(textToOutput, interval=0.01)
                        pyautogui.write(" ", interval=0.01)
                except ImportError:
                    logError("PyAutoGUI cannot be imported inside typeText. Typing disabled.")
                    self._pyautoguiAvailable = False
                except Exception as e:
                    logWarning(f"PyAutoGUI write failed during execution (mode: {typingMode}): {e}")
            # else:
            #     logDebug("Typing skipped: PyAutoGUI method selected but unavailable/failed init.")
        elif self.textOutputMethod == "clipboard":
            if self.clipExePath:
                logDebug(f"Executing clipboard output (typingMode ignored). Text: '{text[:50]}...'")
                try:
                    process = subprocess.run(
                        [self.clipExePath],
                        input=textToOutput + " ",
                        encoding='utf-8',
                        check=True,
                        capture_output=True
                    )
                    logDebug(f"Copied text to Windows clipboard: '{text[:50]}...'")
                except FileNotFoundError:
                    logError(
                        f"Error copying to clipboard: '{self.clipExePath}' not found. Disabling clipboard output.")
                    self.clipExePath = None
                    self.textOutputMethod = "none"
                except subprocess.CalledProcessError as e:
                    logError(f"Error running clip.exe (return code {e.returncode}): {e}",
                             exc_info=True)
                    stderrOutput = "N/A"
                    if e.stderr:
                        try:
                            stderrOutput = e.stderr.decode('utf-8', errors='ignore').strip()
                        except Exception:
                            stderrOutput = "(Could not decode stderr)"
                    logError(f"clip.exe stderr: {stderrOutput}")
                except Exception as e:
                    logError(f"Unexpected error copying text to clipboard: {e}",
                             exc_info=True)
            # else:
            #     logDebug("Clipboard copy skipped: Method selected but clip.exe unavailable.")
        # else:
        #     logDebug("Text output skipped: Method is 'none'.")

    def cleanup(self):
        """Cleans up system interaction resources (pygame mixer)."""
        logDebug("SystemInteractionHandler cleanup.")
        if self.isMixerInitialized:
            try:
                import pygame
                pygame.mixer.quit()
                logInfo("Pygame mixer quit.")
            except Exception as e:
                logError(f"Error quitting pygame mixer: {e}")
