# systemInteractions.py
import os
import platform
import shutil  # For finding clip.exe
import subprocess
import time
import traceback

import keyboard

# Import logging helpers from utils
from utils import logWarning, logDebug, logInfo, logError


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
        self.systemInteractionHandler = None  # To be set by orchestrator or passed to methods

    # No _logDebug needed here, use imported logDebug directly
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
            logError(f"Unexpected error during pygame mixer setup: {e}")
            self.isMixerInitialized = False

    def _setupPyautogui(self):
        """Attempts to import and initialize PyAutoGUI if on Windows."""
        if platform.system() == "Windows":
            try:
                import pyautogui
                pyautogui.size()
                self._pyautoguiAvailable = True
                logInfo("PyAutoGUI loaded successfully (for potential Windows native typing).")
            except ImportError:
                self._pyautoguiErrorMessage = "PyAutoGUI library not found. Install it (`pip install pyautogui`) to enable typing output on Windows."
                logWarning(self._pyautoguiErrorMessage)
                self._pyautoguiAvailable = False
            except Exception as e:
                self._pyautoguiErrorMessage = f"PyAutoGUI could not initialize on Windows (maybe no display?): {e}. Typing output will be disabled."
                logWarning(self._pyautoguiErrorMessage)
                self._pyautoguiAvailable = False
        else:
            self._pyautoguiAvailable = False

    def _setupAudioNotifications(self):
        """Loads sound file paths if mixer is initialized."""
        if not self.isMixerInitialized:
            return
        soundMap = {
            "modelUnloaded": "modelUnloaded.mp3",
            "outputDisabled": "outputDisabled.mp3",
            "outputEnabled": "outputEnabled.mp3",
            "recordingOff": "recordingOff.mp3",
            "recordingOn": "recordingOn.mp3"
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
                logWarning(f"Notification sound file not found: {path}")
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

    def playNotification(self, soundName, forcePlay=False):  # Added forcePlay parameter
        """Plays a notification sound if available and enabled."""
        if not forcePlay and not self.config.get('enableAudioNotifications', True):
            # logDebug(f"Skipping sound '{soundName}' - notifications disabled.") # Keep commented if too noisy
            return
        if not forcePlay and soundName in ['recordingOn', 'outputEnabled'] and not self.config.get(
                'playEnableSounds', False):
            # logDebug(f"Skipping enable sound '{soundName}'.") # Keep commented if too noisy
            return
        if not self.isMixerInitialized or soundName not in self.audioFiles:
            # logDebug(f"Cannot play sound '{soundName}'. Mixer: {self.isMixerInitialized}, Sound exists: {soundName in self.audioFiles}") # Keep commented if too noisy
            return
        # Import should be safe here as we checked isMixerInitialized
        import pygame
        soundPath = self.audioFiles[soundName]
        try:
            # Consider loading sounds once instead of every time playNotification is called
            # For simplicity, loading here is okay for infrequent notifications.
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
        # Set stateManager and systemInteractionHandler if needed by other methods called from here
        self.stateManager = orchestrator.stateManager
        self.systemInteractionHandler = self  # Reference self for modifier key checks if needed in typeText
        recordingToggleKey = self.config.get('recordingToggleKey')
        outputToggleKey = self.config.get('outputToggleKey')
        if not recordingToggleKey or not outputToggleKey:
            logError(
                "Hotkeys not configured ('recordingToggleKey' or 'outputToggleKey'). Keyboard monitor thread stopping.")
            orchestrator.stateManager.stopProgram()
            return
        try:
            # Test if keyboard library can be accessed (might raise permission error)
            _ = keyboard.is_pressed('shift')
            logInfo("Keyboard library access test successful.")
            while orchestrator.stateManager.shouldProgramContinue():
                try:
                    # Use keyboard.is_pressed which is generally less blocking than wait
                    if keyboard.is_pressed(recordingToggleKey):
                        logDebug(f"Hotkey '{recordingToggleKey}' pressed.")
                        orchestrator.toggleRecording()
                        self._waitForKeyRelease(
                            recordingToggleKey)  # Wait for release to avoid rapid toggling
                    if keyboard.is_pressed(outputToggleKey):
                        logDebug(f"Hotkey '{outputToggleKey}' pressed.")
                        orchestrator.toggleOutput()
                        self._waitForKeyRelease(outputToggleKey)
                    # Short sleep to prevent high CPU usage
                    time.sleep(0.05)
                except Exception as keyCheckError:
                    # Log errors occurring during the check phase
                    logError(
                        f"Error checking key press: {keyCheckError}. Hotkeys may stop working.")
                    # Avoid spamming logs if error repeats quickly
                    time.sleep(1)
        except ImportError:
            logError("Keyboard library not installed (`pip install keyboard`). Hotkeys disabled.")
            exitReason = "ImportError"
            orchestrator.stateManager.stopProgram()  # Signal main loop to stop
        except Exception as e:
            # Catch permission errors or others during initial test or loop setup
            logError(f"Unhandled exception in keyboard monitoring setup/loop: {e}")
            logError("Hint: If on Linux, ensure user is in 'input' group or run with sudo.")
            logError("Hint: If on Windows, try running as Administrator.")
            logError(traceback.format_exc())
            exitReason = f"Unhandled Exception: {e}"
            orchestrator.stateManager.stopProgram()  # Signal main loop to stop
        finally:
            # Log thread exit regardless of reason
            logInfo("Keyboard shortcut monitor thread stopping.")

    def _waitForKeyRelease(self, key):
        """Waits until the specified key is released to prevent rapid toggling."""
        startTime = time.time()
        timeout = 2.0  # seconds
        try:
            # Use a loop with a short sleep to check for release
            while keyboard.is_pressed(key):
                if time.time() - startTime > timeout:
                    logDebug(f"Timeout waiting for key release '{key}'.")
                    break  # Avoid getting stuck indefinitely
                time.sleep(0.05)
            logDebug(f"Hotkey '{key}' released.")
        except Exception as e:
            # keyboard library might raise errors here too
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
        # Add a trailing space for better separation between outputs
        textToOutput = text  # Do not add space here, add it after typing/copying if needed
        if self.textOutputMethod == "pyautogui":
            if self._pyautoguiAvailable:
                # Get typing mode from config, default to 'letter'
                typingMode = self.config.get('typingMode', 'letter')
                logDebug(f"Executing PyAutoGUI output with mode: '{typingMode}'")
                try:
                    # PyAutoGUI needs to be imported here if not already globally
                    import pyautogui  # Ensure it's imported within the method context
                    if typingMode == "letter":
                        # Original behavior: letter by letter with small interval
                        pyautogui.write(textToOutput, interval=0.01)
                        pyautogui.write(" ", interval=0.01)  # Add space after
                        logDebug(f"Typed letter-by-letter: '{text[:50]}...'")
                    elif typingMode == "word":
                        # Word by word: split, type word, type space, pause
                        wordInterval = 0.0  # Type words fast
                        spaceInterval = 0.0  # Type space fast
                        pauseBetweenWords = 0.05  # Seconds to pause between words
                        words = textToOutput.split()  # Split by whitespace
                        for i, word in enumerate(words):
                            # Check state inside loop for responsiveness
                            if self.stateManager and not self.stateManager.isOutputEnabled() or self.isModifierKeyPressed(
                                    "ctrl"):
                                logDebug(
                                    "Output disabled or CTRL pressed during word-by-word typing, stopping.")
                                break  # Allow interruption
                            pyautogui.write(word, interval=wordInterval)
                            pyautogui.write(' ',
                                            interval=spaceInterval)  # Always add space after word
                            time.sleep(pauseBetweenWords)  # Pause after typing word and space
                        logDebug(f"Typed word-by-word: '{text[:50]}...'")
                    elif typingMode == "whole":
                        # Type the whole string as fast as possible
                        pyautogui.write(textToOutput, interval=0)
                        pyautogui.write(" ", interval=0)  # Add space after
                        logDebug(f"Typed whole text: '{text[:50]}...'")
                    else:
                        logWarning(f"Unknown typingMode '{typingMode}'. Defaulting to 'letter'.")
                        pyautogui.write(textToOutput, interval=0.01)
                        pyautogui.write(" ", interval=0.01)  # Add space after
                except ImportError:
                    logError("PyAutoGUI cannot be imported inside typeText. Typing disabled.")
                    self._pyautoguiAvailable = False  # Mark as unavailable
                except Exception as e:
                    logWarning(f"PyAutoGUI write failed during execution (mode: {typingMode}): {e}")
                    # Optional: Consider disabling it temporarily if it fails repeatedly
                    # self._pyautoguiAvailable = False
            # else: # Logged during init, no need to repeat here unless debugging
            #     logDebug("Typing skipped: PyAutoGUI method selected but unavailable/failed init.")
        elif self.textOutputMethod == "clipboard":
            # Clipboard method inherently outputs the "whole" text at once.
            # Typing mode configuration is ignored.
            if self.clipExePath:
                logDebug(f"Executing clipboard output (typingMode ignored). Text: '{text[:50]}...'")
                try:
                    # Use subprocess.run for simplicity
                    process = subprocess.run(
                        [self.clipExePath],
                        input=textToOutput + " ",  # Use text with trailing space for clipboard
                        encoding='utf-8',  # Specify encoding
                        check=True,  # Raise exception on non-zero exit code
                        capture_output=True  # Capture stdout/stderr
                    )
                    logDebug(f"Copied text to Windows clipboard: '{text[:50]}...'")
                except FileNotFoundError:
                    logError(
                        f"Error copying to clipboard: '{self.clipExePath}' not found. Disabling clipboard output.")
                    self.clipExePath = None  # Mark unavailable
                    self.textOutputMethod = "none"
                except subprocess.CalledProcessError as e:
                    logError(f"Error running clip.exe (return code {e.returncode}): {e}")
                    # Attempt to decode stderr for more info
                    stderrOutput = "N/A"
                    if e.stderr:
                        try:
                            stderrOutput = e.stderr.decode('utf-8', errors='ignore').strip()
                        except Exception:
                            stderrOutput = "(Could not decode stderr)"
                    logError(f"clip.exe stderr: {stderrOutput}")
                except Exception as e:
                    logError(f"Unexpected error copying text to clipboard: {e}")
            # else: # Logged during init or if FileNotFoundError occurs
            #     logDebug("Clipboard copy skipped: Method selected but clip.exe unavailable.")
        # else: # textOutputMethod == "none"
        # logDebug("Text output skipped: Method is 'none'.") # Can be noisy if logged every time

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
