# systemInteractions.py
# ==============================================================================
# System Interaction (Hotkeys, Notifications, Output)
# ==============================================================================
#
# Purpose:
# - Manages interactions with the keyboard for global hotkeys using the 'keyboard' library.
# - Handles playing audio notifications using the 'pygame' library.
# - Manages text output by simulating typing on Windows native ('pyautogui') or
#   copying to the clipboard in a WSL environment.
# ==============================================================================
import os
import platform
import shutil
import subprocess
import time
import traceback

import keyboard
# Import the new universal logger instances
from utils.loggerInstance import uniLogger, uniDebugLogger


# Pygame and PyAutoGUI are imported conditionally later where needed to avoid errors if not installed.


# ==================================
# System Interaction (Hotkeys, Notifications, Output)
# ==================================
class SystemInteractionHandler:
    """
    Manages system-level interactions like hotkeys, sound notifications, and text output.
    """

    def __init__(self, config):
        """Initializes all handlers for system interaction."""
        self.config = config
        self.audioFiles = {}
        self.isMixerInitialized = False
        self._pyautoguiAvailable = False
        self._pyautoguiErrorMessage = ""
        # Setup handlers for sound, GUI automation, and audio files
        self._setupPygame()
        self._setupPyautogui()
        self._setupAudioNotifications()
        # Determine the method for text output (typing, clipboard, or none)
        self.textOutputMethod = "none"
        self.clipExePath = None
        self.isWslEnvironment = False
        self._determineTextOutputMethod()
        # State manager will be linked by the orchestrator after initialization
        self.stateManager = None
        # Cooldown state for the force transcription hotkey
        self.lastForceTranscriptionTime = 0.0
        self.forceTranscriptionCooldown = 0.5

    def _setupPygame(self):
        """Initializes the pygame mixer for playing notification sounds."""
        try:
            import pygame
            pygame.mixer.init()
            self.isMixerInitialized = True
            uniLogger.info("Pygame mixer initialized for audio notifications.")
        except ImportError:
            uniLogger.warning(
                "Pygame library not found (`pip install pygame`). Audio notifications disabled.")
            self.isMixerInitialized = False
        except pygame.error as e:
            uniLogger.warning(
                f"Failed to initialize pygame mixer: {e}. Audio notifications disabled.")
            self.isMixerInitialized = False
        except Exception as e:
            uniLogger.critical(f"Unexpected error during pygame mixer setup: {e}", excInfo=True)
            self.isMixerInitialized = False

    def _setupPyautogui(self):
        """Attempts to import and initialize PyAutoGUI for simulated typing on Windows."""
        if platform.system() == "Windows":
            try:
                import pyautogui
                pyautogui.size()  # A simple call to check if it's functional
                self._pyautoguiAvailable = True
                uniLogger.info("PyAutoGUI loaded successfully (for Windows native typing).")
            except ImportError:
                self._pyautoguiErrorMessage = "PyAutoGUI not found (`pip install pyautogui`). Typing output disabled on Windows."
                uniLogger.warning(self._pyautoguiErrorMessage)
                self._pyautoguiAvailable = False
            except Exception as e:  # Catches other errors like display not found
                self._pyautoguiErrorMessage = f"PyAutoGUI could not initialize on Windows (maybe no display?): {e}. Typing output will be disabled."
                uniLogger.warning(self._pyautoguiErrorMessage)
                self._pyautoguiAvailable = False
        else:
            self._pyautoguiAvailable = False

    def _setupAudioNotifications(self):
        """Loads the paths to notification sound files if the pygame mixer is ready."""
        if not self.isMixerInitialized:
            return
        soundMap = {
            "modelUnloaded": "modelUnloaded.mp3",
            "outputDisabled": "outputDisabled.mp3",
            "outputEnabled": "outputEnabled.mp3",
            "recordingOff": "recordingOff.mp3",
            "recordingOn": "recordingOn.mp3",
            "forceTranscribe": "forceTranscribe.mp3"  # Sound for force transcribe action
        }
        scriptDir = self.config.get('scriptDir')
        if not scriptDir:
            uniLogger.error("Cannot load notification sounds: scriptDir not found in config.")
            return
        loadedCount = 0
        for name, filename in soundMap.items():
            path = scriptDir / filename
            if path.is_file():
                self.audioFiles[name] = str(path)
                loadedCount += 1
            else:
                # Don't warn if the optional 'forceTranscribe' sound is missing
                if name != "forceTranscribe":
                    uniLogger.warning(f"Notification sound file not found: {path}")
                else:
                    uniDebugLogger.debug(
                        f"Optional notification sound 'forceTranscribe.mp3' not found at {path}.")
        if loadedCount > 0:
            uniLogger.info(f"Loaded {loadedCount} audio notification files.")
        else:
            uniLogger.warning("No audio notification files were loaded.")

    def _determineTextOutputMethod(self):
        """Determines the best available text output method based on OS and configuration."""
        outputEnabledByConfig = self.config.get('enableTypingOutput', True)
        osName = platform.system()
        # Check for WSL environment variables
        if "WSL_DISTRO_NAME" in os.environ or "WSL_INTEROP" in os.environ:
            self.isWslEnvironment = True
            uniLogger.info("WSL environment detected.")
        elif osName == "Windows":
            uniLogger.info("Windows Native environment detected.")
        else:
            uniLogger.info(f"Non-Windows/Non-WSL environment detected ({osName}).")

        if outputEnabledByConfig:
            # On native Windows, use PyAutoGUI for typing if available
            if osName == "Windows" and not self.isWslEnvironment:
                if self._pyautoguiAvailable:
                    self.textOutputMethod = "pyautogui"
                    uniLogger.info("Text Output Method: PyAutoGUI (Windows Native Typing)")
                else:
                    uniLogger.warning(
                        f"PyAutoGUI unavailable ({self._pyautoguiErrorMessage}). Text output disabled.")
                    self.textOutputMethod = "none"
            # In WSL, use clip.exe to copy text to the Windows clipboard
            elif self.isWslEnvironment:
                self.clipExePath = shutil.which('clip.exe')
                if self.clipExePath:
                    self.textOutputMethod = "clipboard"
                    uniLogger.info(
                        f"Text Output Method: Windows Clipboard via '{self.clipExePath}' (WSL)")
                else:
                    uniLogger.warning("Text output disabled in WSL: 'clip.exe' not found in PATH.")
                    self.textOutputMethod = "none"
            # On other OSes, text output is not supported
            else:
                uniLogger.info(
                    f"Simulated text output (typing/clipboard) is not configured for this OS ({osName}).")
                self.textOutputMethod = "none"
        else:
            uniLogger.info(
                "Text output globally disabled by configuration ('enableTypingOutput': False).")
            self.textOutputMethod = "none"

    def playNotification(self, soundName, forcePlay=False):
        """Plays a notification sound if available and enabled in the configuration."""
        # Skip if notifications are disabled, unless forced
        if not forcePlay and not self.config.get('enableAudioNotifications', True):
            return
        # Skip "enable" sounds if configured to do so
        if not forcePlay and soundName in ['recordingOn', 'outputEnabled',
                                           'forceTranscribe'] and not self.config.get(
                'playEnableSounds', False):
            return

        if not self.isMixerInitialized or soundName not in self.audioFiles:
            return
        import pygame
        soundPath = self.audioFiles[soundName]
        try:
            sound = pygame.mixer.Sound(soundPath)
            sound.play()
            uniDebugLogger.debug(f"Played notification sound: {soundName}")
        except Exception as e:
            uniLogger.error(f"Error playing notification sound '{soundPath}': {e}")

    def monitorKeyboardShortcuts(self, orchestrator):
        """
        Runs in a background thread to monitor global hotkeys and trigger orchestrator actions.
        """
        uniLogger.info("Starting keyboard shortcut monitor thread.")
        self.stateManager = orchestrator.stateManager

        # Get hotkey configurations
        recordingToggleKey = self.config.get('recordingToggleKey')
        outputToggleKey = self.config.get('outputToggleKey')
        forceTranscriptionKey = self.config.get('forceTranscriptionKey')

        if not recordingToggleKey or not outputToggleKey:
            uniLogger.critical("Core hotkeys not configured. Keyboard monitor stopping.")
            orchestrator.stateManager.stopProgram()
            return

        try:
            # Test keyboard library access
            _ = keyboard.is_pressed('shift')
            uniLogger.info("Keyboard library access test successful.")
            # Main monitoring loop
            while orchestrator.stateManager.shouldProgramContinue():
                try:
                    currentTime = time.time()
                    # Check for recording toggle hotkey
                    if keyboard.is_pressed(recordingToggleKey):
                        uniDebugLogger.debug(f"Hotkey '{recordingToggleKey}' pressed.")
                        orchestrator.toggleRecording()
                        self._waitForKeyRelease(recordingToggleKey)
                    # Check for output toggle hotkey
                    if keyboard.is_pressed(outputToggleKey):
                        uniDebugLogger.debug(f"Hotkey '{outputToggleKey}' pressed.")
                        orchestrator.toggleOutput()
                        self._waitForKeyRelease(outputToggleKey)
                    # Check for force transcription hotkey
                    if forceTranscriptionKey and keyboard.is_pressed(forceTranscriptionKey):
                        uniDebugLogger.debug(f"Hotkey '{forceTranscriptionKey}' pressed.")
                        # Apply a cooldown to prevent rapid firing
                        if (
                                currentTime - self.lastForceTranscriptionTime) > self.forceTranscriptionCooldown:
                            uniLogger.info("Force transcription hotkey activated.")
                            orchestrator.forceTranscribeCurrentBuffer()
                            self.playNotification("forceTranscribe")
                            self.lastForceTranscriptionTime = currentTime
                        else:
                            uniDebugLogger.debug("Force transcription hotkey in cooldown.")
                        self._waitForKeyRelease(forceTranscriptionKey)
                    time.sleep(0.05)  # Short sleep to prevent high CPU usage
                except Exception as keyCheckError:
                    uniLogger.error(f"Error checking key press: {keyCheckError}. Hotkeys may fail.",
                                    excInfo=True)
                    time.sleep(1)
        except ImportError:
            uniLogger.critical(
                "Keyboard library not installed (`pip install keyboard`). Hotkeys disabled. Stopping.",
                excInfo=True)
            orchestrator.stateManager.stopProgram()
        except Exception as e:
            uniLogger.critical(f"Unhandled exception in keyboard monitoring: {e}", excInfo=True)
            uniLogger.error(
                "Hint: On Linux, ensure user is in 'input' group or run with sudo. On Windows, try Admin.")
            orchestrator.stateManager.stopProgram()
        finally:
            uniLogger.info("Keyboard shortcut monitor thread stopping.")

    def _waitForKeyRelease(self, key):
        """Waits until the specified key is released to prevent rapid, repeated triggers."""
        if not key:
            uniLogger.warning("_waitForKeyRelease called with empty key.")
            return
        try:
            # Use the keyboard library's built-in wait function for efficiency
            keyboard.wait(key, suppress=True, trigger_on_release=True)
            uniDebugLogger.debug(f"Hotkey '{key}' released.")
        except Exception as e:
            uniLogger.warning(f"Error waiting for key release for '{key}': {e}")

    def isModifierKeyPressed(self, key):
        """Checks if a specific modifier key (e.g., 'ctrl', 'alt', 'shift') is currently pressed."""
        try:
            return keyboard.is_pressed(key)
        except Exception as e:
            uniDebugLogger.debug(f"Could not check modifier key '{key}': {e}")
            return False

    def typeText(self, text):
        """Outputs text using the method determined at initialization (typing or clipboard)."""
        if not text:
            uniDebugLogger.debug("typeText called with empty string, skipping output.")
            return

        textToOutput = text

        if self.textOutputMethod == "pyautogui":
            if self._pyautoguiAvailable:
                typingMode = self.config.get('typingMode', 'letter')
                uniDebugLogger.debug(f"Executing PyAutoGUI output with mode: '{typingMode}'")
                try:
                    import pyautogui
                    if typingMode == "letter":
                        pyautogui.write(textToOutput, interval=0.01)
                        pyautogui.write(" ", interval=0.01)
                    elif typingMode == "word":
                        words = textToOutput.split()
                        for word in words:
                            # Abort typing if user disables output or presses CTRL
                            if self.stateManager and not self.stateManager.isOutputEnabled() or self.isModifierKeyPressed(
                                    "ctrl"):
                                uniDebugLogger.debug(
                                    "Output disabled or CTRL pressed during word-by-word typing.")
                                break
                            pyautogui.write(word, interval=0.0)
                            pyautogui.write(' ', interval=0.0)
                            time.sleep(0.05)
                    else:  # "whole" mode
                        pyautogui.write(textToOutput, interval=0)
                        pyautogui.write(" ", interval=0)
                    uniDebugLogger.debug(f"Typed text via PyAutoGUI: '{text[:50]}...'")
                except Exception as e:
                    uniLogger.warning(f"PyAutoGUI write failed (mode: {typingMode}): {e}")
        elif self.textOutputMethod == "clipboard":
            if self.clipExePath:
                uniDebugLogger.debug(f"Executing clipboard output. Text: '{text[:50]}...'")
                try:
                    # Use subprocess to call clip.exe and pipe the text to it
                    subprocess.run(
                        [self.clipExePath],
                        input=textToOutput + " ",
                        encoding='utf-8',
                        check=True,
                        capture_output=True
                    )
                    uniDebugLogger.debug(f"Copied text to Windows clipboard.")
                except Exception as e:
                    # If clipboard fails, disable this output method for the session
                    uniLogger.error(f"Error copying text to clipboard: {e}", excInfo=True)
                    self.textOutputMethod = "none"

    def cleanup(self):
        """Cleans up system interaction resources, specifically the pygame mixer."""
        uniDebugLogger.debug("SystemInteractionHandler cleanup.")
        if self.isMixerInitialized:
            try:
                import pygame
                pygame.mixer.quit()
                uniLogger.info("Pygame mixer quit.")
            except Exception as e:
                uniLogger.error(f"Error quitting pygame mixer: {e}")
