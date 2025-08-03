# systemInteractions.py
import os
import platform
import shutil  # For finding clip.exe
import subprocess
import time
import traceback

import keyboard

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
        self._pyautoguiAvailable = False  # Local state for this instance
        self._pyautoguiErrorMessage = ""

        # Setup attempts moved here
        self._setupPygame()
        self._setupPyautogui()  # Attempt to setup pyautogui
        self._setupAudioNotifications()  # Load sounds if mixer init succeeded

        self.textOutputMethod = "none"
        self.clipExePath = None
        self.isWslEnvironment = False

        self._determineTextOutputMethod()

    def _logDebug(self, message):
        logDebug(message, self.config.get('debugPrint'))

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
                # Test basic functionality that might fail without display
                pyautogui.size()  # Example check
                self._pyautoguiAvailable = True
                logInfo("PyAutoGUI loaded successfully (for potential Windows native typing).")
            except ImportError:
                self._pyautoguiErrorMessage = "PyAutoGUI library not found. Install it (`pip install pyautogui`) to enable typing output on Windows."
                logWarning(self._pyautoguiErrorMessage)
                self._pyautoguiAvailable = False
            except Exception as e:
                # Catch display-related or other init errors
                self._pyautoguiErrorMessage = f"PyAutoGUI could not initialize on Windows (maybe no display?): {e}. Typing output will be disabled."
                logWarning(self._pyautoguiErrorMessage)
                self._pyautoguiAvailable = False
        else:
            self._pyautoguiAvailable = False  # Not expected/needed on non-Windows

    def _setupAudioNotifications(self):
        """Loads sound file paths if mixer is initialized."""
        if not self.isMixerInitialized:
            return  # Skip if mixer failed

        soundMap = {
            "modelUnloaded": "modelUnloaded.mp3",
            "outputDisabled": "outputDisabled.mp3",
            "outputEnabled": "outputEnabled.mp3",
            "recordingOff": "recordingOff.mp3",
            "recordingOn": "recordingOn.mp3"
        }
        scriptDir = self.config.get('scriptDir')  # Get from config
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

        if osName == "Linux" and "WSL_DISTRO_NAME" in os.environ:
            self.isWslEnvironment = True
            logInfo("WSL environment detected.")
        elif osName == "Windows":
            logInfo("Windows Native environment detected.")
        else:
            logInfo(f"Non-Windows/Non-WSL environment detected ({osName}).")

        if outputEnabledByConfig:
            if osName == "Windows" and not self.isWslEnvironment:
                if self._pyautoguiAvailable:  # Check instance variable
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

    def playNotification(self, soundName):
        """Plays a notification sound if available and enabled."""
        if not self.config.get('enableAudioNotifications', True):
            # self._logDebug(f"Skipping sound '{soundName}' - notifications disabled.")
            return

        if soundName in ['recordingOn', 'outputEnabled'] and not self.config.get('playEnableSounds',
                                                                                 False):
            # self._logDebug(f"Skipping enable sound '{soundName}'.")
            return

        if not self.isMixerInitialized or soundName not in self.audioFiles:
            # self._logDebug(f"Cannot play sound '{soundName}'. Mixer: {self.isMixerInitialized}, Sound exists: {soundName in self.audioFiles}")
            return

        import pygame  # Known to be available if mixer initialized
        soundPath = self.audioFiles[soundName]
        try:
            sound = pygame.mixer.Sound(soundPath)
            sound.play()
            self._logDebug(f"Played notification sound: {soundName}")
        except Exception as e:
            logError(f"Error playing notification sound '{soundPath}': {e}")

    def monitorKeyboardShortcuts(self, orchestrator):
        """
        Runs in a thread to monitor global hotkeys. Calls methods on the orchestrator.
        Stops when orchestrator's state indicates program should stop or max duration is reached (if set).
        Logs specific errors encountered.
        """
        logInfo("Starting keyboard shortcut monitor thread.")
        threadStartTime = time.time()
        maxDuration = self.config.get('maxDurationProgramActive', 3600)
        recordingToggleKey = self.config.get('recordingToggleKey')
        outputToggleKey = self.config.get('outputToggleKey')
        checkDuration = maxDuration > 0
        exitReason = "state change"  # Default assumption

        try:
            # Initial check to see if keyboard library is functional here
            _ = keyboard.is_pressed('shift')  # Test a common key
            logInfo("Keyboard library access seems functional.")

            while orchestrator.stateManager.shouldProgramContinue():
                if checkDuration and (time.time() - threadStartTime) >= maxDuration:
                    logInfo("Keyboard monitor thread exiting due to program max duration.")
                    exitReason = "program duration"
                    orchestrator.stateManager.stopProgram()
                    break

                # === Check Hotkeys ===
                # Wrap is_pressed in try-except within the loop for robustness
                try:
                    if keyboard.is_pressed(recordingToggleKey):
                        self._logDebug(f"Hotkey '{recordingToggleKey}' pressed.")
                        orchestrator.toggleRecording()
                        self._waitForKeyRelease(recordingToggleKey)

                    if keyboard.is_pressed(outputToggleKey):
                        self._logDebug(f"Hotkey '{outputToggleKey}' pressed.")
                        orchestrator.toggleOutput()
                        self._waitForKeyRelease(outputToggleKey)

                except Exception as keyCheckError:
                    # This might catch permission errors happening *during* the loop
                    logError(
                        f"Error checking key press: {keyCheckError}. Hotkeys may stop working.")
                    # Depending on the error, might need to break or just continue
                    # For now, log and continue, but if it persists, break might be better
                    time.sleep(1)  # Avoid spamming logs if error repeats quickly

                time.sleep(0.05)  # Prevent high CPU

        except ImportError:
            logError("Keyboard library not installed. Hotkeys disabled.")
            exitReason = "ImportError"
            orchestrator.stateManager.stopProgram()
        except Exception as e:
            # Catch permission errors or others during initial check or loop setup
            logError(f"Unhandled exception in keyboard monitoring setup/loop: {e}")
            logError(traceback.format_exc())
            exitReason = f"Unhandled Exception: {e}"
            orchestrator.stateManager.stopProgram()
        finally:
            logInfo(f"Keyboard shortcut monitor thread stopping (Reason: {exitReason}).")
            # Ensure program stops if thread exits for any reason
            orchestrator.stateManager.stopProgram()

    def _waitForKeyRelease(self, key):
        """Waits until the specified key is released to prevent rapid toggling."""
        startTime = time.time()
        timeout = 2.0  # seconds
        try:
            while keyboard.is_pressed(key):
                if time.time() - startTime > timeout:
                    self._logDebug(f"Timeout waiting for key release '{key}'.")
                    break
                time.sleep(0.05)
            self._logDebug(f"Hotkey '{key}' released.")
        except Exception as e:
            logWarning(f"Error checking key release for '{key}': {e}")

    def isModifierKeyPressed(self, key):
        """Checks if a specific modifier key (e.g., 'ctrl', 'alt', 'shift') is pressed."""
        try:
            return keyboard.is_pressed(key)
        except Exception as e:
            self._logDebug(f"Could not check modifier key '{key}': {e}")
            return False

    def typeText(self, text):
        """
        Outputs text using the method determined during initialization
        (PyAutoGUI typing on Windows native, clipboard copy on WSL).
        Assumes the check for outputEnabled happened before calling this.
        """
        if self.textOutputMethod == "pyautogui":
            if self._pyautoguiAvailable:
                try:
                    import pyautogui  # Import locally
                    pyautogui.write(text, interval=0.01)  # Small interval can help reliability
                    self._logDebug(f"Typed text via PyAutoGUI: '{text[:50]}...'")
                except Exception as e:
                    logWarning(f"PyAutoGUI write failed during execution: {e}")
                    # Could disable it for future calls if needed: self._pyautoguiAvailable = False
            else:
                self._logDebug(
                    "Typing skipped: PyAutoGUI method selected but unavailable/failed init.")

        elif self.textOutputMethod == "clipboard":
            if self.clipExePath:
                try:
                    process = subprocess.run(
                        [self.clipExePath],
                        input=text,
                        encoding='utf-8',
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    self._logDebug(f"Copied text to Windows clipboard: '{text[:50]}...'")
                except FileNotFoundError:
                    logError(f"Error copying to clipboard: '{self.clipExePath}' not found.")
                    self.clipExePath = None  # Mark unavailable
                    self.textOutputMethod = "none"
                except subprocess.CalledProcessError as e:
                    logError(f"Error running clip.exe: {e}")
                    logError(f"clip.exe stderr: {e.stderr.decode('utf-8', errors='ignore')}")
                except Exception as e:
                    logError(f"Unexpected error copying text to clipboard: {e}")
            else:
                self._logDebug("Clipboard copy skipped: Method selected but clip.exe unavailable.")

        # No action needed for self.textOutputMethod == "none"

    def cleanup(self):
        """Cleans up system interaction resources (pygame mixer)."""
        logDebug("SystemInteractionHandler cleanup.", self.config.get('debugPrint'))
        if self.isMixerInitialized:
            try:
                import pygame
                pygame.mixer.quit()
                logInfo("Pygame mixer quit.")
            except Exception as e:
                logError(f"Error quitting pygame mixer: {e}")
