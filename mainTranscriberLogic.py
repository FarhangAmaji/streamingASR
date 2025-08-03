# mainTranscriberLogic.py

# ==============================================================================
# Real-Time Speech-to-Text Transcription Tool - Core Logic
# ==============================================================================
#
# Purpose:
# - Contains the main application classes responsible for configuration, state,
#   audio input/processing, output handling, system interaction (hotkeys, sounds),
#   and model lifecycle management (when running locally).
# - Defines the abstract base class for ASR models.
# - Includes the concrete implementation for local models (e.g., Whisper via Transformers).
# - Includes a client handler class (`RemoteNemoClientHandler`) responsible for
#   communicating with a separate server process (running in WSL) for models
#   that require it (e.g., NeMo models).
#
# Architecture Notes:
# - This file forms the core of the application run on the primary OS (e.g., Windows).
# - It uses composition: the main Orchestrator holds instances of components.
# - ASR models are accessed through the AbstractAsrModelHandler interface,
#   allowing either local processing or remote calls via the client handler.
#
# Dependencies (Ensure installed on the system running this code):
# - Python standard libraries (abc, gc, os, queue, threading, time, pathlib, platform, subprocess, shutil, string)
# - sounddevice: For audio input.
# - soundfile: For audio file operations (used by FileTranscriber).
# - numpy: For numerical audio data manipulation.
# - torch: Required by Transformers and potentially other local models.
# - transformers: For Hugging Face models (like Whisper).
# - huggingface_hub: For listing models.
# - keyboard: For global hotkey monitoring.
# - pygame: For audio notifications.
# - requests: For communicating with the WSL ASR server (used by RemoteNemoClientHandler).
# - pyautogui: (Optional, for Windows native typing) - install if needed.
# ==============================================================================


import abc  # Abstract Base Classes
import gc
import json
import os
import platform  # For OS detection
import queue
import shutil  # For finding clip.exe
import string
import subprocess  # For running clip.exe
import time
import traceback
from pathlib import Path

import huggingface_hub
import keyboard
import numpy as np
import requests  # For client-server communication
# Pygame and PyAutoGUI are imported conditionally later where needed
import sounddevice as sd
import soundfile as sf
import torch
from transformers import pipeline
import logging

# Conditional PyAutoGUI import moved inside SystemInteractionHandler


# ==================================
# Helper Functions & Configuration
# ==================================
def logDebug(message, debugPrintFlag):
    """Helper function for conditional debug printing."""
    if debugPrintFlag:
        print(f"DEBUG: {message}")


def logInfo(message):
    """Helper function for standard info messages."""
    print(f"INFO: {message}")


def logWarning(message):
    """Helper function for warning messages."""
    print(f"WARNING: {message}")


def logError(message):
    """Helper function for error messages."""
    print(f"ERROR: {message}")


# ==================================
# Configuration Management
# ==================================
class ConfigurationManager:
    """Stores and provides access to all application settings."""

    def __init__(self, **kwargs):
        self._config = kwargs
        # --- Derived/Internal Settings ---
        # Ensure scriptDir uses the path of the *running* script (e.g., useRealtimeTranscription.py)
        try:
            # This might be fragile depending on how things are imported/run.
            # A more robust way might be to pass the script path explicitly during init.
            import __main__
            main_file_path = os.path.abspath(__main__.__file__)
            self._config['scriptDir'] = Path(os.path.dirname(main_file_path))
        except (AttributeError, ImportError):
            # Fallback if __main__.__file__ isn't available (e.g., interactive session)
            # This assumes mainTranscriberLogic.py is in the same directory as the sounds
            self._config['scriptDir'] = Path(os.path.dirname(os.path.abspath(__file__)))
            logWarning(
                f"Could not reliably determine main script directory, using fallback: {self._config['scriptDir']}")

        self._config['device'] = None  # Will be set by AsrModelHandler (local or remote info)
        self._config['actualSampleRate'] = self._config.get('sampleRate', 16000)  # Default/Initial
        self._config['actualChannels'] = self._config.get('channels', 1)  # Default/Initial

    def get(self, key, default=None):
        """Gets a configuration value."""
        return self._config.get(key, default)

    def set(self, key, value):
        """Sets or updates a configuration value."""
        self._config[key] = value

    def getAll(self):
        """Returns the entire configuration dictionary."""
        return self._config.copy()


# ==================================
# State Management
# ==================================
class StateManager:
    """Manages the dynamic state of the real-time transcriber."""

    def __init__(self, config):
        self.config = config
        self.isProgramActive = True  # Overall application loop control
        self.isRecordingActive = config.get('isRecordingActive', True)
        self.outputEnabled = config.get('outputEnabled', False)

        # Timing state
        self.programStartTime = time.time()
        self.lastActivityTime = time.time()  # Used for model unloading timeout (local models or server interaction)
        self.recordingStartTime = time.time() if self.isRecordingActive else 0
        self.lastValidTranscriptionTime = time.time()  # Used for consecutive idle timeout

    def _logDebug(self, message):
        logDebug(message, self.config.get('debugPrint'))

    # --- Getters ---
    def isRecording(self):
        return self.isRecordingActive

    def isOutputEnabled(self):
        return self.outputEnabled

    def shouldProgramContinue(self):
        return self.isProgramActive

    # --- Setters ---
    def startRecording(self):
        if not self.isRecordingActive:
            self._logDebug("Setting state to Recording: ON")
            self.isRecordingActive = True
            now = time.time()
            self.recordingStartTime = now
            self.lastActivityTime = now  # Mark activity for model manager/timeout checks
            self.lastValidTranscriptionTime = now  # Reset idle timer
            return True  # State changed
        return False  # No change

    def stopRecording(self):
        if self.isRecordingActive:
            self._logDebug("Setting state to Recording: OFF")
            self.isRecordingActive = False
            self.recordingStartTime = 0  # Reset session start time
            # Mark activity time when stopping recording as well, so server communication timeout resets
            self.lastActivityTime = time.time()
            return True  # State changed
        return False  # No change

    def toggleOutput(self):
        self.outputEnabled = not self.outputEnabled
        status = 'enabled' if self.outputEnabled else 'disabled'
        self._logDebug(f"Setting state Output: {status.upper()}")
        logInfo(f"Output {status}")
        # Mark activity when toggling output, might interact with server
        self.updateLastActivityTime()
        return self.outputEnabled  # Return new state

    def stopProgram(self):
        self._logDebug("Setting state Program Active: OFF")
        self.isProgramActive = False

    def updateLastActivityTime(self):
        """Updates the timestamp of the last significant activity (local processing or server interaction)."""
        self.lastActivityTime = time.time()
        # self._logDebug("Updated last activity time.") # Can be noisy

    def updateLastValidTranscriptionTime(self):
        """Updates the timestamp of the last valid transcription output."""
        self.lastValidTranscriptionTime = time.time()
        self._logDebug("Updated last valid transcription time (idle timer reset).")

    # --- Timeout Checks ---
    def checkRecordingTimeout(self):
        """Checks if the maximum recording session duration has been exceeded, treating 0 or less as no limit."""
        maxDuration = self.config.get('maxDurationRecording', 3600)
        # If maxDuration is 0 or negative, treat it as no limit
        if maxDuration <= 0:
            return False  # <<<--- THIS LINE IS CRITICAL ---

        # --- The rest only runs if maxDuration > 0 ---
        if not self.isRecordingActive or self.recordingStartTime == 0:
            return False

        elapsed = time.time() - self.recordingStartTime
        if elapsed >= maxDuration:
            # This log should now ONLY appear if maxDuration was > 0
            logInfo(f"Maximum recording session duration ({maxDuration}s) reached.")
            return True
        return False

    def checkIdleTimeout(self):
        """Checks if the consecutive idle time limit has been reached, treating 0 or less as no limit."""
        if not self.isRecordingActive:  # Only check if recording is supposed to be active
            return False

        idleTimeout = self.config.get('consecutiveIdleTime', 120)

        # <<<--- Start of Change (Optional but good practice) --->>>
        # If idleTimeout is 0 or negative, treat it as no limit
        if idleTimeout <= 0:
            return False  # Never time out
        # <<<--- End of Change --->>>

        silentFor = time.time() - self.lastValidTranscriptionTime
        if silentFor >= idleTimeout:
            logInfo(f"Consecutive idle time ({idleTimeout}s) reached.")
            # Action (stopping) should be triggered by the orchestrator
            return True
        return False

    def timeSinceLastActivity(self):
        """Calculates the time elapsed since the last recorded activity."""
        return time.time() - self.lastActivityTime

    def checkProgramTimeout(self):
        """Checks if the maximum program duration has been exceeded, treating 0 or less as no limit."""
        maxDuration = self.config.get('maxDurationProgramActive', 3600)
        if maxDuration <= 0:
            return False  # Never time out
        elapsed = time.time() - self.programStartTime
        if elapsed >= maxDuration:
            logInfo(f"Maximum program duration ({maxDuration}s) reached.")
            self.stopProgram()
            return True
        return False


# ==================================
# Audio Handling
# ==================================
class AudioHandler:
    """Manages audio input stream using sounddevice."""

    def __init__(self, config, stateManager):
        self.config = config
        self.stateManager = stateManager
        self.audioQueue = queue.Queue()
        self.stream = None
        # Device info setup might happen before or after ASR handler determines device.
        # Ensure actual rates are used correctly.
        self._setupDeviceInfo()

    def _logDebug(self, message):
        logDebug(message, self.config.get('debugPrint'))

    def _printAudioDevices(self):
        """Helper function to print available audio devices."""
        logInfo("--- Available Audio Devices ---")
        try:
            devices = sd.query_devices()
            print(devices)
        except Exception as e:
            logError(f"Could not query audio devices: {e}")
        logInfo("-----------------------------")

    def _setupDeviceInfo(self):
        """Queries audio device info and sets actual sample rate/channels in config."""
        self._printAudioDevices()
        deviceId = self.config.get('deviceId')
        requestedRate = self.config.get('sampleRate')
        requestedChannels = self.config.get('channels')
        actualRate = requestedRate
        actualChannels = requestedChannels

        self._logDebug(
            f"Setting up device info. Requested device ID: {deviceId}, Rate: {requestedRate}, Channels: {requestedChannels}")
        try:
            deviceInfo = sd.query_devices(deviceId, kind='input')
            self._logDebug(f"Device info: {deviceInfo}")
            if isinstance(deviceInfo, dict):
                actualRate = int(deviceInfo.get("default_samplerate", requestedRate))
                actualChannels = min(requestedChannels,
                                     int(deviceInfo.get("max_input_channels", requestedChannels)))
            else:
                logWarning("Could not retrieve detailed device info, using configured defaults.")
        except Exception as e:
            logWarning(f"Could not query audio device information: {e}. Using configured defaults.")

        if actualChannels < 1:
            logWarning(f"Determined invalid input channels ({actualChannels}), defaulting to 1.")
            actualChannels = 1

        self.config.set('actualSampleRate', actualRate)
        self.config.set('actualChannels', actualChannels)
        logInfo(f"Using Sample Rate: {actualRate} Hz, Channels: {actualChannels}")

    def _audioCallback(self, inData, frames, timeInfo, status):
        """Callback function executed by sounddevice stream thread."""
        if status:
            logWarning(f"Audio callback status: {status}")
        if self.stateManager.isRecording():
            self.audioQueue.put(inData.copy())

    def startStream(self):
        """Starts the sounddevice input stream."""
        if self.stream is not None and self.stream.active:
            self._logDebug("Audio stream already active.")
            return True
        try:
            logInfo("Starting audio stream...")
            self.stream = sd.InputStream(
                samplerate=self.config.get('actualSampleRate'),
                channels=self.config.get('actualChannels'),
                device=self.config.get('deviceId'),
                blocksize=self.config.get('blockSize', 1024),
                callback=self._audioCallback
            )
            self.stream.start()
            logInfo("Audio stream started successfully.")
            return True
        except Exception as e:
            logError(f"Failed to start audio stream: {e}")
            # Provide more specific error guidance if possible
            if "invalid channel count" in str(e).lower():
                logError(
                    "Hint: Check if the configured 'channels' count is supported by your selected audio device.")
            elif "invalid sample rate" in str(e).lower():
                logError(
                    "Hint: Check if the configured 'sampleRate' is supported by your selected audio device.")
            elif "device unavailable" in str(e).lower():
                logError(
                    f"Hint: Check if audio device ID '{self.config.get('deviceId')}' is correct and available.")
            self.stream = None
            return False

    def stopStream(self):
        """Stops the sounddevice input stream."""
        if self.stream is not None and self.stream.active:
            try:
                logInfo("Stopping audio stream...")
                self.stream.stop()
                self.stream.close()
                logInfo("Audio stream stopped.")
            except Exception as e:
                logError(f"Error stopping audio stream: {e}")
            finally:
                self.stream = None
        else:
            self._logDebug("Audio stream already stopped or not initialized.")

    def getAudioChunk(self):
        """Retrieves the next available audio chunk from the queue (non-blocking)."""
        try:
            return self.audioQueue.get_nowait()
        except queue.Empty:
            return None

    def getQueueSize(self):
        """Returns the approximate number of items in the audio queue."""
        return self.audioQueue.qsize()

    def clearQueue(self):
        """Clears all items from the audio queue."""
        while not self.audioQueue.empty():
            try:
                self.audioQueue.get_nowait()
            except queue.Empty:
                break
        self._logDebug("Audio queue cleared.")


# ==================================
# Real-Time Audio Processing Logic
# ==================================
class RealTimeAudioProcessor:
    """
    Handles audio buffer accumulation, processing modes (dictation, constant interval),
    and determines when transcription should be triggered based on audio analysis.
    This component is agnostic to where the transcription happens (local/remote).
    """

    def __init__(self, config, stateManager):
        self.config = config
        self.stateManager = stateManager
        self.audioBuffer = np.array([], dtype=np.float32)
        self.lastTranscriptionTriggerTime = time.time()  # For constant interval mode

        # Dictation mode state
        self.isCurrentlySpeaking = False
        self.silenceStartTime = None

    def _logDebug(self, message):
        logDebug(message, self.config.get('debugPrint'))

    def _calculateChunkLoudness(self, audioChunk):
        """Calculates the average absolute amplitude of an audio chunk."""
        if audioChunk is None or len(audioChunk) == 0:
            return 0.0
        # Ensure float input for abs
        if audioChunk.dtype.kind != 'f':
            audioChunk = audioChunk.astype(np.float32) / np.iinfo(
                audioChunk.dtype).max  # Basic normalization if int
        return np.mean(np.abs(audioChunk))

    def processIncomingChunk(self, audioChunk):
        """Processes a new audio chunk: converts to float32, mono, updates state, appends to buffer."""
        if audioChunk is None:
            return False  # No chunk processed

        # --- Ensure Float32 Audio ---
        if audioChunk.dtype != np.float32:
            # Attempt normalization if integer type
            if audioChunk.dtype.kind in ('i', 'u'):
                max_val = np.iinfo(audioChunk.dtype).max
                min_val = np.iinfo(audioChunk.dtype).min
                # Avoid division by zero for empty range
                if max_val > min_val:
                    audioChunk = (audioChunk.astype(np.float32) - min_val) / (
                            max_val - min_val) * 2.0 - 1.0
                else:
                    audioChunk = audioChunk.astype(np.float32)  # Just cast if range is zero
            else:
                # Direct cast for other float types
                audioChunk = audioChunk.astype(np.float32)

        # --- Ensure Mono Audio ---
        monoChunk = audioChunk.flatten()
        if self.config.get('actualChannels') > 1:
            if len(audioChunk.shape) > 1 and audioChunk.shape[1] > 1:
                monoChunk = np.mean(audioChunk, axis=1)

        # --- Update Dictation Mode State (if applicable) ---
        if self.config.get('transcriptionMode') == "dictationMode":
            self._updateDictationState(monoChunk)

        # --- Append Chunk to Buffer ---
        if monoChunk is not None and len(monoChunk) > 0:
            self.audioBuffer = np.concatenate((self.audioBuffer, monoChunk))
            return True  # Chunk was processed and added
        return False

    def _updateDictationState(self, monoChunk):
        """Updates the speaking flag and silence timer for dictation mode."""
        chunkLoudness = self._calculateChunkLoudness(monoChunk)
        silenceThreshold = self.config.get('dictationMode_silenceLoudnessThreshold', 0.001)

        if chunkLoudness >= silenceThreshold:
            if not self.isCurrentlySpeaking:
                self._logDebug(
                    f"Speech detected (Loudness {chunkLoudness:.6f} >= {silenceThreshold:.6f})")
            self.isCurrentlySpeaking = True
            self.silenceStartTime = None
        else:
            if self.isCurrentlySpeaking and self.silenceStartTime is None:
                self._logDebug(
                    f"Silence detected after speech (Loudness {chunkLoudness:.6f}). Starting silence timer ({self.config.get('dictationMode_silenceDurationToOutput')}s)")
                self.silenceStartTime = time.time()

    def checkTranscriptionTrigger(self):
        """
        Checks if conditions are met to trigger transcription based on the current mode.

        Returns:
            numpy.ndarray or None: The audio data (Transcription Window) to be transcribed,
                                   or None if transcription should not be triggered yet.
        """
        mode = self.config.get('transcriptionMode')
        audioDataToTranscribe = None

        if mode == "constantIntervalMode":
            audioDataToTranscribe = self._checkTriggerConstantInterval()
        elif mode == "dictationMode":
            audioDataToTranscribe = self._checkTriggerDictationMode()

        if audioDataToTranscribe is not None and len(audioDataToTranscribe) > 0:
            self.stateManager.updateLastActivityTime()  # Mark activity when preparing data for ASR
            if mode == "constantIntervalMode":
                self.clearBuffer()
                self.lastTranscriptionTriggerTime = time.time()
            # Dictation mode buffer is cleared *within* _checkTriggerDictationMode on success

            return audioDataToTranscribe
        else:
            # Check and potentially reset constant interval timer even if buffer is empty
            if mode == "constantIntervalMode" and self._isConstantIntervalTimeReached():
                self.lastTranscriptionTriggerTime = time.time()
                # self._logDebug("Constant interval reached, buffer empty. Resetting timer.") # Can be noisy

        return None

    def _isConstantIntervalTimeReached(self):
        """Checks if the time interval for constant mode has passed."""
        interval = self.config.get('constantIntervalMode_transcriptionInterval', 3.0)
        return (time.time() - self.lastTranscriptionTriggerTime) >= interval

    def _checkTriggerConstantInterval(self):
        """Checks trigger conditions for constant interval mode."""
        if self._isConstantIntervalTimeReached() and len(self.audioBuffer) > 0:
            self._logDebug(f"Constant interval trigger. Buffer: {self.getBufferDuration():.2f}s.")
            return self.audioBuffer.copy()
        return None

    def _checkTriggerDictationMode(self):
        """Checks trigger conditions for dictation mode."""
        if self.isCurrentlySpeaking and self.silenceStartTime is not None:
            elapsedSilence = time.time() - self.silenceStartTime
            requiredSilence = self.config.get('dictationMode_silenceDurationToOutput', 0.6)

            if elapsedSilence >= requiredSilence:
                self._logDebug(
                    f"Dictation mode trigger: Silence duration ({elapsedSilence:.2f}s) >= threshold ({requiredSilence}s). Buffer: {self.getBufferDuration():.2f}s.")
                if len(self.audioBuffer) > 0:
                    audioData = self.audioBuffer.copy()
                    # Reset state *after* confirming trigger and getting data
                    self.clearBuffer()  # Clear buffer for next utterance
                    self.isCurrentlySpeaking = False
                    self.silenceStartTime = None
                    self._logDebug("Dictation mode state reset after trigger.")
                    return audioData
                else:
                    self._logDebug(
                        "Dictation mode trigger met, but buffer is empty. Resetting state.")
                    self.isCurrentlySpeaking = False
                    self.silenceStartTime = None
                    return None
        return None

    def getBufferDuration(self):
        """Calculates the duration of the current audio buffer in seconds."""
        if len(self.audioBuffer) == 0:
            return 0.0
        sampleRate = self.config.get('actualSampleRate')
        return len(self.audioBuffer) / sampleRate if sampleRate > 0 else 0.0

    def clearBufferIfOutputDisabled(self):
        """Clears the buffer if output is disabled to prevent backlog."""
        if not self.stateManager.isOutputEnabled():
            if len(self.audioBuffer) > 0:
                self._logDebug("Clearing audio buffer because output is disabled.")
                self.clearBuffer()

    def clearBuffer(self):
        """Clears the internal audio buffer."""
        bufferLen = len(self.audioBuffer)
        if bufferLen > 0:
            self._logDebug(
                f"Clearing audio buffer with {bufferLen} samples ({bufferLen / self.config.get('actualSampleRate', 16000):.2f}s).")
        self.audioBuffer = np.array([], dtype=np.float32)
        # Reset dictation state associated with the buffer being cleared
        # This seems safer here than in the trigger logic sometimes
        # self.isCurrentlySpeaking = False # Careful, might prematurely stop detecting if called mid-speech
        # self.silenceStartTime = None


# ==================================
# Transcription Output Handling
# ==================================
class TranscriptionOutputHandler:
    """
    Handles filtering, formatting, and outputting transcription results received
    either from a local ASR handler or the remote client handler.
    """

    def __init__(self, config, stateManager, systemInteractionHandler):
        self.config = config
        self.stateManager = stateManager
        self.systemInteractionHandler = systemInteractionHandler  # For typing/clipboard output

    def _logDebug(self, message):
        logDebug(message, self.config.get('debugPrint'))

    def _calculateSegmentLoudness(self, audioData):
        """Calculates the average absolute amplitude of the entire segment."""
        if audioData is None or len(audioData) == 0:
            return 0.0
        # Ensure float for calculation
        if audioData.dtype.kind != 'f':
            audioData = audioData.astype(np.float32) / np.iinfo(
                audioData.dtype).max  # Basic normalization
        return np.mean(np.abs(audioData))

    def processTranscriptionResult(self, transcription, audioData):
        """
        Processes the ASR result: checks for silence, filters false positives,
        formats, and triggers output (print/type/clipboard). Updates idle timer.
        Requires audioData for loudness-based filtering.
        """
        if audioData is None or len(audioData) == 0:
            # If audio data is missing, we cannot perform loudness checks.
            # Decide whether to skip or process without loudness filtering.
            # Let's log a warning and attempt processing without loudness checks.
            logWarning("Processing transcription result without audio data for loudness checks.")
            segmentLoudness = -1  # Indicate unavailable loudness
            # Proceed, but filtering based on loudness will be skipped/ineffective.
        else:
            segmentLoudness = self._calculateSegmentLoudness(audioData)
            self._logDebug(
                f"Processing transcription. Segment Avg Loudness = {segmentLoudness:.6f}")

        # --- Initial Checks & Filtering ---
        shouldOutput, finalText = self._filterAndFormatTranscription(transcription, segmentLoudness,
                                                                     audioData)

        # --- Output Actions & State Update ---
        if shouldOutput:
            self._handleValidOutput(finalText)
        else:
            self._handleSilentOrFilteredSegment()

    def _filterAndFormatTranscription(self, transcription, segmentLoudness, audioData):
        """
        Applies filtering rules (silence, false positives) and formatting to the raw transcription.

        Returns:
            tuple[bool, str]: (shouldOutput, formattedText)
        """
        cleanedText = transcription.strip() if isinstance(transcription, str) else ""
        if not cleanedText or cleanedText == ".":
            self._logDebug("Transcription is effectively empty after initial strip.")
            return False, ""

        cleanedText_lower = cleanedText.lower()

        # Silence/Low Content Filtering (only if loudness is available)
        if segmentLoudness != -1:
            if self._shouldSkipTranscriptionDueToSilenceOrLowContent(segmentLoudness, audioData):
                return False, ""
        else:
            self._logDebug("Skipping silence/low content filtering due to missing audio data.")

        # False Positive Filtering (only if loudness is available)
        if segmentLoudness != -1:
            if self._isFalsePositive(cleanedText_lower, segmentLoudness):
                return False, ""
        else:
            self._logDebug("Skipping false positive filtering due to missing audio data.")

        # Final Formatting
        formattedText = cleanedText
        if self.config.get('removeTrailingDots'):
            formattedText = formattedText.rstrip('. ')
        formattedText = formattedText.lstrip(" ")

        if not formattedText:
            self._logDebug("Text became empty after final formatting steps.")
            return False, ""

        self._logDebug(f"Final formatted text ready for output: '{formattedText}'")
        return True, formattedText

    def _shouldSkipTranscriptionDueToSilenceOrLowContent(self, segmentMeanLoudness, audioData):
        """
        Checks if the transcription should be ignored based on combined silence and
        minimum content duration rules. Requires valid audioData.
        Returns True if the segment SHOULD be skipped, False otherwise.
        """
        if audioData is None or len(audioData) == 0:
            # This case should ideally be handled before calling this function if audioData is required.
            logWarning("Silence check called without audio data, cannot perform check.")
            return False  # Don't skip if we can't check

        sampleRate = self.config.get('actualSampleRate')
        if not sampleRate or sampleRate <= 0:
            logWarning("Invalid sample rate for silence check.")
            return False  # Cannot perform check

        chunkSilenceThreshold = self.config.get('dictationMode_silenceLoudnessThreshold', 0.001)
        minLoudDuration = self.config.get('minLoudDurationForTranscription', 0.6)
        silenceSkipThreshold = self.config.get('silenceSkip_threshold', 0.0002)
        checkLeadingSec = self.config.get('skipSilence_beforeNSecSilence', 0.0)
        checkTrailingSec = self.config.get('skipSilence_afterNSecSilence', 0.3)

        # --- 1. Minimum Loud Duration Check ---
        if minLoudDuration > 0:
            loudSamplesMask = np.abs(audioData) >= chunkSilenceThreshold
            numLoudSamples = np.sum(loudSamplesMask)
            totalLoudDuration = numLoudSamples / sampleRate

            if totalLoudDuration < minLoudDuration:
                self._logDebug(
                    f"Silence skip CONFIRMED: Total loud duration ({totalLoudDuration:.2f}s) < min ({minLoudDuration:.2f}s). (Avg Loudness: {segmentMeanLoudness:.6f})")
                return True
            # else:
            #    self._logDebug(f"Passed min loud duration check ({totalLoudDuration:.2f}s >= {minLoudDuration:.2f}s).")

        # --- 2. Average Loudness Check ---
        if segmentMeanLoudness >= silenceSkipThreshold:
            # self._logDebug(f"Segment mean loudness ({segmentMeanLoudness:.6f}) >= skip threshold ({silenceSkipThreshold:.6f}). Not skipping.")
            return False  # DO NOT SKIP

        # --- 3. Low Average Loudness - Check Overrides ---
        # Only if Min Duration Passed but Average Loudness Failed
        self._logDebug(
            f"Segment passed min loud duration but mean loudness ({segmentMeanLoudness:.6f}) < skip threshold ({silenceSkipThreshold:.6f}). Checking overrides...")

        # Check Beginning
        if checkLeadingSec > 0:
            leadingSamples = int(checkLeadingSec * sampleRate)
            if len(audioData) >= leadingSamples:  # Avoid slice errors
                leadingAudio = audioData[:leadingSamples]
                leadingLoudness = np.mean(np.abs(leadingAudio))
                if leadingLoudness >= chunkSilenceThreshold:
                    self._logDebug(
                        f"Silence skip OVERRIDDEN: Leading {checkLeadingSec:.2f}s loud enough ({leadingLoudness:.6f}).")
                    return False  # DO NOT SKIP

        # Check Trailing
        if checkTrailingSec > 0:
            trailingSamples = int(checkTrailingSec * sampleRate)
            if len(audioData) >= trailingSamples:  # Check length for negative index
                trailingAudio = audioData[-trailingSamples:]
                trailingLoudness = np.mean(np.abs(trailingAudio))
                if trailingLoudness >= chunkSilenceThreshold:
                    self._logDebug(
                        f"Silence skip OVERRIDDEN: Trailing {checkTrailingSec:.2f}s loud enough ({trailingLoudness:.6f}).")
                    return False  # DO NOT SKIP

        # --- 4. Final Decision (Low Avg, No Overrides) ---
        self._logDebug(
            f"Silence skip CONFIRMED: Low avg loudness ({segmentMeanLoudness:.6f}) and no start/end overrides triggered.")
        return True  # SKIP

    def _isFalsePositive(self, cleanedText_lower, segmentLoudness):
        """
        Checks if the transcription is a common false word detected in low loudness.
        Requires segmentLoudness.
        """
        commonFalseWords = self.config.get('commonFalseDetectedWords', [])
        if not commonFalseWords:
            return False

        # Further clean the lowercased text for comparison
        translator = str.maketrans('', '', string.punctuation)
        checkText = cleanedText_lower.translate(translator).strip()
        checkText = ' '.join(checkText.split())  # Normalize spaces

        commonFalseWords_normalized = [w.lower() for w in commonFalseWords]

        if checkText in commonFalseWords_normalized:
            loudnessThreshold = self.config.get('loudnessThresholdOf_commonFalseDetectedWords',
                                                0.0008)
            if segmentLoudness < loudnessThreshold:
                self._logDebug(
                    f"'{checkText}' IS false positive (Loudness {segmentLoudness:.6f} < {loudnessThreshold:.6f}). Filtering.")
                return True
            else:
                self._logDebug(
                    f"'{checkText}' matches false positive BUT loudness ({segmentLoudness:.6f}) >= threshold. Not filtering.")

        return False

    def _handleValidOutput(self, finalText):
        """Handles actions for valid, filtered transcription text."""
        print("Transcription:", finalText)  # Always print to console

        # Use system interaction handler for typing/clipboard based on OS/config
        if self.stateManager.isOutputEnabled() and not self.systemInteractionHandler.isModifierKeyPressed(
                "ctrl"):
            self.systemInteractionHandler.typeText(finalText + " ")

        # Reset the idle timer only when valid output is produced
        self.stateManager.updateLastValidTranscriptionTime()

    def _handleSilentOrFilteredSegment(self):
        """Handles actions when transcription is empty, silent, or filtered."""
        # No valid output, let idle timer continue. Logging done in filter methods.
        pass


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


# ==================================
# ASR Model Handling (Abstraction)
# ==================================

class AbstractAsrModelHandler(abc.ABC):
    """
    Abstract Base Class defining the interface for ASR model handlers.
    Implementations handle specific ASR libraries/models (local or remote client).
    """

    def __init__(self, config):
        self.config = config
        self.modelLoaded = False  # Status flag for subclasses
        self._logDebug = lambda msg: logDebug(msg, self.config.get('debugPrint'))

    @abc.abstractmethod
    def loadModel(self):
        """Loads the ASR model into memory (local) or ensures server connection (remote)."""
        pass

    @abc.abstractmethod
    def unloadModel(self):
        """Unloads the ASR model (local) or potentially signals server (remote)."""
        pass

    @abc.abstractmethod
    def transcribeAudioSegment(self, audioData, sampleRate):
        """
        Transcribes a given audio data segment.

        Args:
            audioData (numpy.ndarray): The audio segment (float32, mono expected).
            sampleRate (int): Sample rate of the audio data.

        Returns:
            str: The transcribed text, or an empty string on failure.
        """
        pass

    def isModelLoaded(self):
        """Checks if the model is considered loaded/ready."""
        return self.modelLoaded

    def getDevice(self):
        """Returns the compute device being used ('cuda', 'cpu', 'remote', etc.)."""
        # Subclasses should implement this to report accurately.
        return self.config.get('device', 'unknown')

    def cleanup(self):
        """Default cleanup action is to unload the model if loaded."""
        self._logDebug(f"{type(self).__name__} cleanup initiated.")
        if self.isModelLoaded():
            self.unloadModel()
        self._logDebug(f"{type(self).__name__} cleanup complete.")


# ==================================
# Whisper/Transformers Implementation (Local)
# ==================================
class WhisperModelHandler(AbstractAsrModelHandler):
    """
    Concrete implementation for Whisper models using Hugging Face Transformers.
    Runs the model locally on the machine executing this script.
    """

    def __init__(self, config):
        super().__init__(config)
        self.asrPipeline = None
        self.device = None
        self._determineDevice()
        self.config.set('device', str(self.device))  # Update config with actual device string

    def _determineDevice(self):
        """Determines the compute device (CUDA GPU or CPU) for local execution."""
        if self.config.get('onlyCpu'):
            self.device = torch.device('cpu')
            logInfo("CPU usage forced by 'onlyCpu=True' for Whisper model.")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cuda':
                logInfo("CUDA GPU detected and will be used for Whisper model.")
            else:
                logInfo("CUDA GPU not found or 'onlyCpu=True', using CPU for Whisper model.")

    def _cudaClean(self):
        """Performs garbage collection and attempts to clear PyTorch's CUDA cache."""
        self._logDebug("Cleaning CUDA memory (Whisper Handler)...")
        gc.collect()
        if self.device and self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                # Optional: synchronize and collect ipc handles if issues persist
                # torch.cuda.synchronize()
                # torch.cuda.ipc_collect()
            except Exception as e:
                logWarning(f"CUDA memory cleaning attempt failed partially (Whisper): {e}")
        self._logDebug("CUDA memory cleaning attempt finished (Whisper Handler).")

    def _monitorMemory(self):
        """Monitors and prints current GPU memory usage if debugPrint is enabled."""
        if self.config.get('debugPrint') and self.device and self.device.type == 'cuda':
            try:
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                print(
                    f"GPU Memory (Whisper) - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            except Exception as e:
                logWarning(f"Failed to get GPU memory stats: {e}")

    def loadModel(self):
        """Loads the Whisper ASR model pipeline locally."""
        if self.modelLoaded:
            self._logDebug(f"Whisper model '{self.config.get('modelName')}' already loaded.")
            return

        modelName = self.config.get('modelName')
        if not modelName:
            logError("Cannot load Whisper model: 'modelName' not specified in config.")
            return

        self._logDebug(f"Loading Whisper model '{modelName}' locally to {self.device}...")
        self._monitorMemory()
        self._cudaClean()  # Clean before loading

        # Prepare generation arguments
        genKwargs = {"language": self.config.get('language')}
        # Consider adding task (transcribe/translate) if needed, based on config
        # genKwargs["task"] = self.config.get('whisperTask', 'transcribe')
        # Timestamps can be useful for post-processing or word timings if required
        genKwargs["return_timestamps"] = self.config.get('whisperReturnTimestamps',
                                                         False)  # Default False unless needed
        self._logDebug(f"Pipeline generate_kwargs: {genKwargs}")

        try:
            self.asrPipeline = pipeline(
                "automatic-speech-recognition",
                model=modelName,
                generate_kwargs=genKwargs,
                device=self.device,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
                # Use float16 on GPU if available
            )
            self.modelLoaded = True
            logInfo(f"Whisper model '{modelName}' loaded successfully locally on {self.device}.")
            self._warmUpModel()  # Warm up after loading

        except Exception as e:
            logError(f"Failed loading local Whisper model '{modelName}': {e}")
            logError(
                "Check model name, internet connection (for download), dependencies, and memory.")
            # More specific errors
            if "requires the PyTorch library" in str(e):
                logError("Hint: Ensure PyTorch is installed correctly (`pip install torch`).")
            elif "safetensors_rust" in str(e):
                logError("Hint: Ensure `safetensors` is installed (`pip install safetensors`).")
            self.modelLoaded = False
            self.asrPipeline = None
            self._cudaClean()  # Clean up potential partial load

        self._monitorMemory()  # Monitor after load attempt

    def _warmUpModel(self):
        """Warms up the loaded model with a silent clip to reduce first inference latency."""
        if not self.modelLoaded or not self.asrPipeline:
            return
        try:
            self._logDebug("Warming up the Whisper model...")
            # Whisper expects 16kHz sample rate
            warmupSampleRate = 16000
            dummyAudio = np.zeros(warmupSampleRate, dtype=np.float32)  # 1 second silence
            # Prepare input in the format the pipeline expects
            asrInput = {"raw": dummyAudio, "sampling_rate": warmupSampleRate}
            # Execute inference
            _ = self.asrPipeline(asrInput)
            self._logDebug("Whisper model warm-up complete.")
        except Exception as e:
            logWarning(f"Whisper model warm-up failed: {e}")

    def unloadModel(self):
        """Unloads the local Whisper ASR model and cleans GPU cache."""
        if not self.modelLoaded:
            self._logDebug("Whisper model already unloaded.")
            return

        modelName = self.config.get('modelName')
        self._logDebug(f"Unloading Whisper model '{modelName}' from {self.device}...")
        if self.asrPipeline is not None:
            # Explicitly delete the pipeline object
            try:
                del self.asrPipeline.model  # Try deleting inner model first if possible
            except AttributeError:
                pass  # Ignore if structure is different
            del self.asrPipeline
            self.asrPipeline = None

        self.modelLoaded = False
        self._cudaClean()  # Clean memory *after* deleting reference
        logInfo(f"Whisper model '{modelName}' unloaded.")
        self._monitorMemory()

    def transcribeAudioSegment(self, audioData, sampleRate):
        """Transcribes audio using the loaded local Whisper pipeline."""
        if not self.modelLoaded or self.asrPipeline is None:
            self._logDebug("Whisper transcription skipped: Model not loaded.")
            return ""
        if audioData is None or len(audioData) == 0:
            self._logDebug("Whisper transcription skipped: No audio data provided.")
            return ""

        # Pre-process: Ensure float32 (already done by RealTimeAudioProcessor usually)
        #             Ensure mono (already done by RealTimeAudioProcessor usually)
        #             Check sample rate (Whisper expects 16kHz, but pipeline might handle resampling)
        if sampleRate != 16000:
            logWarning(
                f"Whisper model prefers 16kHz, received {sampleRate}Hz. Pipeline *should* handle resampling.")
            # If issues occur, manual resampling might be needed here using librosa or similar.

        transcription = ""
        try:
            segmentDurationSec = len(audioData) / sampleRate if sampleRate > 0 else 0
            self._logDebug(
                f"Sending {segmentDurationSec:.2f}s audio segment to local Whisper pipeline...")

            # Prepare input dictionary
            asrInput = {"raw": audioData, "sampling_rate": sampleRate}

            # Perform transcription
            result = self.asrPipeline(asrInput)
            # self._logDebug(f"Whisper Raw Result: {result}") # Can be verbose

            # Extract text - structure might vary slightly based on args (e.g., with timestamps)
            if isinstance(result, dict) and "text" in result:
                transcription = result["text"].strip()
            elif isinstance(result, str):  # Sometimes pipeline might return just the string
                transcription = result.strip()
            else:
                logWarning(
                    f"Unexpected Whisper result structure: {type(result)}. Could not extract text.")
                transcription = ""

        except Exception as e:
            logError(f"Error during local Whisper transcription: {e}")
            import traceback
            logError(traceback.format_exc())  # Log full traceback for easier debugging
            transcription = ""
            # Attempt to clean GPU memory if a CUDA error occurred
            if self.device.type == 'cuda' and 'cuda' in str(e).lower():
                self._cudaClean()

        return transcription  # Already stripped in extraction logic

    def getDevice(self):
        """Returns the compute device being used (e.g., 'cuda', 'cpu')."""
        return str(self.device) if self.device else 'unknown'

    @staticmethod
    def listAvailableModels():
        """Static method to retrieve Whisper/ASR models from Hugging Face Hub."""
        try:
            logInfo("Fetching list of available ASR models from Hugging Face Hub...")
            models = huggingface_hub.list_models(filter="automatic-speech-recognition",
                                                 sort="downloads", direction=-1)
            modelIds = [model.id for model in models]
            logInfo(f"Found {len(modelIds)} ASR models on Hub.")
            return modelIds
        except Exception as e:
            logError(f"Could not fetch models from Hugging Face Hub: {e}")
            return []


# ==================================
# Remote NeMo Client Implementation
# ==================================
class RemoteNemoClientHandler(AbstractAsrModelHandler):
    """
    Concrete implementation that acts as a client to a remote ASR server (WSL).
    It sends audio data via HTTP requests and receives transcription results.
    It does *not* load the NeMo model itself; the server does.
    """

    def __init__(self, config):
        super().__init__(config)
        self.serverUrl = config.get('wslServerUrl')
        if not self.serverUrl:
            logError(
                "RemoteNemoClientHandler cannot operate: 'wslServerUrl' not found in configuration.")
            # We cannot function without the URL, but don't raise error here,
            # let methods fail gracefully if called.
            self.serverReachable = False
        else:
            self.serverReachable = None  # Unknown until first request
            logInfo(f"RemoteNemoClientHandler initialized. Target server URL: {self.serverUrl}")
        # The 'modelLoaded' status here reflects reachability/status of the *server's* model
        self.modelLoaded = False  # Assume not loaded initially
        self.config.set('device', 'remote_wsl')  # Indicate execution happens remotely

    def _makeServerRequest(self, method, endpoint, **kwargs):
        """Helper function to make requests to the WSL server."""
        if not self.serverUrl:
            logError("Cannot make server request: Server URL not configured.")
            return None

        url = f"{self.serverUrl.rstrip('/')}/{endpoint.lstrip('/')}"
        self._logDebug(f"Sending {method.upper()} request to server: {url}")
        timeoutSeconds = self.config.get('serverRequestTimeout', 10.0)  # Configurable timeout

        try:
            response = requests.request(method, url, timeout=timeoutSeconds, **kwargs)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            # Mark server as reachable on successful communication
            if self.serverReachable is not True:
                logInfo(f"Successfully connected to WSL server at {self.serverUrl}.")
                self.serverReachable = True

            try:
                return response.json()  # Assume server sends JSON
            except json.JSONDecodeError:
                logError(f"Server response from {url} is not valid JSON: {response.text[:100]}...")
                return None  # Treat non-JSON response as an error

        except requests.exceptions.ConnectionError as e:
            logError(f"Connection Error connecting to WSL server at {url}: {e}")
            if self.serverReachable is not False:
                logError(
                    "Hint: Ensure the wslNemoServer.py script is running in WSL and accessible.")
            self.serverReachable = False
            self.modelLoaded = False  # Assume model unloaded if server unreachable
            return None
        except requests.exceptions.Timeout:
            logError(f"Request Timeout connecting to WSL server at {url} (>{timeoutSeconds}s).")
            self.serverReachable = False  # Potentially reachable but slow
            self.modelLoaded = False
            return None
        except requests.exceptions.RequestException as e:
            logError(f"Error during request to WSL server at {url}: {e}")
            # Log response body if available and useful
            if hasattr(e, 'response') and e.response is not None:
                logError(f"Server Response ({e.response.status_code}): {e.response.text[:200]}...")
            self.serverReachable = False  # General request error
            self.modelLoaded = False
            return None

    def loadModel(self):
        """Attempts to tell the remote server to load the model (if not already loaded)."""
        self._logDebug("Checking remote server model status...")
        # Check status first
        statusResponse = self._makeServerRequest('get', '/status')
        if statusResponse and statusResponse.get('status') == 'loaded':
            self._logDebug("Remote NeMo model is already loaded on server.")
            self.modelLoaded = True
            return

        # If not loaded or status check failed, attempt to trigger load
        self._logDebug("Requesting remote server to load NeMo model...")
        loadResponse = self._makeServerRequest('post', '/load')  # Assuming POST triggers load

        if loadResponse and loadResponse.get('status') == 'loaded':
            logInfo(
                f"Successfully requested remote server to load model '{loadResponse.get('modelName', 'N/A')}'.")
            self.modelLoaded = True
        elif loadResponse:
            logError(
                f"Remote server reported an issue during load: {loadResponse.get('message', 'Unknown error')}")
            self.modelLoaded = False
        else:
            # Error already logged by _makeServerRequest
            logError("Failed to trigger model load on remote server (communication error).")
            self.modelLoaded = False

    def unloadModel(self):
        """Attempts to tell the remote server to unload the model."""
        self._logDebug("Requesting remote server to unload NeMo model...")
        unloadResponse = self._makeServerRequest('post', '/unload')  # Assuming POST triggers unload

        if unloadResponse and unloadResponse.get('status') == 'unloaded':
            logInfo("Successfully requested remote server to unload model.")
            self.modelLoaded = False
        elif unloadResponse:
            logWarning(
                f"Remote server reported an issue during unload: {unloadResponse.get('message', 'Unknown error')}")
            # State might be ambiguous, assume unloaded for safety
            self.modelLoaded = False
        else:
            # Error already logged by _makeServerRequest
            logWarning(
                "Failed to trigger model unload on remote server (communication error). State unknown.")
            # Assume unloaded for safety
            self.modelLoaded = False

    def transcribeAudioSegment(self, audioData, sampleRate):
        """Sends audio data to the remote server for transcription."""
        if not self.serverReachable and self.serverReachable is not None:  # Check reachability if known
            self._logDebug("Remote transcription skipped: Server known to be unreachable.")
            return ""
        if audioData is None or len(audioData) == 0:
            self._logDebug("Remote transcription skipped: No audio data provided.")
            return ""

        segmentDurationSec = len(audioData) / sampleRate if sampleRate > 0 else 0
        self._logDebug(
            f"Sending {segmentDurationSec:.2f}s audio segment to remote NeMo server for transcription...")

        # Ensure audioData is float32 bytes
        if audioData.dtype != np.float32:
            audioData = audioData.astype(np.float32)
        audioBytes = audioData.tobytes()

        # Prepare request data and parameters
        files = {'audio_data': ('audio.bin', audioBytes, 'application/octet-stream')}
        params = {'sample_rate': sampleRate}

        # Make the request
        transcribeResponse = self._makeServerRequest('post', '/transcribe', params=params,
                                                     files=files)

        if transcribeResponse and 'transcription' in transcribeResponse:
            transcription = transcribeResponse['transcription']
            # Update model status based on successful transcription
            if not self.modelLoaded:
                self._logDebug("Received transcription, marking remote model as loaded.")
                self.modelLoaded = True
            self._logDebug(f"Received transcription from server: '{transcription[:100]}...'")
            return transcription.strip()  # Server should ideally strip, but ensure here
        else:
            logError("Failed to get valid transcription from remote server.")
            # Assume model might have unloaded or server failed
            self.modelLoaded = False
            return ""

    def isModelLoaded(self):
        """Returns the last known status of the remote model."""
        # Note: This might be slightly out of sync. A periodic status check could improve accuracy.
        # For simplicity, we rely on the status updated during operations.
        return self.modelLoaded

    def getDevice(self):
        """Returns 'remote_wsl' to indicate where processing occurs."""
        return 'remote_wsl'

    def cleanup(self):
        """Optionally trigger unload on the server during cleanup."""
        self._logDebug("RemoteNemoClientHandler cleanup initiated.")
        # Decide whether to unload the server model on client exit.
        # If the server is persistent, maybe don't unload. If it's tied to the client, unload.
        shouldUnloadOnExit = self.config.get('unloadRemoteModelOnExit', True)
        if shouldUnloadOnExit and self.modelLoaded:
            self.unloadModel()
        self._logDebug("RemoteNemoClientHandler cleanup complete.")


# ==================================
# Model Lifecycle Management
# ==================================
class ModelLifecycleManager:
    """
    Handles automatic loading/unloading of the ASR model based on activity.
    Works for both local handlers (Whisper) and the remote client handler (NeMo).
    For remote, 'load'/'unload' interact with the server status/endpoints.
    """

    def __init__(self, config, stateManager, asrModelHandler, systemInteractionHandler):
        self.config = config
        self.stateManager = stateManager
        self.asrModelHandler = asrModelHandler  # Can be WhisperModelHandler or RemoteNemoClientHandler
        self.systemInteractionHandler = systemInteractionHandler
        self._logDebug = lambda msg: logDebug(msg, self.config.get('debugPrint'))

    def manageModelLifecycle(self):
        """
        Runs in a thread to monitor activity and load/unload the model (local or remote).
        """
        handlerType = type(self.asrModelHandler).__name__
        logInfo(f"Starting Model Lifecycle Manager thread (Handler: {handlerType}).")
        checkInterval = 10  # Seconds to wait between checks

        while self.stateManager.shouldProgramContinue():
            isRecording = self.stateManager.isRecording()
            # Use the handler's method to check loaded status (works for local/remote)
            modelIsCurrentlyLoaded = self.asrModelHandler.isModelLoaded()
            unloadTimeout = self.config.get('model_unloadTimeout', 1200)  # In seconds

            # --- Unload Condition ---
            # Unload if NOT recording AND model IS loaded AND timeout exceeded
            if not isRecording and modelIsCurrentlyLoaded and unloadTimeout > 0:
                timeInactive = self.stateManager.timeSinceLastActivity()
                if timeInactive >= unloadTimeout:
                    self._logDebug(
                        f"Model inactive for {timeInactive:.1f}s (>= {unloadTimeout}s), requesting unload...")
                    try:
                        self.asrModelHandler.unloadModel()  # Request unload (local or remote)
                        # Only play sound if unload seemed successful (modelLoaded becomes False)
                        if not self.asrModelHandler.isModelLoaded():
                            self.systemInteractionHandler.playNotification("modelUnloaded")
                        else:
                            logWarning(
                                "Unload requested, but handler still reports model as loaded.")
                    except Exception as e:
                        logError(f"Error during model unload request: {e}")
                # else:
                #     if self.config.get('debugPrint'):
                #          print(f"DEBUG: Model loaded but inactive. Time since last activity: {timeInactive:.1f}s / {unloadTimeout}s")

            # --- Load Condition ---
            # Load if recording IS active AND model IS NOT loaded
            elif isRecording and not modelIsCurrentlyLoaded:
                logInfo("Recording active but model not loaded. Triggering model load...")
                try:
                    self.asrModelHandler.loadModel()  # Request load (local or remote)
                    # If loading fails, modelLoaded will remain false, loop will retry later.
                except Exception as e:
                    logError(f"Error during model load request: {e}")
                # Update activity time after a load attempt (success or fail) to reset unload timer
                self.stateManager.updateLastActivityTime()

            # --- Periodic Check ---
            # Use a timed sleep that can be interrupted if the program stops
            startTime = time.time()
            while (
                    time.time() - startTime < checkInterval) and self.stateManager.shouldProgramContinue():
                time.sleep(0.5)  # Sleep in smaller chunks

        logInfo("Model Lifecycle Manager thread stopping.")


# ==================================
# File Transcriber Class
# ==================================
class FileTranscriber:
    """
    Handles transcription of pre-recorded audio files using a provided ASR handler.
    Works with both local handlers (Whisper) and the remote client handler (NeMo).
    """

    def __init__(self, config, asrModelHandler):
        """
        Initialize the file transcriber.

        Args:
             config (ConfigurationManager): Application configuration object.
             asrModelHandler (AbstractAsrModelHandler): The ASR model handler to use (local or remote client).
        """
        self.config = config
        self.asrModelHandler = asrModelHandler
        self._logDebug = lambda msg: logDebug(msg, self.config.get('debugPrint'))

    def transcribeFile(self, audioFilePath, outputFilePath=None):
        """
        Transcribe an audio file and optionally save the transcription.

        Args:
            audioFilePath (str | Path): Path to the input audio file.
            outputFilePath (str | Path, optional): Path to save the transcription text file.
                                                   If None, prints to console. Defaults to None.

        Returns:
            str | None: Transcribed text, or None if transcription fails.
        """
        self._logDebug(f"Attempting to transcribe file: {audioFilePath}")
        transcription = None
        audioFilePath = Path(audioFilePath)

        handlerType = type(self.asrModelHandler).__name__
        self._logDebug(f"Using ASR Handler: {handlerType}")

        try:
            # --- Ensure Model is Ready ---
            # Ask the handler to ensure the model is loaded/ready.
            # For local: loads the model. For remote: checks server status, maybe triggers load.
            if not self.asrModelHandler.isModelLoaded():
                logInfo(
                    f"ASR model ({handlerType}) not ready for file transcription, attempting to load/prepare...")
                self.asrModelHandler.loadModel()  # Trigger load/check

            # Check again after attempting load
            if not self.asrModelHandler.isModelLoaded():
                logError(
                    f"Model ({handlerType}) could not be loaded/prepared. File transcription aborted.")
                return None

            # --- Read Audio File ---
            if not audioFilePath.is_file():
                raise FileNotFoundError(f"Audio file not found at {audioFilePath}")

            # Read audio data ensuring float32 format
            try:
                audioData, sampleRate = sf.read(audioFilePath, dtype='float32', always_2d=False)
            except Exception as e:
                logError(f"Error reading audio file {audioFilePath} using soundfile: {e}")
                logError(
                    "Hint: Ensure the file is a valid audio format (WAV, FLAC, MP3 with libraries, etc.) and not corrupted.")
                return None

            fileDuration = len(audioData) / sampleRate if sampleRate > 0 else 0
            logInfo(
                f"Audio file read successfully: {audioFilePath.name} (Sample Rate: {sampleRate}, Duration: {fileDuration:.2f}s)")

            # --- Perform Transcription ---
            # Call the handler's transcribe method - it handles local or remote execution.
            logInfo("Starting transcription...")
            startTime = time.time()
            transcription = self.asrModelHandler.transcribeAudioSegment(audioData, sampleRate)
            elapsedTime = time.time() - startTime
            logInfo(f"Transcription finished in {elapsedTime:.2f} seconds.")

            # --- Post-Processing & Output ---
            if transcription is not None and isinstance(transcription, str):
                # Basic cleanup (handler might already do some)
                transcription = transcription.strip()
                if self.config.get('removeTrailingDots'):
                    transcription = transcription.rstrip('. ')

                if transcription:  # Check if not empty after stripping
                    self._handleOutput(transcription, outputFilePath)
                else:
                    logInfo("Transcription result was empty after processing.")
                    transcription = ""  # Ensure consistent return type
            else:
                logWarning("Transcription failed or returned unexpected type.")
                transcription = None  # Indicate failure

            return transcription  # Return the final text or None

        except FileNotFoundError as e:
            logError(str(e))
            return None
        except Exception as e:
            logError(f"Unexpected error during file transcription '{audioFilePath}': {e}")
            import traceback
            logError(traceback.format_exc())
            return None

    def _handleOutput(self, transcription, outputFilePath):
        """Saves transcription to file or prints to console."""
        if outputFilePath:
            try:
                outputFilePath = Path(outputFilePath)
                outputFilePath.parent.mkdir(parents=True, exist_ok=True)
                with open(outputFilePath, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                logInfo(f"Transcription saved to: {outputFilePath}")
            except IOError as e:
                logError(f"Error writing transcription to file {outputFilePath}: {e}")
                print("\n--- Transcription Fallback Output ---")
                print(transcription)
                print("-------------------------------------\n")
        else:
            print("\n--- Transcription ---")
            print(transcription)
            print("---------------------\n")

    def cleanup(self):
        """Optional cleanup for file transcriber (usually handled by orchestrator or handler)."""
        # FileTranscriber itself holds little state. Cleanup is typically managed
        # by the entity that created the ASR handler instance used here.
        logInfo("FileTranscriber cleanup.")
        # If the ASR handler was created *only* for this transcriber instance,
        # it should be cleaned up here. Otherwise, leave it to the caller.
        # Example (if handler is exclusive):
        # if hasattr(self, 'asrModelHandler') and self.asrModelHandler:
        #     self.asrModelHandler.cleanup()
