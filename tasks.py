# tasks.py
import string
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from utils import logWarning, logDebug, logInfo, logError


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
