# tasks.py
# ==============================================================================
# Transcription Processing and File Handling Tasks
# ==============================================================================
#
# Purpose:
# - TranscriptionOutputHandler: Handles the post-processing of ASR results.
#   It filters transcriptions for silence or common false positives, formats the
#   text, and then passes it to the SystemInteractionHandler for output.
# - FileTranscriber: Provides functionality to transcribe a pre-recorded audio
#   file using a given ASR model handler.
# ==============================================================================
import re
import string
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
# Import the new universal logger instances
from utils.loggerInstance import uniLogger, uniDebugLogger


# ==================================
# Transcription Output Handling
# ==================================
class TranscriptionOutputHandler:
    """
    Handles filtering, formatting, and outputting transcription results received
    from an ASR handler.
    """

    def __init__(self, config, stateManager, systemInteractionHandler):
        """Initializes the handler with necessary application components."""
        self.config = config
        self.stateManager = stateManager
        self.systemInteractionHandler = systemInteractionHandler

    def _calculateSegmentLoudness(self, audioData):
        """Calculates the average absolute amplitude (a proxy for loudness) of an audio segment."""
        if audioData is None or len(audioData) == 0:
            return 0.0
        # Ensure audio data is in float format for accurate calculation
        if audioData.dtype.kind != 'f':
            if audioData.dtype.kind in ('i', 'u'):
                # Normalize integer audio data to a float range of [-1.0, 1.0]
                maxVal = np.iinfo(audioData.dtype).max
                if maxVal > 0:
                    audioData = audioData.astype(np.float32) / maxVal
            else:
                # Convert other non-float types
                audioData = audioData.astype(np.float32)
        return np.mean(np.abs(audioData))

    def processTranscriptionResult(self, transcription, audioData):
        """
        Processes the ASR result: checks for silence, filters false positives,
        formats, and triggers output (print/type/clipboard). Updates idle timer.
        """
        if audioData is None or len(audioData) == 0:
            uniLogger.warning("Processing transcription without audio data for loudness checks.")
            segmentLoudness = -1  # Indicate that loudness is unavailable
        else:
            segmentLoudness = self._calculateSegmentLoudness(audioData)
            uniDebugLogger.debug(
                f"Processing transcription. Segment Avg Loudness = {segmentLoudness:.6f}")

        # Apply filtering and formatting rules
        shouldOutput, finalText = self._filterAndFormatTranscription(transcription, segmentLoudness,
                                                                     audioData)

        # Handle the result based on whether it should be outputted
        if shouldOutput:
            self._handleValidOutput(finalText)
        else:
            self._handleSilentOrFilteredSegment()

    def _filterAndFormatTranscription(self, transcription, segmentLoudness, audioData):
        """
        Applies filtering rules (silence, false positives, banned words) and formatting
        to the raw transcription.
        Returns:
            tuple[bool, str]: (shouldOutput, formattedText)
        """
        cleanedText = transcription.strip() if isinstance(transcription, str) else ""
        originalCleanedText = cleanedText

        # Filter out empty or meaningless transcriptions early
        if not cleanedText or cleanedText == ".":
            uniDebugLogger.debug("Transcription is effectively empty after initial strip.")
            return False, ""

        cleanedTextLower = cleanedText.lower()

        # Apply loudness-based filtering if audio data is available
        if segmentLoudness != -1:
            if self._shouldSkipTranscriptionDueToSilenceOrLowContent(segmentLoudness, audioData):
                uniDebugLogger.debug(
                    f"Filtered due to silence/low content. Original: '{originalCleanedText}'")
                return False, ""
            if self._isFalsePositive(cleanedTextLower, segmentLoudness):
                uniDebugLogger.debug(
                    f"Filtered as false positive. Original: '{originalCleanedText}'")
                return False, ""

        # Remove any configured banned words from the transcription
        bannedWords = self.config.get('bannedWords', [])
        if bannedWords:
            textBeforeBanning = cleanedText
            for wordToBan in bannedWords:
                if not wordToBan: continue
                try:
                    # Use regex for case-insensitive replacement
                    cleanedText = re.sub(re.escape(wordToBan), '', cleanedText, flags=re.IGNORECASE)
                except Exception as reErr:
                    uniLogger.warning(f"Regex error banning word '{wordToBan}': {reErr}")
            # Clean up extra spaces if words were removed
            if len(cleanedText) < len(textBeforeBanning):
                uniDebugLogger.debug(
                    f"Removed banned words. Before: '{textBeforeBanning}', After: '{cleanedText}'")
                cleanedText = ' '.join(cleanedText.split())

        # Apply final formatting
        formattedText = cleanedText.strip()
        if self.config.get('removeTrailingDots', True):
            formattedText = formattedText.rstrip('. ')

        # Final check to ensure text is not empty after all processing
        if not formattedText:
            uniDebugLogger.debug(
                f"Text empty after all filtering. Original: '{originalCleanedText}'")
            return False, ""

        uniDebugLogger.debug(f"Final formatted text: '{formattedText}'")
        return True, formattedText

    def _shouldSkipTranscriptionDueToSilenceOrLowContent(self, segmentMeanLoudness, audioData):
        """
        Checks if the transcription should be ignored based on combined silence and
        minimum content duration rules. Returns True if the segment SHOULD be skipped.
        """
        if audioData is None or len(audioData) == 0: return False
        sampleRate = self.config.get('actualSampleRate')
        if not sampleRate or sampleRate <= 0: return False

        # Get filtering parameters from config
        chunkSilenceThreshold = self.config.get('dictationMode_silenceLoudnessThreshold', 0.001)
        minLoudDuration = self.config.get('minLoudDurationForTranscription', 0.3)
        silenceSkipThreshold = self.config.get('silenceSkip_threshold', 0.0002)
        checkLeadingSec = self.config.get('skipSilence_beforeNSecSilence', 0.3)
        checkTrailingSec = self.config.get('skipSilence_afterNSecSilence', 0.3)

        # 1. Check if the total duration of "loud" audio is sufficient
        if minLoudDuration > 0:
            loudSamplesMask = np.abs(audioData) >= chunkSilenceThreshold
            totalLoudDuration = np.sum(loudSamplesMask) / sampleRate
            if totalLoudDuration < minLoudDuration:
                uniDebugLogger.debug(
                    f"Silence skip: Loud duration ({totalLoudDuration:.2f}s) < min ({minLoudDuration:.2f}s).")
                return True

        # 2. If the average loudness is high enough, don't skip
        if segmentMeanLoudness >= silenceSkipThreshold:
            return False

        # 3. If average loudness is low, check for loud segments at the start/end to override skipping
        uniDebugLogger.debug(
            f"Low mean loudness ({segmentMeanLoudness:.6f}) < skip threshold ({silenceSkipThreshold:.6f}). Checking overrides...")
        if checkLeadingSec > 0:
            leadingSamples = min(int(checkLeadingSec * sampleRate), len(audioData))
            if leadingSamples > 0 and np.mean(
                    np.abs(audioData[:leadingSamples])) >= chunkSilenceThreshold:
                uniDebugLogger.debug("Silence skip OVERRIDDEN: Leading segment is loud enough.")
                return False
        if checkTrailingSec > 0:
            trailingSamples = min(int(checkTrailingSec * sampleRate), len(audioData))
            if trailingSamples > 0 and np.mean(
                    np.abs(audioData[-trailingSamples:])) >= chunkSilenceThreshold:
                uniDebugLogger.debug("Silence skip OVERRIDDEN: Trailing segment is loud enough.")
                return False

        # If no overrides were met, confirm skipping
        uniDebugLogger.debug("Silence skip CONFIRMED: Low avg loudness and no overrides.")
        return True

    def _isFalsePositive(self, cleanedTextLower, segmentLoudness):
        """Checks if the transcription is a common false positive word detected in low loudness."""
        commonFalseWords = self.config.get('commonFalseDetectedWords', [])
        if not commonFalseWords: return False

        # Normalize text for comparison (remove punctuation, extra spaces)
        translator = str.maketrans('', '', string.punctuation)
        checkText = ' '.join(cleanedTextLower.translate(translator).strip().split())
        commonFalseWords_normalized = set(
            ' '.join(w.lower().translate(translator).strip().split()) for w in commonFalseWords if
            w)

        # Check if the text matches a common false positive
        if checkText in commonFalseWords_normalized:
            loudnessThreshold = self.config.get('loudnessThresholdOf_commonFalseDetectedWords',
                                                0.0008)
            # Only filter if the audio loudness is below the threshold
            if segmentLoudness < loudnessThreshold:
                uniDebugLogger.debug(
                    f"'{checkText}' is false positive (Loudness {segmentLoudness:.6f} < {loudnessThreshold:.6f}). Filtering.")
                return True
            else:
                uniDebugLogger.debug(
                    f"'{checkText}' matches false positive BUT loudness >= threshold. Not filtering.")
        return False

    def _handleValidOutput(self, finalText):
        """Handles actions for valid, filtered transcription text."""
        # Log the final output
        uniLogger.info(f"Output: {finalText}")
        # If output is enabled, send the text to be typed or copied to clipboard
        if self.stateManager.isOutputEnabled():
            # Allow user to hold CTRL to prevent typing
            if not self.systemInteractionHandler.isModifierKeyPressed("ctrl"):
                self.systemInteractionHandler.typeText(finalText)
            else:
                uniDebugLogger.debug("CTRL key pressed, skipping text output action.")
        # Reset the idle timer since valid text was produced
        self.stateManager.updateLastValidTranscriptionTime()

    def _handleSilentOrFilteredSegment(self):
        """Handles actions when transcription is empty, silent, or filtered out."""
        # No action is needed here, as logging is done within the filtering methods.
        # The idle timer is NOT reset.
        pass


# ==================================
# File Transcriber Class
# ==================================
class FileTranscriber:
    """
    Handles transcription of pre-recorded audio files using a provided ASR handler.
    """

    def __init__(self, config, asrModelHandler):
        """Initializes the transcriber with config and an ASR model handler."""
        self.config = config
        self.asrModelHandler = asrModelHandler

    def transcribeFile(self, audioFilePath, outputFilePath=None):
        """
        Transcribes an audio file and optionally saves the transcription to a text file.
        Returns:
            str | None: The transcribed text, or None if transcription fails.
        """
        uniDebugLogger.debug(f"Attempting to transcribe file: {audioFilePath}")
        audioFilePath = Path(audioFilePath)
        handlerType = type(self.asrModelHandler).__name__
        uniDebugLogger.debug(f"Using ASR Handler: {handlerType}")

        try:
            # --- Ensure Model is Ready ---
            if not self.asrModelHandler.isModelLoaded():
                uniLogger.info(f"ASR model ({handlerType}) not ready, attempting to load...")
                if not self.asrModelHandler.loadModel():
                    uniLogger.error(f"Model ({handlerType}) failed to load. Aborting.")
                    return None

            # --- Read Audio File ---
            if not audioFilePath.is_file():
                uniLogger.error(f"Audio file not found at {audioFilePath}")
                return None

            try:
                # Use soundfile for robust reading of various audio formats
                audioData, sampleRate = sf.read(audioFilePath, dtype='float32', always_2d=False)
            except Exception as e:
                uniLogger.error(f"Error reading audio file {audioFilePath}: {e}", excInfo=True)
                uniLogger.error("Hint: Ensure the file is a valid audio format and not corrupted.")
                return None

            fileDuration = len(audioData) / sampleRate if sampleRate > 0 else 0
            uniLogger.info(
                f"Audio file read: {audioFilePath.name} (Rate: {sampleRate}Hz, Duration: {fileDuration:.2f}s)")

            # --- Perform Transcription ---
            uniLogger.info("Starting transcription...")
            startTime = time.time()
            transcription = self.asrModelHandler.transcribeAudioSegment(audioData, sampleRate)
            elapsedTime = time.time() - startTime
            uniLogger.info(f"Transcription finished in {elapsedTime:.2f} seconds.")

            # --- Post-Processing & Output ---
            if transcription is not None:
                finalText = transcription.strip()
                if self.config.get('removeTrailingDots', True):
                    finalText = finalText.rstrip('. ')

                self._handleOutput(finalText, outputFilePath)
                return finalText
            else:
                uniLogger.warning("Transcription failed or returned None from handler.")
                return None
        except Exception as e:
            uniLogger.critical(f"Unexpected error during file transcription '{audioFilePath}': {e}",
                               excInfo=True)
            return None

    def _handleOutput(self, transcription, outputFilePath):
        """Saves transcription to a file or logs it to the console."""
        if outputFilePath:
            try:
                outputFilePath = Path(outputFilePath)
                # Ensure the output directory exists
                outputFilePath.parent.mkdir(parents=True, exist_ok=True)
                with open(outputFilePath, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                uniLogger.info(f"Transcription saved to: {outputFilePath}")
            except IOError as e:
                # If file writing fails, print the transcription to the log as a fallback
                uniLogger.error(f"Error writing transcription to file {outputFilePath}: {e}",
                                excInfo=True)
                uniLogger.info("\n--- Transcription Fallback Log ---\n" + transcription)
        else:
            # If no output path is specified, print the result to the log
            uniLogger.info("\n--- Transcription Result ---\n" + transcription)

    def cleanup(self):
        """Optional cleanup method for the file transcriber."""
        # The ASR handler's lifecycle is managed externally, so this is usually not needed.
        uniLogger.info("FileTranscriber cleanup.")
