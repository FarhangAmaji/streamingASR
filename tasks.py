# tasks.py
import re  # Need for regex replacement
import string
import time
import traceback
from pathlib import Path

import numpy as np
# Pygame and PyAutoGUI are imported conditionally later where needed
import soundfile as sf

# Import logging helpers from utils
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
        self.systemInteractionHandler = systemInteractionHandler

    # No _logDebug needed here, use imported logDebug directly
    def _calculateSegmentLoudness(self, audioData):
        """Calculates the average absolute amplitude of the entire segment."""
        if audioData is None or len(audioData) == 0:
            return 0.0
        # Ensure float for calculation
        if audioData.dtype.kind != 'f':
            # Basic normalization assuming int16 range if integer
            if audioData.dtype.kind in ('i', 'u'):
                maxVal = np.iinfo(audioData.dtype).max
                minVal = np.iinfo(audioData.dtype).min
                # Avoid division by zero for zero range types
                if maxVal > minVal:
                    audioData = (audioData.astype(np.float32) - minVal) / (
                            maxVal - minVal) * 2.0 - 1.0
                else:
                    audioData = audioData.astype(np.float32)
            else:  # Just cast other types
                audioData = audioData.astype(np.float32)
        return np.mean(np.abs(audioData))

    def processTranscriptionResult(self, transcription, audioData):
        """
        Processes the ASR result: checks for silence, filters false positives,
        formats, and triggers output (print/type/clipboard). Updates idle timer.
        Requires audioData for loudness-based filtering.
        """
        if audioData is None or len(audioData) == 0:
            logWarning("Processing transcription result without audio data for loudness checks.")
            segmentLoudness = -1  # Indicate unavailable loudness
        else:
            segmentLoudness = self._calculateSegmentLoudness(audioData)
            logDebug(f"Processing transcription. Segment Avg Loudness = {segmentLoudness:.6f}")
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
        Applies filtering rules (silence, false positives, banned words) and formatting
        to the raw transcription.
        Returns:
            tuple[bool, str]: (shouldOutput, formattedText)
        """
        cleanedText = transcription.strip() if isinstance(transcription, str) else ""
        originalCleanedText = cleanedText  # Keep original for logging comparison if needed
        if not cleanedText or cleanedText == ".":
            logDebug("Transcription is effectively empty after initial strip.")
            return False, ""
        cleanedTextLower = cleanedText.lower()  # Used for checks
        # --- Silence/Low Content Filtering ---
        if segmentLoudness != -1:
            if self._shouldSkipTranscriptionDueToSilenceOrLowContent(segmentLoudness, audioData):
                logDebug(
                    f"Filtered due to silence/low content rules. Original: '{originalCleanedText}'")
                return False, ""
        else:
            logDebug("Skipping silence/low content filtering due to missing audio data.")
        # --- False Positive Filtering (Loudness Dependent) ---
        if segmentLoudness != -1:
            if self._isFalsePositive(cleanedTextLower, segmentLoudness):
                logDebug(f"Filtered as false positive. Original: '{originalCleanedText}'")
                return False, ""
        else:
            logDebug("Skipping false positive filtering due to missing audio data.")
        # --- Banned Word Filtering (Always Applied) ---
        bannedWords = self.config.get('bannedWords', [])
        if bannedWords:
            textBeforeBanning = cleanedText  # For logging comparison
            for wordToBan in bannedWords:
                if not wordToBan: continue  # Skip empty strings in the list
                # Use re.sub for case-insensitive replacement
                # This replaces the word regardless of its surrounding characters (e.g., "assistant" in "assistants")
                # For whole-word matching only, use boundaries: r'\b' + re.escape(wordToBan) + r'\b'
                try:
                    # Simple case-insensitive replacement:
                    cleanedText = re.sub(re.escape(wordToBan), '', cleanedText, flags=re.IGNORECASE)
                except Exception as reErr:
                    logWarning(f"Regex error banning word '{wordToBan}': {reErr}")
            # Clean up extra spaces potentially left by removal
            if len(cleanedText) < len(textBeforeBanning):  # Only cleanup if something was removed
                logDebug(
                    f"Removed banned words. Before: '{textBeforeBanning}', After: '{cleanedText}'")
                # Replace multiple spaces with a single space
                cleanedText = ' '.join(cleanedText.split())
        # --- Final Formatting & Empty Check ---
        formattedText = cleanedText.strip()  # Strip again after potential removals/space cleanup
        if self.config.get('removeTrailingDots', True):
            formattedText = formattedText.rstrip('. ')
        if not formattedText:
            logDebug(
                f"Text became empty after all filtering/formatting. Original: '{originalCleanedText}'")
            return False, ""
        logDebug(f"Final formatted text ready for output: '{formattedText}'")
        return True, formattedText

    def _shouldSkipTranscriptionDueToSilenceOrLowContent(self, segmentMeanLoudness, audioData):
        """
        Checks if the transcription should be ignored based on combined silence and
        minimum content duration rules. Requires valid audioData.
        Returns True if the segment SHOULD be skipped, False otherwise.
        """
        if audioData is None or len(audioData) == 0:
            logWarning("Silence check called without audio data, cannot perform check.")
            return False  # Don't skip if we can't check
        sampleRate = self.config.get('actualSampleRate')
        if not sampleRate or sampleRate <= 0:
            logWarning("Invalid sample rate for silence check.")
            return False  # Cannot perform check
        # Use get with defaults for robustness
        chunkSilenceThreshold = self.config.get('dictationMode_silenceLoudnessThreshold', 0.001)
        minLoudDuration = self.config.get('minLoudDurationForTranscription', 0.3)  # Reduced default
        silenceSkipThreshold = self.config.get('silenceSkip_threshold', 0.0002)
        checkLeadingSec = self.config.get('skipSilence_beforeNSecSilence', 0.3)  # Check first 0.3s
        checkTrailingSec = self.config.get('skipSilence_afterNSecSilence', 0.3)  # Check last 0.3s
        # --- 1. Minimum Loud Duration Check ---
        if minLoudDuration > 0:
            # Ensure audioData is float for comparison
            if audioData.dtype.kind != 'f': audioData = audioData.astype(np.float32)
            loudSamplesMask = np.abs(audioData) >= chunkSilenceThreshold
            numLoudSamples = np.sum(loudSamplesMask)
            totalLoudDuration = numLoudSamples / sampleRate
            if totalLoudDuration < minLoudDuration:
                logDebug(
                    f"Silence skip CONFIRMED: Total loud duration ({totalLoudDuration:.2f}s) < min ({minLoudDuration:.2f}s). (Avg Loudness: {segmentMeanLoudness:.6f})")
                return True
            # else:
            #    logDebug(f"Passed min loud duration check ({totalLoudDuration:.2f}s >= {minLoudDuration:.2f}s).")
        # --- 2. Average Loudness Check ---
        if segmentMeanLoudness >= silenceSkipThreshold:
            # logDebug(f"Segment mean loudness ({segmentMeanLoudness:.6f}) >= skip threshold ({silenceSkipThreshold:.6f}). Not skipping.")
            return False  # DO NOT SKIP
        # --- 3. Low Average Loudness - Check Overrides ---
        logDebug(
            f"Segment passed min loud duration but mean loudness ({segmentMeanLoudness:.6f}) < skip threshold ({silenceSkipThreshold:.6f}). Checking overrides...")
        # Check Beginning
        if checkLeadingSec > 0:
            leadingSamples = min(int(checkLeadingSec * sampleRate),
                                 len(audioData))  # Avoid over-indexing
            if leadingSamples > 0:
                leadingAudio = audioData[:leadingSamples]
                # Ensure float for calculation
                if leadingAudio.dtype.kind != 'f': leadingAudio = leadingAudio.astype(np.float32)
                leadingLoudness = np.mean(np.abs(leadingAudio))
                if leadingLoudness >= chunkSilenceThreshold:
                    logDebug(
                        f"Silence skip OVERRIDDEN: Leading {checkLeadingSec:.2f}s loud enough ({leadingLoudness:.6f}).")
                    return False  # DO NOT SKIP
        # Check Trailing
        if checkTrailingSec > 0:
            trailingSamples = min(int(checkTrailingSec * sampleRate),
                                  len(audioData))  # Avoid over-indexing
            if trailingSamples > 0:
                trailingAudio = audioData[-trailingSamples:]
                # Ensure float for calculation
                if trailingAudio.dtype.kind != 'f': trailingAudio = trailingAudio.astype(np.float32)
                trailingLoudness = np.mean(np.abs(trailingAudio))
                if trailingLoudness >= chunkSilenceThreshold:
                    logDebug(
                        f"Silence skip OVERRIDDEN: Trailing {checkTrailingSec:.2f}s loud enough ({trailingLoudness:.6f}).")
                    return False  # DO NOT SKIP
        # --- 4. Final Decision (Low Avg, No Overrides) ---
        logDebug(
            f"Silence skip CONFIRMED: Low avg loudness ({segmentMeanLoudness:.6f}) and no start/end overrides triggered.")
        return True  # SKIP

    def _isFalsePositive(self, cleanedTextLower, segmentLoudness):
        """
        Checks if the transcription is a common false word detected in low loudness.
        Requires segmentLoudness.
        """
        commonFalseWords = self.config.get('commonFalseDetectedWords', [])
        if not commonFalseWords:
            return False
        # Normalize the lowercased text for comparison (remove punctuation, extra spaces)
        translator = str.maketrans('', '', string.punctuation)
        checkText = cleanedTextLower.translate(translator).strip()
        checkText = ' '.join(checkText.split())  # Collapse multiple spaces
        # Ensure words in the config list are also normalized (lower, no punctuation)
        commonFalseWords_normalized = set(
            ' '.join(w.lower().translate(translator).strip().split())
            for w in commonFalseWords if w  # Handle potential empty strings in list
        )
        # Remove potential empty strings resulted from normalization
        commonFalseWords_normalized.discard('')
        if checkText in commonFalseWords_normalized:
            loudnessThreshold = self.config.get('loudnessThresholdOf_commonFalseDetectedWords',
                                                0.0008)
            if segmentLoudness < loudnessThreshold:
                logDebug(
                    f"'{checkText}' IS false positive (Loudness {segmentLoudness:.6f} < {loudnessThreshold:.6f}). Filtering.")
                return True
            else:
                logDebug(
                    f"'{checkText}' matches false positive BUT loudness ({segmentLoudness:.6f}) >= threshold. Not filtering.")
        return False

    def _handleValidOutput(self, finalText):
        """Handles actions for valid, filtered transcription text."""
        # Always log the valid output
        logInfo(f"Output: {finalText}")
        # Use system interaction handler for typing/clipboard based on OS/config
        # Check output state and modifier keys
        if self.stateManager.isOutputEnabled():
            # Check if CTRL is pressed (common 'copy' or 'pause typing' intention)
            if not self.systemInteractionHandler.isModifierKeyPressed("ctrl"):
                self.systemInteractionHandler.typeText(
                    finalText)  # Pass text without trailing space here
            else:
                logDebug("CTRL key pressed, skipping text output action.")
        # else: # Output is disabled state handled by isOutputEnabled check
        #    logDebug("Text output skipped: Output state is disabled.")
        # Reset the idle timer only when valid output is produced
        self.stateManager.updateLastValidTranscriptionTime()

    def _handleSilentOrFilteredSegment(self):
        """Handles actions when transcription is empty, silent, or filtered."""
        # Logging is done within the filtering methods (_shouldSkip..., _isFalsePositive...)
        # No valid output, so the idle timer is NOT reset here.
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
             asrModelHandler (AbstractAsrModelHandler): The ASR model handler to use.
        """
        self.config = config
        self.asrModelHandler = asrModelHandler
        # No _logDebug needed here

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
        logDebug(f"Attempting to transcribe file: {audioFilePath}")
        transcription = None
        audioFilePath = Path(audioFilePath)
        handlerType = type(self.asrModelHandler).__name__
        logDebug(f"Using ASR Handler: {handlerType}")
        try:
            # --- Ensure Model is Ready ---
            if not self.asrModelHandler.isModelLoaded():
                logInfo(
                    f"ASR model ({handlerType}) not ready for file transcription, attempting to load/prepare...")
                loadSuccess = self.asrModelHandler.loadModel()  # Trigger load/check
                if not loadSuccess:
                    # loadModel method should log errors
                    logError(f"Model ({handlerType}) failed to load. File transcription aborted.")
                    return None
            # Check again even if load reported success, just in case state is inconsistent
            if not self.asrModelHandler.isModelLoaded():
                logError(
                    f"Model ({handlerType}) still not loaded after load attempt. File transcription aborted.")
                return None
            # --- Read Audio File ---
            if not audioFilePath.is_file():
                logError(f"Audio file not found at {audioFilePath}")
                return None  # Return None on file not found
            try:
                # Use soundfile for robust reading
                audioData, sampleRate = sf.read(audioFilePath, dtype='float32', always_2d=False)
            except Exception as e:
                logError(f"Error reading audio file {audioFilePath} using soundfile: {e}")
                logError(
                    "Hint: Ensure the file is a valid audio format (WAV, FLAC, OGG, etc.) and not corrupted.")
                return None
            fileDuration = len(audioData) / sampleRate if sampleRate > 0 else 0
            logInfo(
                f"Audio file read successfully: {audioFilePath.name} (Sample Rate: {sampleRate}Hz, Duration: {fileDuration:.2f}s)")
            # --- Perform Transcription ---
            logInfo("Starting transcription...")
            startTime = time.time()
            # Delegate transcription entirely to the handler
            transcription = self.asrModelHandler.transcribeAudioSegment(audioData, sampleRate)
            elapsedTime = time.time() - startTime
            logInfo(f"Transcription finished in {elapsedTime:.2f} seconds.")
            # --- Post-Processing & Output ---
            if transcription is not None and isinstance(transcription, str):
                finalText = transcription.strip()
                if self.config.get('removeTrailingDots', True):
                    finalText = finalText.rstrip('. ')
                if finalText:  # Check if not empty after stripping
                    self._handleOutput(finalText, outputFilePath)
                    return finalText  # Return the final processed text
                else:
                    logInfo("Transcription result was empty after processing.")
                    return ""  # Return empty string for consistency
            else:
                # Handler should log transcription errors
                logWarning("Transcription failed or returned unexpected type from handler.")
                return None  # Indicate failure
        except FileNotFoundError as e:
            # This is redundant as we check is_file earlier, but keep for safety
            logError(str(e))
            return None
        except Exception as e:
            logError(f"Unexpected error during file transcription '{audioFilePath}': {e}")
            logError(traceback.format_exc())  # Log full traceback
            return None

    def _handleOutput(self, transcription, outputFilePath):
        """Saves transcription to file or logs to console."""
        if outputFilePath:
            try:
                outputFilePath = Path(outputFilePath)
                outputFilePath.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                with open(outputFilePath, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                logInfo(f"Transcription saved to: {outputFilePath}")
            except IOError as e:
                logError(f"Error writing transcription to file {outputFilePath}: {e}")
                # Fallback log
                logInfo(
                    "\n--- Transcription Fallback Log ---\n" + transcription + "\n--------------------------------\n")
        else:
            # Log transcription result clearly when not writing to file
            logInfo(
                "\n--- Transcription Result ---\n" + transcription + "\n--------------------------\n")

    def cleanup(self):
        """Optional cleanup for file transcriber."""
        logInfo("FileTranscriber cleanup.")
        # Usually, the ASR handler lifecycle is managed externally (e.g., by the main orchestrator)
        # If this FileTranscriber instance exclusively owns the handler, cleanup here:
        # if self.asrModelHandler:
        #     self.asrModelHandler.cleanup()
