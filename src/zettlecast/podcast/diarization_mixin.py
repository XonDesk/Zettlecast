"""
Diarization Mixin

Shared pyannote.audio diarization functionality for transcribers.
Allows any transcriber to add speaker detection capabilities.
"""

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class DiarizationMixin:
    """
    Mixin that adds pyannote.audio diarization to any transcriber.
    
    Usage:
        class MyTranscriber(BaseTranscriber, DiarizationMixin):
            def transcribe(self, audio_path, episode=None):
                # Get transcription segments
                segments = self._do_transcription(audio_path)
                
                # Apply diarization if enabled
                if self.config.enable_diarization:
                    segments = self._apply_diarization(audio_path, segments)
                
                return result
    """
    
    _diarization_pipeline = None
    _diarization_model_id = "pyannote/speaker-diarization-3.1"
    
    def _is_diarization_available(self) -> bool:
        """Check if pyannote.audio is available."""
        try:
            import pyannote.audio
            return True
        except ImportError:
            return False
    
    def _load_diarization_pipeline(self):
        """Load pyannote diarization pipeline."""
        if self._diarization_pipeline is None:
            if not self.config.enable_diarization:
                return None
            
            if not self.config.hf_token:
                logger.warning(
                    "No HuggingFace token provided. Diarization requires "
                    "accepting pyannote's license at: "
                    "https://huggingface.co/pyannote/speaker-diarization-3.1"
                )
                return None
            
            from pyannote.audio import Pipeline
            
            logger.info(f"Loading pyannote diarization: {self._diarization_model_id}")
            self._diarization_pipeline = Pipeline.from_pretrained(
                self._diarization_model_id,
                token=self.config.hf_token,  # pyannote 4.x uses 'token' instead of 'use_auth_token'
            )
            logger.info("Pyannote diarization pipeline loaded")
        
        return self._diarization_pipeline
    
    def _run_diarization(self, audio_path: Path):
        """
        Run diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            pyannote.core.Annotation or None if diarization failed/disabled
        """
        pipeline = self._load_diarization_pipeline()
        if pipeline is None:
            return None
        
        logger.info(f"Running pyannote diarization on: {audio_path.name}")
        annotation = pipeline(str(audio_path))
        
        # Count speakers
        speakers = set()
        for _, _, speaker in annotation.itertracks(yield_label=True):
            speakers.add(speaker)
        
        logger.info(f"Diarization complete: {len(speakers)} speakers detected")
        return annotation
    
    def _apply_diarization_to_segments(
        self,
        audio_path: Path,
        segments,  # List[TranscriptSegment]
    ):
        """
        Apply diarization to existing transcript segments.
        
        For transcribers that don't have word-level timestamps,
        this assigns speakers based on segment overlap with diarization.
        
        Args:
            audio_path: Path to audio file
            segments: List of TranscriptSegment objects
            
        Returns:
            Updated segments with speaker labels, and speaker count
        """
        from .aligner import parse_pyannote_annotation, SpeakerSegment
        from .models import TranscriptSegment
        
        annotation = self._run_diarization(audio_path)
        if annotation is None:
            return segments, 1
        
        # Parse annotation to speaker segments
        speaker_segments = parse_pyannote_annotation(annotation)
        
        if not speaker_segments:
            return segments, 1
        
        # Assign speakers to segments based on overlap
        updated_segments = []
        speakers = set()
        
        for seg in segments:
            # Find overlapping speaker segments
            overlapping = []
            for ss in speaker_segments:
                # Check overlap
                if not (seg.end <= ss.start or seg.start >= ss.end):
                    # Calculate overlap duration
                    overlap_start = max(seg.start, ss.start)
                    overlap_end = min(seg.end, ss.end)
                    overlap_duration = overlap_end - overlap_start
                    overlapping.append((ss, overlap_duration))
            
            if overlapping:
                # Assign to speaker with most overlap
                best_speaker = max(overlapping, key=lambda x: x[1])[0].speaker
            else:
                # Find closest speaker segment
                closest = min(
                    speaker_segments,
                    key=lambda s: min(abs(s.start - seg.start), abs(s.end - seg.end))
                )
                best_speaker = closest.speaker
            
            speakers.add(best_speaker)
            
            updated_segments.append(
                TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    speaker=best_speaker,
                )
            )
        
        return updated_segments, len(speakers)
    
    def _apply_diarization_to_words(
        self,
        audio_path: Path,
        words,  # List of word objects with start/end
    ):
        """
        Apply diarization to word-level timestamps.
        
        For transcribers with word-level timestamps, this provides
        more accurate speaker assignment.
        
        Args:
            audio_path: Path to audio file
            words: List of Word objects (from aligner.py)
            
        Returns:
            List of TranscriptSegment objects with speaker labels, and speaker count
        """
        from .aligner import align_with_pyannote
        
        annotation = self._run_diarization(audio_path)
        if annotation is None:
            # Return words grouped as single-speaker segments
            from .models import TranscriptSegment
            if not words:
                return [], 1
            
            segments = []
            current_words = [words[0]]
            
            for word in words[1:]:
                if word.start - current_words[-1].end > 2.0:
                    segments.append(
                        TranscriptSegment(
                            start=current_words[0].start,
                            end=current_words[-1].end,
                            text=" ".join(w.text for w in current_words),
                            speaker=None,
                        )
                    )
                    current_words = [word]
                else:
                    current_words.append(word)
            
            if current_words:
                segments.append(
                    TranscriptSegment(
                        start=current_words[0].start,
                        end=current_words[-1].end,
                        text=" ".join(w.text for w in current_words),
                        speaker=None,
                    )
                )
            
            return segments, 1
        
        # Use aligner for word-level assignment
        aligned_segments = align_with_pyannote(words, annotation)
        
        # Convert aligner TranscriptSegment to models TranscriptSegment
        from .models import TranscriptSegment
        
        segments = []
        speakers = set()
        
        for seg in aligned_segments:
            speakers.add(seg.speaker)
            segments.append(
                TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    speaker=seg.speaker,
                )
            )
        
        return segments, len(speakers)
