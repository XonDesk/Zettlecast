"""
Mac Transcriber - Apple Silicon Optimized

Uses parakeet-mlx for transcription and pyannote.audio for diarization.
Designed for Mac M-series chips using Apple's MLX framework.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

from .aligner import Word, align_transcription_with_diarization
from .base_transcriber import BaseTranscriber, TranscriberCapabilities, TranscriberConfig
from .models import PodcastEpisode, TranscriptSegment, TranscriptionResult

logger = logging.getLogger(__name__)


class MacTranscriber(BaseTranscriber):
    """
    Mac-optimized transcriber using Apple Silicon MLX.
    
    Transcription: parakeet-mlx (NVIDIA Parakeet ported to MLX)
    Diarization: pyannote.audio 3.1
    VAD: Silero VAD (used by pyannote internally)
    """
    
    def __init__(self, config: Optional[TranscriberConfig] = None):
        super().__init__(config)
        
        # Models loaded lazily
        self._parakeet_model = None
        self._diarization_pipeline = None
        
        # Default model paths - handle 'auto' value
        model_config = getattr(self.config, 'transcription_model', None)
        if model_config and model_config != "auto":
            self.parakeet_model_id = model_config
        else:
            self.parakeet_model_id = "mlx-community/parakeet-tdt-0.6b-v3"
        
        self.diarization_model_id = "pyannote/speaker-diarization-3.1"
    
    def is_available(self) -> bool:
        """Check if parakeet-mlx is available."""
        try:
            import parakeet_mlx
            return True
        except ImportError:
            return False
    
    def _load_parakeet(self):
        """Load parakeet-mlx model."""
        if self._parakeet_model is None:
            from parakeet_mlx import from_pretrained
            
            logger.info(f"Loading Parakeet-MLX model: {self.parakeet_model_id}")
            self._parakeet_model = from_pretrained(self.parakeet_model_id)
            logger.info("Parakeet-MLX model loaded")
        
        return self._parakeet_model
    
    def _load_diarization(self):
        """Load pyannote diarization pipeline."""
        if self._diarization_pipeline is None:
            if not self.config.enable_diarization:
                return None
            
            if not self.config.hf_token:
                logger.warning(
                    "No HuggingFace token provided. Diarization requires "
                    "accepting pyannote's license on HuggingFace."
                )
                return None
            
            from pyannote.audio import Pipeline
            
            logger.info(f"Loading pyannote diarization: {self.diarization_model_id}")
            self._diarization_pipeline = Pipeline.from_pretrained(
                self.diarization_model_id,
                token=self.config.hf_token,  # pyannote 4.x uses 'token' instead of 'use_auth_token'
            )
            logger.info("Pyannote diarization pipeline loaded")
        
        return self._diarization_pipeline
    
    def warmup(self) -> None:
        """Pre-load models."""
        self._load_parakeet()
        if self.config.enable_diarization:
            self._load_diarization()
    
    def get_capabilities(self) -> TranscriberCapabilities:
        """Return Mac transcriber capabilities."""
        return TranscriberCapabilities(
            platform="darwin",
            transcriber_name="parakeet-mlx",
            diarizer_name="pyannote.audio" if self.config.enable_diarization else None,
            vad_name="silero",
            supports_diarization=True,
            supports_gpu=True,  # Apple Silicon Neural Engine
            supports_streaming=True,  # parakeet-mlx supports streaming
            requires_container=False,
        )
    
    def transcribe(
        self,
        audio_path: Path,
        episode: Optional[PodcastEpisode] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio using parakeet-mlx with optional pyannote diarization.
        
        Args:
            audio_path: Path to audio file
            episode: Optional episode metadata
            
        Returns:
            TranscriptionResult with segments and metadata
        """
        import subprocess
        
        start_time = time.time()
        audio_path = Path(audio_path)
        
        # Get audio duration for progress estimates
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
                capture_output=True, text=True, timeout=10
            )
            audio_duration = float(result.stdout.strip())
            est_transcribe_time = audio_duration / 10  # ~10x realtime on M-series
        except:
            audio_duration = 0
            est_transcribe_time = 0
        
        print(f"\n📄 File: {audio_path.name}")
        if audio_duration:
            print(f"   Duration: {audio_duration/60:.1f} minutes")
            print(f"   Est. transcription time: ~{est_transcribe_time/60:.1f} minutes")
        
        # Step 1: Load model
        print(f"\n[1/4] Loading Parakeet-MLX model...", flush=True)
        step_start = time.time()
        model = self._load_parakeet()
        print(f"      ✓ Model loaded ({time.time()-step_start:.1f}s)", flush=True)
        
        # Step 2: Transcribe
        print(f"\n[2/4] Transcribing audio...", flush=True)
        step_start = time.time()
        parakeet_result = model.transcribe(
            str(audio_path),
            chunk_duration=self.config.chunk_duration_minutes * 60.0,
            overlap_duration=15.0,  # 15 second overlap between chunks
        )
        print(f"      ✓ Transcription complete ({time.time()-step_start:.1f}s)", flush=True)
        
        # Extract words with timestamps from parakeet result
        words = self._extract_words(parakeet_result)
        print(f"      Words extracted: {len(words)}", flush=True)
        
        # Step 3: Diarization (if enabled)
        transcript_segments = []
        speakers_detected = 1
        
        if self.config.enable_diarization:
            print(f"\n[3/4] Running speaker diarization...", flush=True)
            step_start = time.time()
            diarizer = self._load_diarization()
            if diarizer:
                try:
                    print(f"      Analyzing speakers (this may take a while)...", flush=True)
                    diarization = diarizer(str(audio_path))
                    print(f"      ✓ Diarization complete ({time.time()-step_start:.1f}s)", flush=True)
                    
                    # Convert pyannote annotation to RTTM format for aligner
                    rttm_content = self._annotation_to_rttm(diarization)
                    
                    # Align words with speakers
                    print(f"      Aligning words with speakers...", flush=True)
                    aligned_segments = align_transcription_with_diarization(
                        words, rttm_content
                    )
                    
                    # Convert to TranscriptSegment
                    speakers = set()
                    for seg in aligned_segments:
                        transcript_segments.append(
                            TranscriptSegment(
                                start=seg.start,
                                end=seg.end,
                                text=seg.text,
                                speaker=seg.speaker,
                            )
                        )
                        speakers.add(seg.speaker)
                    
                    speakers_detected = len(speakers)
                    print(f"      ✓ Found {speakers_detected} speakers", flush=True)
                except Exception as e:
                    print(f"      ⚠ Diarization failed: {str(e)[:60]}", flush=True)
                    print(f"      Continuing without speaker detection...", flush=True)
                    # Leave transcript_segments empty to trigger fallback below
                logger.info(f"Diarization complete: {speakers_detected} speakers detected")
        
        # If no diarization or it failed, create segments from words
        if not transcript_segments:
            transcript_segments = self._words_to_segments(words)
        
        # Step 4: Format and save
        print(f"\n[4/4] Formatting transcript...", flush=True)
        
        # Format full transcript
        full_text = self._format_transcript(transcript_segments)
        
        # Calculate duration from last word timestamp
        duration = words[-1].end if words else 0.0
        processing_time = time.time() - start_time
        
        print(f"      ✓ Complete! Total time: {processing_time/60:.1f} minutes", flush=True)
        if duration > 0:
            print(f"      Speed: {processing_time/duration:.2f}x realtime", flush=True)
        
        return TranscriptionResult(
            episode_id=episode.id if episode else "unknown",
            segments=transcript_segments,
            full_text=full_text,
            language=self.config.language,
            speakers_detected=speakers_detected,
            duration_seconds=duration,
            processing_time_seconds=processing_time,
        )
    
    def _extract_words(self, parakeet_result) -> List[Word]:
        """Extract Word objects from parakeet-mlx AlignedResult."""
        words = []
        
        for sentence in parakeet_result.sentences:
            for token in sentence.tokens:
                words.append(Word(
                    text=token.text,
                    start=token.start,
                    end=token.end,
                ))
        
        return words
    
    def _annotation_to_rttm(self, annotation) -> str:
        """Convert pyannote Annotation to RTTM format string."""
        lines = []
        
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            # RTTM format: SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
            start = turn.start
            duration = turn.end - turn.start
            lines.append(f"SPEAKER audio 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>")
        
        return "\n".join(lines)
    
    def _words_to_segments(self, words: List[Word]) -> List[TranscriptSegment]:
        """Convert word list to transcript segments (no speaker labels)."""
        if not words:
            return []
        
        # Group words into segments by pause gaps
        segments = []
        current_words = [words[0]]
        
        for word in words[1:]:
            # If gap > 2 seconds, start new segment
            if word.start - current_words[-1].end > 2.0:
                segments.append(TranscriptSegment(
                    start=current_words[0].start,
                    end=current_words[-1].end,
                    text=" ".join(w.text for w in current_words),
                    speaker=None,
                ))
                current_words = [word]
            else:
                current_words.append(word)
        
        # Add final segment
        if current_words:
            segments.append(TranscriptSegment(
                start=current_words[0].start,
                end=current_words[-1].end,
                text=" ".join(w.text for w in current_words),
                speaker=None,
            ))
        
        return segments
    
    def _format_transcript(self, segments: List[TranscriptSegment]) -> str:
        """Format segments into readable transcript."""
        lines = []
        for seg in segments:
            timestamp = f"[{seg.start:.1f}s]"
            speaker = f"{seg.speaker}: " if seg.speaker else ""
            lines.append(f"{timestamp} {speaker}{seg.text}")
        
        return "\n".join(lines)
