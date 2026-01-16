"""
NeMo-based Parallel Transcription and Diarization

Uses NVIDIA NeMo models for fast, parallel processing:
- Parakeet-TDT for transcription (~60 min in <2 sec)
- MSDD for speaker diarization (~1/10th real-time)

Pipeline:
1. Chunk audio into 10-minute segments
2. Run transcription and diarization in parallel for each chunk
3. Align word timestamps with speaker labels
4. Merge chunks back into complete transcript
"""

import logging
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from ..config import settings
from .aligner import Word, align_transcription_with_diarization
from .chunker import AudioChunk, chunk_audio
from .models import PodcastEpisode, TranscriptSegment, TranscriptionResult

logger = logging.getLogger(__name__)


class NeMoTranscriber:
    """
    Parallel transcription and diarization using NVIDIA NeMo models.

    Models:
    - nvidia/parakeet-tdt-0.6b-v2: Fast transcription with word timestamps
    - diar_msdd_telephonic: MSDD speaker diarization (RTTM output)
    """

    def __init__(
        self,
        device: str = "cpu",
        chunk_duration_minutes: int = 10,
        enable_diarization: bool = True,
    ):
        """
        Initialize NeMo transcriber.

        Args:
            device: Device for inference ('cpu' or 'cuda')
            chunk_duration_minutes: Duration of audio chunks for processing
            enable_diarization: Whether to perform speaker diarization
        """
        self.device = device or settings.whisper_device  # Reuse device config
        self.chunk_duration_minutes = chunk_duration_minutes
        self.enable_diarization = enable_diarization

        # Models loaded lazily
        self._parakeet_model = None

    def _transcribe_chunk_fallback(self, chunk: AudioChunk) -> Tuple[List[Word], float]:
        """
        Fallback transcription using faster-whisper if NeMo fails.
        """
        start_time = time.time()
        try:
            from faster_whisper import WhisperModel
            
            whisper = WhisperModel("large-v3-turbo", device=self.device, compute_type="float16" if self.device == "cuda" else "int8")
            segments_raw, info = whisper.transcribe(str(chunk.path), beam_size=5, language="en", vad_filter=True)
            segments_list = list(segments_raw)
            
            words = []
            for seg in segments_list:
                words.append(Word(
                    text=seg.text.strip(),
                    start=seg.start + chunk.start_time,
                    end=seg.end + chunk.start_time,
                ))
            
            logger.info(f"Chunk {chunk.chunk_index}: fallback transcribed {len(words)} words")
            return words, time.time() - start_time
        except Exception as e:
            logger.error(f"Fallback transcription also failed: {e}")
            return [], time.time() - start_time

    def _load_parakeet(self):
        """Load Parakeet transcription model."""
        if self._parakeet_model is None:
            try:
                # Avoid deadlock by importing and loading in correct order
                # Set to eval mode before device placement
                import torch
                import nemo.collections.asr as nemo_asr

                logger.info("Loading Parakeet-TDT model...")
                self._parakeet_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                    "nvidia/parakeet-tdt-0.6b-v2",
                    map_location=torch.device(self.device),  # Explicit device placement
                )
                self._parakeet_model.eval()
                if self.device == "cuda":
                    self._parakeet_model = self._parakeet_model.cuda()
                logger.info("Parakeet model loaded")
            except Exception as e:
                logger.error(f"Failed to load Parakeet model: {e}")
                raise

        return self._parakeet_model

    def _get_diarization_config(self, output_dir: Path, manifest_path: Path):
        """
        Create diarization config for NeuralDiarizer.

        NeMo's NeuralDiarizer requires a Hydra config with manifest and output paths.
        """
        from omegaconf import OmegaConf

        config = OmegaConf.create({
            "device": self.device,  # NeMo 2.x expects device at top level
            "diarizer": {
                "manifest_filepath": str(manifest_path),
                "out_dir": str(output_dir),
                "oracle_vad": False,
                "collar": 0.25,
                "ignore_overlap": True,
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "parameters": {
                        "onset": 0.8,
                        "offset": 0.6,
                        "min_duration_on": 0.1,
                        "min_duration_off": 0.1,
                        "pad_onset": 0.1,
                        "pad_offset": 0.1,
                    },
                },
                "speaker_embeddings": {
                    "model_path": "titanet_large",
                    "parameters": {
                        "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                        "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                        "multiscale_weights": [1, 1, 1, 1, 1],
                    },
                },
                "clustering": {
                    "parameters": {
                        "oracle_num_speakers": False,
                        "max_num_speakers": 8,
                        "enhanced_count_thres": 80,
                        "max_rp_threshold": 0.25,
                        "sparse_search_volume": 30,
                    },
                },
                "msdd_model": {
                    "model_path": "diar_msdd_telephonic",
                    "parameters": {
                        "use_speaker_model_from_ckpt": True,
                        "infer_batch_size": 25,
                        "sigmoid_threshold": [0.7],
                        "seq_eval_mode": False,
                        "split_infer": True,
                        "diar_window_length": 50,
                        "overlap_infer_spk_limit": 5,
                    },
                },
            },
        })
        return config

    def _transcribe_chunk(self, chunk: AudioChunk) -> Tuple[List[Word], float]:
        """
        Transcribe a single audio chunk.

        Args:
            chunk: AudioChunk object

        Returns:
            Tuple of (words with timestamps, processing time)
        """
        start_time = time.time()
        
        try:
            model = self._load_parakeet()
        except Exception as e:
            logger.error(f"Failed to load Parakeet model: {e}")
            logger.warning("Falling back to faster-whisper for chunk transcription")
            return self._transcribe_chunk_fallback(chunk)

        try:
            # Transcribe with word timestamps
            # NeMo 2.x API uses positional audio argument instead of paths2audio_files
            transcription = model.transcribe(
                [str(chunk.path)],
                timestamps=True,
                return_hypotheses=True,
            )

            # Extract words with timestamps
            words = []
            # NeMo RNNT models return tuple (hypotheses, optional_beam_hypotheses)
            if isinstance(transcription, tuple):
                transcription = transcription[0]
            
            if transcription and len(transcription) > 0:
                hypothesis = transcription[0]

                # Parakeet returns word-level timestamps in timestep attribute
                if hasattr(hypothesis, "timestep") and hypothesis.timestep:
                    # NeMo 2.x uses timestep dict with 'word' key for word-level timestamps
                    word_timestamps = hypothesis.timestep.get("word", [])
                    for word_info in word_timestamps:
                        word = Word(
                            text=word_info["word"],
                            start=word_info["start_offset"] + chunk.start_time,
                            end=word_info["end_offset"] + chunk.start_time,
                        )
                        words.append(word)
                elif hasattr(hypothesis, "words") and hypothesis.words:
                    for word_info in hypothesis.words:
                        # Adjust timestamps by chunk offset
                        word = Word(
                            text=word_info.word,
                            start=word_info.start_offset + chunk.start_time,
                            end=word_info.end_offset + chunk.start_time,
                        )
                        words.append(word)
                elif hasattr(hypothesis, "text"):
                    # Fallback: no word timestamps, create single segment
                    logger.warning(f"No word timestamps for chunk {chunk.chunk_index}")
                    words = [
                        Word(
                            text=hypothesis.text,
                            start=chunk.start_time,
                            end=chunk.end_time,
                        )
                    ]

            processing_time = time.time() - start_time
            logger.info(
                f"Chunk {chunk.chunk_index}: transcribed {len(words)} words "
                f"in {processing_time:.2f}s"
            )

            return words, processing_time

        except Exception as e:
            logger.error(f"Transcription failed for chunk {chunk.chunk_index}: {e}")
            return [], time.time() - start_time

    def _diarize_chunk(self, chunk: AudioChunk) -> Tuple[str, float]:
        """
        Perform speaker diarization on a single audio chunk.

        Args:
            chunk: AudioChunk object

        Returns:
            Tuple of (RTTM content string, processing time)
        """
        start_time = time.time()

        try:
            from nemo.collections.asr.models import NeuralDiarizer

            # Create temp directory for diarization output
            with tempfile.TemporaryDirectory(prefix="diarization_") as temp_dir:
                temp_dir_path = Path(temp_dir)
                manifest_path = temp_dir_path / "manifest.json"
                output_dir = temp_dir_path / "output"
                output_dir.mkdir()

                # Create manifest file (required by NeMo diarization)
                # NeMo uses the audio filename (stem) for output RTTM naming
                import json

                manifest = {
                    "audio_filepath": str(chunk.path),
                    "offset": 0,  # Offset within the chunk file itself
                    "duration": chunk.duration,
                    "label": "infer",
                    "uniq_id": f"chunk_{chunk.chunk_index:03d}",
                }
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f)
                    f.write("\n")

                # Get diarization config
                config = self._get_diarization_config(output_dir, manifest_path)

                # Create and run NeuralDiarizer
                logger.info(f"Running NeuralDiarizer on chunk {chunk.chunk_index}...")
                diarizer = NeuralDiarizer(cfg=config).to(self.device)
                diarizer.diarize()

                # Find RTTM output - NeMo names it based on audio file stem
                audio_stem = chunk.path.stem  # e.g., "chunk_000"
                rttm_path = output_dir / "pred_rttms" / f"{audio_stem}.rttm"

                # Also check alternative paths NeMo might use
                if not rttm_path.exists():
                    rttm_path = output_dir / f"{audio_stem}.rttm"
                if not rttm_path.exists():
                    # Search for any .rttm file
                    rttm_files = list(output_dir.rglob("*.rttm"))
                    if rttm_files:
                        rttm_path = rttm_files[0]
                        logger.debug(f"Found RTTM at: {rttm_path}")

                if rttm_path.exists():
                    rttm_content = rttm_path.read_text()

                    # Adjust timestamps by chunk offset (chunk's position in original audio)
                    adjusted_lines = []
                    for line in rttm_content.strip().split("\n"):
                        if line.startswith("SPEAKER"):
                            parts = line.split()
                            if len(parts) >= 4:
                                parts[3] = str(float(parts[3]) + chunk.start_time)
                                adjusted_lines.append(" ".join(parts))
                    rttm_content = "\n".join(adjusted_lines)
                else:
                    logger.warning(
                        f"No RTTM output for chunk {chunk.chunk_index}. "
                        f"Searched in {output_dir}"
                    )
                    rttm_content = ""

            processing_time = time.time() - start_time
            logger.info(f"Chunk {chunk.chunk_index}: diarized in {processing_time:.2f}s")

            return rttm_content, processing_time

        except Exception as e:
            logger.error(f"Diarization failed for chunk {chunk.chunk_index}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return "", time.time() - start_time

    def _process_chunk_parallel(
        self, chunk: AudioChunk
    ) -> Tuple[List[Word], str, float, float]:
        """
        Process a chunk with transcription and diarization.
        
        NOTE: Running these in parallel can cause deadlocks with NeMo's module loading.
        Processing sequentially is safer and still fast enough.

        Args:
            chunk: AudioChunk object

        Returns:
            Tuple of (words, rttm_content, transcription_time, diarization_time)
        """
        logger.info(f"Processing {chunk}")

        # Transcribe first
        words, trans_time = self._transcribe_chunk(chunk)
        
        # Then diarize (sequential to avoid deadlock)
        if self.enable_diarization:
            rttm_content, diar_time = self._diarize_chunk(chunk)
        else:
            rttm_content = ""
            diar_time = 0.0

        return words, rttm_content, trans_time, diar_time

    def transcribe(
        self,
        audio_path: Path,
        episode: Optional[PodcastEpisode] = None,
    ) -> TranscriptionResult:
        """
        Transcribe podcast episode with speaker diarization.

        Args:
            audio_path: Path to audio file
            episode: Optional episode metadata

        Returns:
            TranscriptionResult with speaker-labeled segments
        """
        start_time = time.time()
        audio_path = Path(audio_path)

        logger.info(f"Starting NeMo transcription: {audio_path.name}")

        # Create temp directory for chunks
        temp_dir = Path(tempfile.mkdtemp(prefix="nemo_chunks_"))

        try:
            # Step 1: Chunk audio
            chunks = list(chunk_audio(audio_path, self.chunk_duration_minutes, temp_dir))
            logger.info(f"Created {len(chunks)} audio chunks")

            # Step 2: Process chunks (transcription + diarization in parallel)
            all_words = []
            all_rttm_lines = []
            total_trans_time = 0.0
            total_diar_time = 0.0

            for chunk in chunks:
                words, rttm_content, trans_time, diar_time = self._process_chunk_parallel(chunk)
                all_words.extend(words)
                if rttm_content:
                    all_rttm_lines.append(rttm_content)
                total_trans_time += trans_time
                total_diar_time += diar_time

            logger.info(f"Transcribed {len(all_words)} total words")
            logger.info(f"Total transcription time: {total_trans_time:.2f}s")
            logger.info(f"Total diarization time: {total_diar_time:.2f}s")

            # Step 3: Align words with speakers
            if self.enable_diarization and all_rttm_lines:
                combined_rttm = "\n".join(all_rttm_lines)
                aligned_segments = align_transcription_with_diarization(
                    all_words, combined_rttm
                )

                # Convert to TranscriptSegment objects
                transcript_segments = []
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
            else:
                # No diarization - create segments from words
                transcript_segments = [
                    TranscriptSegment(
                        start=all_words[0].start if all_words else 0.0,
                        end=all_words[-1].end if all_words else 0.0,
                        text=" ".join(w.text for w in all_words),
                        speaker=None,
                    )
                ]
                speakers_detected = 1

            # Format full transcript
            full_text = self._format_transcript(transcript_segments)

            # Calculate metrics
            duration = chunks[-1].end_time if chunks else 0.0
            processing_time = time.time() - start_time

            logger.info(
                f"Transcription complete in {processing_time:.1f}s "
                f"({processing_time/duration:.2f}x realtime)"
            )

            return TranscriptionResult(
                episode_id=episode.id if episode else "unknown",
                segments=transcript_segments,
                full_text=full_text,
                language="en",
                speakers_detected=speakers_detected,
                duration_seconds=duration,
                processing_time_seconds=processing_time,
            )

        finally:
            # Clean up temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temp directory: {temp_dir}")

    def _format_transcript(self, segments: List[TranscriptSegment]) -> str:
        """Format segments into readable transcript."""
        lines = []
        for seg in segments:
            timestamp = f"[{seg.start:.1f}s]"
            speaker = f"{seg.speaker}: " if seg.speaker else ""
            lines.append(f"{timestamp} {speaker}{seg.text}")

        return "\n".join(lines)
