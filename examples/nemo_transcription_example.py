"""
Example: Using the NeMo Parallel Transcription Pipeline

This demonstrates the new architecture with:
- Parakeet-TDT for fast transcription (~60 min in <2 sec)
- MSDD for speaker diarization (parallel execution)
- 10-minute macro-chunking for stable memory
"""

from pathlib import Path
from zettlecast.podcast import NeMoTranscriber

# Example 1: Basic transcription with diarization
def basic_example():
    """Simple transcription with speaker diarization."""
    transcriber = NeMoTranscriber(
        device="cuda",  # Use "cpu" if no GPU available
        chunk_duration_minutes=10,
        enable_diarization=True,
    )

    audio_path = Path("path/to/podcast.mp3")
    result = transcriber.transcribe(audio_path)

    print(f"Duration: {result.duration_seconds:.1f}s")
    print(f"Processing time: {result.processing_time_seconds:.1f}s")
    print(f"Speakers detected: {result.speakers_detected}")
    print(f"\nTranscript:\n{result.full_text}")


# Example 2: Transcription only (no diarization)
def transcription_only_example():
    """Fast transcription without speaker identification."""
    transcriber = NeMoTranscriber(
        device="cuda",
        enable_diarization=False,  # Disable for faster processing
    )

    audio_path = Path("path/to/podcast.mp3")
    result = transcriber.transcribe(audio_path)

    print(f"Transcript:\n{result.full_text}")


# Example 3: Custom chunk size for very long podcasts
def custom_chunk_example():
    """Use larger chunks for very long podcasts (e.g., 3+ hours)."""
    transcriber = NeMoTranscriber(
        device="cuda",
        chunk_duration_minutes=15,  # Larger chunks for long files
        enable_diarization=True,
    )

    audio_path = Path("path/to/long_podcast.mp3")
    result = transcriber.transcribe(audio_path)

    # Access individual segments with speaker labels
    for segment in result.segments:
        print(f"[{segment.start:.1f}s] {segment.speaker}: {segment.text}")


# Example 4: Processing segments for further analysis
def segment_processing_example():
    """Extract and process individual speaker segments."""
    transcriber = NeMoTranscriber(device="cuda")
    audio_path = Path("path/to/podcast.mp3")
    result = transcriber.transcribe(audio_path)

    # Group text by speaker
    by_speaker = {}
    for segment in result.segments:
        speaker = segment.speaker or "Unknown"
        if speaker not in by_speaker:
            by_speaker[speaker] = []
        by_speaker[speaker].append(segment.text)

    # Print word counts per speaker
    for speaker, texts in by_speaker.items():
        word_count = sum(len(text.split()) for text in texts)
        print(f"{speaker}: {word_count} words")


if __name__ == "__main__":
    # Run the basic example
    basic_example()
