"""
Transcription Job Handler for NeMo Container

Reads job requests from JSON files and writes results.
This runs inside the Docker container.
"""

import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_transcription_job(request_path: str) -> None:
    """Run a transcription job from a request file."""
    request_file = Path(request_path)
    
    if not request_file.exists():
        logger.error(f"Request file not found: {request_file}")
        sys.exit(1)
    
    # Parse request
    request = json.loads(request_file.read_text())
    job_id = request["job_id"]
    audio_file = Path(request["audio_file"])
    enable_diarization = request.get("enable_diarization", True)
    chunk_duration_minutes = request.get("chunk_duration_minutes", 10)
    
    logger.info(f"Starting job {job_id}: {audio_file.name}")
    
    if not audio_file.exists():
        logger.error(f"Audio file not found: {audio_file}")
        write_error_result(job_id, f"Audio file not found: {audio_file}")
        sys.exit(1)
    
    start_time = time.time()
    
    try:
        # Import NeMo transcriber (container has NeMo installed)
        from nemo_transcriber import NeMoTranscriber
        
        transcriber = NeMoTranscriber(
            device="cuda",
            chunk_duration_minutes=chunk_duration_minutes,
            enable_diarization=enable_diarization,
        )
        
        result = transcriber.transcribe(audio_file)
        
        # Serialize result to JSON
        result_data = {
            "job_id": job_id,
            "status": "completed",
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "speaker": seg.speaker,
                }
                for seg in result.segments
            ],
            "full_text": result.full_text,
            "language": result.language,
            "speakers_detected": result.speakers_detected,
            "duration_seconds": result.duration_seconds,
            "processing_time_seconds": time.time() - start_time,
        }
        
        # Write result
        output_dir = Path("/data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = output_dir / f"{job_id}_result.json"
        result_file.write_text(json.dumps(result_data, indent=2))
        
        logger.info(f"Job {job_id} completed in {result_data['processing_time_seconds']:.1f}s")
        
    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        write_error_result(job_id, str(e))
        sys.exit(1)


def write_error_result(job_id: str, error: str) -> None:
    """Write an error result file."""
    output_dir = Path("/data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / f"{job_id}_result.json"
    result_file.write_text(json.dumps({
        "job_id": job_id,
        "status": "failed",
        "error": error,
    }))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m podcast.transcribe_job <request.json>")
        sys.exit(1)
    
    run_transcription_job(sys.argv[1])
