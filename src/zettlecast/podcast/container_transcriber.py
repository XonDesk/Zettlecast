"""
NeMo Container Transcriber

Host-side class that manages Docker container lifecycle for NeMo-based
transcription on Windows with NVIDIA GPU.
"""

import json
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from .base_transcriber import BaseTranscriber, TranscriberCapabilities, TranscriberConfig
from .models import PodcastEpisode, TranscriptSegment, TranscriptionResult

logger = logging.getLogger(__name__)


class NeMoContainerTranscriber(BaseTranscriber):
    """
    NeMo transcriber that runs inside a Docker container.
    
    Uses NVIDIA Container Toolkit for GPU passthrough.
    Communicates with container via volume mounts and JSON files.
    """
    
    CONTAINER_NAME = "zettlecast-nemo"
    DEFAULT_IMAGE = "zettlecast/nemo-asr:latest"
    
    def __init__(self, config: Optional[TranscriberConfig] = None):
        super().__init__(config)
        
        self.image = getattr(
            self.config, 'nemo_container_image', None
        ) or self.DEFAULT_IMAGE
        
        self.auto_start = getattr(
            self.config, 'nemo_container_auto_start', True
        )
        
        # Temp directory for communication with container
        self._temp_dir = None
    
    def is_available(self) -> bool:
        """Check if Docker is available and NVIDIA runtime is present."""
        try:
            # Check Docker is running
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.debug("Docker not running")
                return False
            
            # Check for NVIDIA runtime
            result = subprocess.run(
                ["docker", "info", "--format", "{{json .Runtimes}}"],
                capture_output=True,
                timeout=10,
                text=True,
            )
            if "nvidia" not in result.stdout.lower():
                logger.debug("NVIDIA runtime not found in Docker")
                return False
            
            return True
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_capabilities(self) -> TranscriberCapabilities:
        """Return NeMo container capabilities."""
        return TranscriberCapabilities(
            platform="win32+cuda",
            transcriber_name="nemo-parakeet-tdt",
            diarizer_name="nemo-msdd",
            vad_name="marblenet",
            supports_diarization=True,
            supports_gpu=True,
            supports_streaming=False,
            requires_container=True,
        )
    
    def _ensure_container_running(self) -> bool:
        """Ensure the NeMo container is running."""
        # Check if already running
        result = subprocess.run(
            ["docker", "ps", "-q", "-f", f"name={self.CONTAINER_NAME}"],
            capture_output=True,
            text=True,
        )
        
        if result.stdout.strip():
            logger.debug("NeMo container already running")
            return True
        
        if not self.auto_start:
            logger.error(
                f"NeMo container not running. Start it with: "
                f"docker start {self.CONTAINER_NAME}"
            )
            return False
        
        # Try to start existing container
        result = subprocess.run(
            ["docker", "start", self.CONTAINER_NAME],
            capture_output=True,
        )
        
        if result.returncode == 0:
            logger.info("Started existing NeMo container")
            time.sleep(2)  # Give container time to initialize
            return True
        
        # Container doesn't exist - need to create it
        logger.info(f"Creating NeMo container from image: {self.image}")
        
        # Create temp directory for volume mount
        self._temp_dir = Path(tempfile.mkdtemp(prefix="zettlecast_nemo_"))
        
        result = subprocess.run([
            "docker", "run", "-d",
            "--name", self.CONTAINER_NAME,
            "--gpus", "all",
            "-v", f"{self._temp_dir}:/data",
            self.image,
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Failed to create container: {result.stderr}")
            return False
        
        logger.info("NeMo container started successfully")
        time.sleep(5)  # Give container time to initialize
        return True
    
    def _get_temp_dir(self) -> Path:
        """Get or create temp directory for container communication."""
        if self._temp_dir is None or not self._temp_dir.exists():
            self._temp_dir = Path(tempfile.mkdtemp(prefix="zettlecast_nemo_"))
        return self._temp_dir
    
    def transcribe(
        self,
        audio_path: Path,
        episode: Optional[PodcastEpisode] = None,
        progress_callback=None,
    ) -> TranscriptionResult:
        """
        Transcribe audio using NeMo in Docker container.
        
        Args:
            audio_path: Path to audio file
            episode: Optional episode metadata
            
        Returns:
            TranscriptionResult with segments and metadata
        """
        start_time = time.time()
        audio_path = Path(audio_path)
        
        logger.info(f"Starting NeMo container transcription: {audio_path.name}")
        
        # Ensure container is running
        if not self._ensure_container_running():
            raise RuntimeError("Failed to start NeMo container")
        
        temp_dir = self._get_temp_dir()
        
        # Copy audio file to shared volume
        audio_dest = temp_dir / "input" / audio_path.name
        audio_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audio_path, audio_dest)
        
        # Create job request
        job_id = f"job_{int(time.time())}"
        request = {
            "job_id": job_id,
            "audio_file": f"/data/input/{audio_path.name}",
            "enable_diarization": self.config.enable_diarization,
            "chunk_duration_minutes": self.config.chunk_duration_minutes,
            "max_speakers": self.config.max_speakers,
        }
        
        request_file = temp_dir / "requests" / f"{job_id}.json"
        request_file.parent.mkdir(parents=True, exist_ok=True)
        request_file.write_text(json.dumps(request))
        
        # Execute transcription in container
        result = subprocess.run([
            "docker", "exec", self.CONTAINER_NAME,
            "python", "-m", "podcast.transcribe_job",
            f"/data/requests/{job_id}.json",
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode != 0:
            logger.error(f"Container transcription failed: {result.stderr}")
            raise RuntimeError(f"Transcription failed: {result.stderr}")
        
        # Read result from container output
        result_file = temp_dir / "output" / f"{job_id}_result.json"
        
        if not result_file.exists():
            raise RuntimeError("Transcription result not found")
        
        result_data = json.loads(result_file.read_text())
        
        # Parse result into TranscriptionResult
        transcript_segments = [
            TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                speaker=seg.get("speaker"),
            )
            for seg in result_data["segments"]
        ]
        
        processing_time = time.time() - start_time
        
        # Cleanup temp files
        try:
            audio_dest.unlink()
            request_file.unlink()
            result_file.unlink()
        except Exception:
            pass
        
        logger.info(
            f"Transcription complete in {processing_time:.1f}s "
            f"({processing_time/result_data['duration_seconds']:.2f}x realtime)"
        )
        
        return TranscriptionResult(
            episode_id=episode.id if episode else "unknown",
            segments=transcript_segments,
            full_text=result_data["full_text"],
            language=result_data.get("language", "en"),
            speakers_detected=result_data.get("speakers_detected", 1),
            duration_seconds=result_data["duration_seconds"],
            processing_time_seconds=processing_time,
        )
    
    def stop_container(self) -> bool:
        """Stop the NeMo container."""
        result = subprocess.run(
            ["docker", "stop", self.CONTAINER_NAME],
            capture_output=True,
        )
        return result.returncode == 0
    
    def remove_container(self) -> bool:
        """Remove the NeMo container."""
        self.stop_container()
        result = subprocess.run(
            ["docker", "rm", self.CONTAINER_NAME],
            capture_output=True,
        )
        return result.returncode == 0
    
    @classmethod
    def build_image(cls, dockerfile_path: Optional[Path] = None) -> bool:
        """Build the NeMo Docker image."""
        dockerfile_path = dockerfile_path or Path("docker/Dockerfile.nemo")
        
        if not dockerfile_path.exists():
            logger.error(f"Dockerfile not found: {dockerfile_path}")
            return False
        
        logger.info("Building NeMo Docker image (this may take a while)...")
        
        result = subprocess.run([
            "docker", "build",
            "-t", cls.DEFAULT_IMAGE,
            "-f", str(dockerfile_path),
            ".",
        ], capture_output=False)  # Show build output
        
        return result.returncode == 0
