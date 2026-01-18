"""
Transcriber Factory

Platform-aware factory that creates the appropriate transcriber backend
based on the current system and configuration.
"""

import logging
import platform
import sys
from typing import Optional

from ..config import settings
from .base_transcriber import BaseTranscriber, TranscriberConfig

logger = logging.getLogger(__name__)


def detect_platform() -> str:
    """
    Detect the current platform and GPU availability.
    
    Returns:
        Platform identifier: 'darwin', 'win32+cuda', 'linux+cuda', or 'cpu'
    """
    system = platform.system().lower()
    
    if system == "darwin":
        # macOS - check for Apple Silicon
        if platform.machine() == "arm64":
            logger.info("Detected macOS Apple Silicon (M-series)")
            return "darwin"
        else:
            logger.info("Detected macOS Intel - using CPU backend")
            return "cpu"
    
    # Check for CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Detected CUDA GPU: {gpu_name}")
            if system == "windows":
                return "win32+cuda"
            else:
                return "linux+cuda"
    except ImportError:
        pass
    
    logger.info(f"No GPU detected, using CPU backend on {system}")
    return "cpu"


class TranscriberFactory:
    """
    Factory for creating platform-appropriate transcriber instances.
    
    Automatically selects the best backend based on:
    - Platform (macOS, Windows, Linux)
    - GPU availability (CUDA, Apple Silicon)
    - Configuration preferences
    
    Usage:
        transcriber = TranscriberFactory.create()
        result = transcriber.transcribe(audio_path)
    """
    
    _platform_cache: Optional[str] = None
    
    @classmethod
    def get_platform(cls) -> str:
        """Get cached platform detection result."""
        if cls._platform_cache is None:
            cls._platform_cache = detect_platform()
        return cls._platform_cache
    
    @classmethod
    def create(
        cls,
        config: Optional[TranscriberConfig] = None,
        backend: Optional[str] = None,
    ) -> BaseTranscriber:
        """
        Create a transcriber instance for the current platform.
        
        Args:
            config: Optional transcriber configuration
            backend: Force a specific backend ('nemo', 'parakeet-mlx', 'mlx-whisper', 'whisper')
                    If None, auto-selects based on platform
                    
        Returns:
            BaseTranscriber instance appropriate for the platform
            
        Raises:
            RuntimeError: If no suitable backend is available
        """
        config = config or TranscriberConfig()
        
        # Apply HF token from settings if not in config
        if not config.hf_token:
            config.hf_token = settings.hf_token
        
        # Determine backend
        if backend and backend != "auto":
            return cls._create_backend(backend, config)
        
        # Check settings for preferred backend
        preferred_backend = getattr(settings, 'asr_backend', 'auto')
        if preferred_backend and preferred_backend != "auto":
            return cls._create_backend(preferred_backend, config)
        
        # Auto-select based on platform
        platform_id = cls.get_platform()
        
        if platform_id == "darwin":
            return cls._create_mac_transcriber(config)
        elif platform_id in ("win32+cuda", "linux+cuda"):
            return cls._create_nemo_transcriber(config)
        else:
            return cls._create_whisper_transcriber(config)
    
    @classmethod
    def _create_backend(cls, backend: str, config: TranscriberConfig) -> BaseTranscriber:
        """Create a specific backend by name."""
        backend = backend.lower().replace("-", "_")
        
        if backend in ("nemo", "nemo_container"):
            return cls._create_nemo_transcriber(config)
        elif backend in ("parakeet_mlx", "parakeet", "mac"):
            return cls._create_mac_transcriber(config)
        elif backend in ("mlx_whisper", "mlx"):
            return cls._create_mlx_whisper_transcriber(config)
        elif backend in ("whisper", "faster_whisper"):
            return cls._create_whisper_transcriber(config)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    @classmethod
    def _create_mac_transcriber(cls, config: TranscriberConfig) -> BaseTranscriber:
        """Create Mac transcriber with parakeet-mlx + pyannote."""
        try:
            from .mac_transcriber import MacTranscriber
            transcriber = MacTranscriber(config)
            if transcriber.is_available():
                logger.info("Using MacTranscriber (parakeet-mlx + pyannote)")
                return transcriber
        except ImportError as e:
            logger.warning(f"MacTranscriber not available: {e}")
        
        # Fallback to Whisper
        logger.info("Falling back to WhisperTranscriber")
        return cls._create_whisper_transcriber(config)
    
    @classmethod
    def _create_nemo_transcriber(cls, config: TranscriberConfig) -> BaseTranscriber:
        """Create NeMo transcriber (container or direct)."""
        # Try container-based transcriber first
        try:
            from .container_transcriber import NeMoContainerTranscriber
            transcriber = NeMoContainerTranscriber(config)
            if transcriber.is_available():
                logger.info("Using NeMoContainerTranscriber (Docker)")
                return transcriber
        except ImportError as e:
            logger.debug(f"NeMoContainerTranscriber not available: {e}")
        
        # Try direct NeMo (if installed locally)
        try:
            from .nemo_transcriber import NeMoTranscriber
            transcriber = NeMoTranscriber(
                device=config.device if config.device != "auto" else "cuda",
                chunk_duration_minutes=config.chunk_duration_minutes,
                enable_diarization=config.enable_diarization,
            )
            logger.info("Using NeMoTranscriber (direct)")
            return transcriber
        except ImportError as e:
            logger.warning(f"NeMoTranscriber not available: {e}")
        
        # Fallback to Whisper
        logger.info("Falling back to WhisperTranscriber")
        return cls._create_whisper_transcriber(config)
    
    @classmethod
    def _create_mlx_whisper_transcriber(cls, config: TranscriberConfig) -> BaseTranscriber:
        """Create MLX Whisper transcriber for Apple Silicon."""
        try:
            from .mlx_whisper_transcriber import MLXWhisperTranscriber
            transcriber = MLXWhisperTranscriber(config)
            if transcriber.is_available():
                logger.info("Using MLXWhisperTranscriber (mlx-whisper)")
                return transcriber
        except ImportError as e:
            logger.warning(f"MLXWhisperTranscriber not available: {e}")
        
        # Fallback to faster-whisper
        logger.info("Falling back to WhisperTranscriber")
        return cls._create_whisper_transcriber(config)
    
    @classmethod
    def _create_whisper_transcriber(cls, config: TranscriberConfig) -> BaseTranscriber:
        """Create Whisper-based transcriber as fallback."""
        from .whisper_transcriber import WhisperTranscriber
        logger.info("Using WhisperTranscriber (faster-whisper)")
        return WhisperTranscriber(config)
    
    @classmethod
    def list_available_backends(cls) -> list[dict]:
        """
        List all available backends on the current system.
        
        Returns:
            List of dicts with backend info and availability status
        """
        backends = []
        platform_id = cls.get_platform()
        
        # Check Mac backend
        try:
            from .mac_transcriber import MacTranscriber
            backends.append({
                "name": "parakeet-mlx",
                "available": platform_id == "darwin",
                "platform": "macOS (Apple Silicon)",
                "diarization": "pyannote.audio",
            })
        except ImportError:
            backends.append({
                "name": "parakeet-mlx",
                "available": False,
                "reason": "parakeet-mlx not installed",
            })
        
        # Check NeMo container backend
        try:
            from .container_transcriber import NeMoContainerTranscriber
            backends.append({
                "name": "nemo-container",
                "available": platform_id in ("win32+cuda", "linux+cuda"),
                "platform": "Windows/Linux (CUDA)",
                "diarization": "NeMo MSDD",
            })
        except ImportError:
            backends.append({
                "name": "nemo-container",
                "available": False,
                "reason": "docker not installed",
            })
        
        # Check MLX Whisper backend
        try:
            from .mlx_whisper_transcriber import MLXWhisperTranscriber
            backends.append({
                "name": "mlx-whisper",
                "available": platform_id == "darwin",
                "platform": "macOS (Apple Silicon)",
                "diarization": "None (transcription only)",
            })
        except ImportError:
            backends.append({
                "name": "mlx-whisper",
                "available": False,
                "reason": "mlx-whisper not installed",
            })
        
        # Whisper is always available as fallback
        backends.append({
            "name": "whisper",
            "available": True,
            "platform": "All platforms",
            "diarization": "None (transcription only)",
        })
        
        return backends
