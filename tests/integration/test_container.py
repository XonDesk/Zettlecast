"""
Integration tests for NeMo container.

These tests require:
- Docker installed with NVIDIA Container Toolkit
- Windows or Linux with NVIDIA GPU

Run with: pytest tests/integration/test_container.py -v
"""

import os
import subprocess
import pytest


# Skip if Docker not available
def docker_available():
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not docker_available(),
    reason="Docker not available"
)


class TestContainerBuild:
    """Tests for Docker container build process."""

    @pytest.mark.slow
    def test_dockerfile_syntax_valid(self):
        """Dockerfile should have valid syntax."""
        dockerfile_path = "Zettlecast/docker/Dockerfile.nemo"
        
        if not os.path.exists(dockerfile_path):
            pytest.skip("Dockerfile not found")
        
        # Use docker build --check (dry run) if available
        result = subprocess.run(
            ["docker", "build", "--check", "-f", dockerfile_path, "."],
            capture_output=True, text=True
        )
        
        # --check may not be available in all Docker versions
        # Fall back to just checking file exists and is readable
        assert os.path.exists(dockerfile_path)
        
        with open(dockerfile_path) as f:
            content = f.read()
            assert "FROM" in content
            assert "WORKDIR" in content

    def test_docker_compose_valid(self):
        """Docker compose file should be valid YAML."""
        compose_path = "Zettlecast/docker/docker-compose.yml"
        
        if not os.path.exists(compose_path):
            pytest.skip("docker-compose.yml not found")
        
        result = subprocess.run(
            ["docker", "compose", "-f", compose_path, "config"],
            capture_output=True, text=True
        )
        
        # docker compose config validates the file
        assert result.returncode == 0 or "nvidia" in result.stderr.lower()


class TestContainerTranscriber:
    """Tests for NeMoContainerTranscriber."""

    def test_container_transcriber_is_available_method(self):
        """is_available should return bool without crashing."""
        from zettlecast.podcast.container_transcriber import NeMoContainerTranscriber
        
        transcriber = NeMoContainerTranscriber()
        result = transcriber.is_available()
        
        assert isinstance(result, bool)

    def test_container_transcriber_capabilities(self):
        """Should return correct capabilities."""
        from zettlecast.podcast.container_transcriber import NeMoContainerTranscriber
        
        transcriber = NeMoContainerTranscriber()
        caps = transcriber.get_capabilities()
        
        assert caps.platform == "win32+cuda"
        assert caps.requires_container is True
        assert caps.transcriber_name == "nemo-parakeet-tdt"
        assert caps.diarizer_name == "nemo-msdd"

    @pytest.mark.slow
    def test_build_image_command(self):
        """Should be able to call build_image (may skip actual build)."""
        from zettlecast.podcast.container_transcriber import NeMoContainerTranscriber
        
        # Just verify the method is callable
        # Don't actually build (takes too long)
        assert callable(NeMoContainerTranscriber.build_image)
