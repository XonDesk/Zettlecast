"""Tests for the LLM client's llama-server backend (HTTP and processes mocked)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from zettlecast import llm
from zettlecast.config import settings


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A fake Ollama store holding qwen3.6:35b-a3b (model + projector)."""
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "sha256-aaa").write_bytes(b"gguf")
    (blobs / "sha256-bbb").write_bytes(b"proj")
    manifest = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "qwen3.6" / "35b-a3b"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"layers": [
        {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:aaa"},
        {"mediaType": "application/vnd.ollama.image.projector", "digest": "sha256:bbb"},
    ]}))
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    return blobs


@pytest.fixture
def server_backend(tmp_path, monkeypatch):
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(settings, "llm_backend", "llama-server")
    monkeypatch.setattr(settings, "llama_server_exe", str(exe))
    monkeypatch.setattr(settings, "ollama_model", "qwen3.6:35b-a3b")
    monkeypatch.setattr(settings, "storage_path", tmp_path / "storage")
    return exe


def reply(content):
    return MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": content}}]})


def test_finds_model_and_projector_in_ollama_store(store):
    assert llm.ollama_blob("qwen3.6:35b-a3b", "model") == store / "sha256-aaa"
    assert llm.ollama_blob("qwen3.6:35b-a3b", "projector") == store / "sha256-bbb"
    assert llm.ollama_blob("missing:tag", "model") is None


def test_schema_and_images_use_openai_format(server_backend, tmp_path):
    image = tmp_path / "shot.png"
    image.write_bytes(b"png")
    with patch.object(llm, "_ensure_server"), \
            patch("zettlecast.llm.httpx.post", return_value=reply('{"a": 1}')) as post:
        out = llm.chat("Describe", schema={"type": "object"}, images=[image])
    assert out == {"a": 1}
    body = post.call_args.kwargs["json"]
    assert post.call_args.args[0].endswith("/v1/chat/completions")
    assert body["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    parts = body["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "Describe"}
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_other_model_or_missing_exe_goes_to_ollama(server_backend, monkeypatch):
    ollama = MagicMock(status_code=200, json=lambda: {"message": {"content": "hi"}})
    with patch.object(llm, "_ensure_server") as ensure, \
            patch("zettlecast.llm.httpx.post", return_value=ollama) as post:
        assert llm.chat("x", model="llava:7b") == "hi"
        monkeypatch.setattr(settings, "llama_server_exe", str(server_backend.parent / "nope.exe"))
        assert llm.chat("x") == "hi"
    ensure.assert_not_called()
    assert all(c.args[0].endswith("/api/chat") for c in post.call_args_list)


def test_starts_text_server_from_store_with_shared_slots(server_backend, store, monkeypatch):
    monkeypatch.setattr(settings, "llm_parallel", 4)
    monkeypatch.setattr(settings, "llama_server_ctx", 65536)
    monkeypatch.setattr(settings, "llm_idle_minutes", 0)
    health = iter([None, 503, 200])
    process = MagicMock(pid=4242, poll=lambda: None)
    with patch.object(llm, "_health", side_effect=lambda: next(health)), \
            patch("zettlecast.llm.subprocess.Popen", return_value=process) as popen, \
            patch("zettlecast.llm.time.sleep"):
        llm._ensure_server()
    cmd = popen.call_args.args[0]
    assert cmd[cmd.index("-m") + 1] == str(store / "sha256-aaa")
    assert "--mmproj" not in cmd  # text runs keep the ~1 GB projector off the GPU
    assert cmd[cmd.index("-np") + 1] == "4" and cmd[cmd.index("-c") + 1] == "65536" and "--kv-unified" in cmd
    assert "--spec-type" in cmd
    assert (settings.storage_path / ".llama-server.pid").read_text() == "4242"


def test_image_call_restarts_text_server_with_projector(server_backend, store, monkeypatch):
    monkeypatch.setattr(settings, "llm_idle_minutes", 0)
    settings.storage_path.mkdir(parents=True)
    (settings.storage_path / ".llama-server.pid").write_text("4242")  # running, text only
    health = iter([200, 200])
    with patch.object(llm, "_health", side_effect=lambda: next(health, 200)), \
            patch.object(llm, "stop_server") as stop, \
            patch("zettlecast.llm.subprocess.Popen", return_value=MagicMock(pid=77, poll=lambda: None)) as popen:
        llm._ensure_server(vision=True)
    stop.assert_called_once()
    cmd = popen.call_args.args[0]
    assert cmd[cmd.index("--mmproj") + 1] == str(store / "sha256-bbb")
    assert (settings.storage_path / ".llama-server.pid").read_text() == "77 vision"


def test_idle_and_exit_stop_only_own_server(server_backend, monkeypatch):
    settings.storage_path.mkdir(parents=True)
    (settings.storage_path / ".llama-server.pid").write_text("999")
    monkeypatch.setattr(llm, "_own_pid", 4242)
    with patch("zettlecast.llm.subprocess.run") as run, patch.object(llm, "_health", return_value=None):
        llm.stop_server(only_own=True)
        run.assert_not_called()
        llm.stop_server()  # a run freeing the GPU stops it whoever started it
        assert "999" in run.call_args.args[0]
    assert not (settings.storage_path / ".llama-server.pid").exists()


def test_restarts_server_once_when_it_vanished(server_backend):
    import httpx

    calls = [httpx.ConnectError("gone"), reply("ok")]

    def post(*_a, **_k):
        r = calls.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    with patch.object(llm, "_ensure_server") as ensure, patch("zettlecast.llm.httpx.post", side_effect=post):
        assert llm.chat("x") == "ok"
    assert ensure.call_count == 2
