# Zettlecast setup for Windows + NVIDIA GPU (RTX 50-series supported)
#
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#
# Safe to re-run: installs only what is missing.

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Have($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }
function Refresh-Path {
    $env:PATH = [Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("PATH", "User") + ";" +
                "$env:LOCALAPPDATA\Programs\Ollama"
}
function Winget-Install($id, $cmd) {
    if (Have $cmd) { Write-Host "    $cmd found"; return }
    Write-Host "    installing $id"
    winget install --id $id -e --silent --accept-source-agreements --accept-package-agreements | Out-Null
    Refresh-Path
    if (-not (Have $cmd)) { throw "$id installed but '$cmd' is not on PATH; open a new terminal and re-run setup.ps1" }
}

Step "Checking NVIDIA GPU"
if (-not (Have "nvidia-smi")) { throw "nvidia-smi not found. Install the NVIDIA driver first." }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

Step "Installing tools (uv, FFmpeg, Node.js, Ollama)"
Refresh-Path
Winget-Install "astral-sh.uv" "uv"
Winget-Install "Gyan.FFmpeg" "ffmpeg"
Winget-Install "OpenJS.NodeJS.LTS" "npm"
Winget-Install "Ollama.Ollama" "ollama"

Step "Creating Python 3.12 environment (.venv)"
if (-not (Test-Path ".venv\Scripts\python.exe")) { uv venv .venv --python 3.12 }
$py = Join-Path $Root ".venv\Scripts\python.exe"

Step "Installing Python packages (PyTorch CUDA 13, NeMo, ...) - first run downloads several GB"
uv pip install --python $py -e ".[dev]" `
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
if ($LASTEXITCODE -ne 0) { throw "Python package install failed" }
& $py -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('    torch', torch.__version__, '-', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "PyTorch cannot see the GPU" }

Step "Writing configuration"
if (-not (Test-Path ".env")) {
    $token = & $py -c "import secrets; print(secrets.token_urlsafe(32))"
    $storage = (Join-Path $Root "_BRAIN_STORAGE") -replace "\\", "/"
    @(
        "# Zettlecast configuration (all options: src/zettlecast/config.py)",
        "API_TOKEN=$token",
        "HF_TOKEN=",
        "",
        "STORAGE_PATH=$storage",
        "LANCEDB_PATH=$storage/.lancedb"
    ) | Set-Content -Encoding utf8 .env
    Write-Host "    created .env"
} else { Write-Host "    .env exists" }
$token = (Select-String -Path .env -Pattern "^API_TOKEN=(.*)$").Matches[0].Groups[1].Value
@(
    "NEXT_PUBLIC_API_URL=http://localhost:8000",
    "NEXT_PUBLIC_API_TOKEN=$token"
) | Set-Content -Encoding utf8 frontend\.env.local

Step "Installing frontend packages"
Push-Location frontend
npm install --no-fund --no-audit
if ($LASTEXITCODE -ne 0) { Pop-Location; throw "npm install failed" }
Pop-Location

Step "Pulling the LLM (Ollama)"
if (-not (Get-Process ollama -ErrorAction SilentlyContinue)) {
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep 3
}
$model = & $py -c "from zettlecast.config import settings; print(settings.ollama_model)"
ollama pull $model

Step "Downloading speech, embedding and reranker models (Hugging Face)"
& $py -c @"
import zettlecast
from zettlecast.config import settings
from huggingface_hub import snapshot_download
for repo in (settings.nemo_parakeet_model, settings.nemo_diarization_model, settings.embedding_model, settings.reranker_model):
    print('   ', repo); snapshot_download(repo)
"@

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "  Start everything:   .\start.ps1"
Write-Host "  CLI:                .\zc.ps1 --help"
