# Zettlecast Setup Script for Windows
# Installs all dependencies including Ollama

Write-Host "Zettlecast Setup" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan

# Detect Windows version
$OSInfo = Get-WmiObject -Class Win32_OperatingSystem
Write-Host "Detected: Windows $($OSInfo.Version) ($env:PROCESSOR_ARCHITECTURE)"

# Check for Python
Write-Host ""
Write-Host "[*] Checking Python installation..." -ForegroundColor Yellow

$pythonCmd = $null
$pythonVerArg = ""

# Try finding Python 3.11 specifically via py launcher first
try {
    & py -3.11 --version | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $pythonCmd = "py"
        $pythonVerArg = "-3.11"
        Write-Host "[OK] Using Python 3.11 via py launcher" -ForegroundColor Green
    }
} catch {}

if (-not $pythonCmd) {
    foreach ($cmd in @("python", "python3", "py")) {
        try {
            $version = & $cmd --version 2>&1
            if ($version -match "Python 3\.1[1-2]") { # Match 3.11 or 3.12 only
                $pythonCmd = $cmd
                $pythonVerArg = ""
                Write-Host "[OK] Found compatible Python: $version" -ForegroundColor Green
                break
            }
        } catch {}
    }
}

# Check for Ollama and add to PATH if needed
$ollamaPath = "$env:LOCALAPPDATA\Programs\Ollama"
if (Test-Path $ollamaPath) {
    $env:PATH += ";$ollamaPath"
    Write-Host "Added Ollama to PATH: $ollamaPath"
}

if (-not $pythonCmd) {
    Write-Host "[ERROR] Python 3.11 or 3.12 not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python 3.11 from:" -ForegroundColor Yellow
    Write-Host "  https://www.python.org/downloads/release/python-3119/"
    Write-Host ""
    Write-Host "[!] Important: Python 3.13+ is NOT supported due to NeMo/ML library compatibility."
    Write-Host "Make sure to check 'Add Python to PATH' during installation."
    exit 1
}

# --- Python Environment ---
Write-Host ""
Write-Host "[*] Setting up Python environment..." -ForegroundColor Yellow

if (-not (Test-Path ".venv")) {
    if ($pythonVerArg) {
        & $pythonCmd $pythonVerArg -m venv .venv
    } else {
        & $pythonCmd -m venv .venv
    }
    Write-Host "Created virtual environment"
}

# Activate virtual environment
$activateScript = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "Activated virtual environment"
} else {
    Write-Host "[!] Could not find activation script at $activateScript" -ForegroundColor Yellow
}

$venvPython = ".\.venv\Scripts\python.exe"

# Upgrade pip
if (Test-Path $venvPython) {
    & $venvPython -m pip install --upgrade pip --quiet
} else {
    Write-Host "[!] Could not find venv python at $venvPython" -ForegroundColor Red
    exit 1
}

# --- Install Zettlecast ---
Write-Host ""
Write-Host "[*] Installing Zettlecast dependencies..." -ForegroundColor Yellow
Write-Host "(This may take a few minutes...)"

& $venvPython -m pip install -e ".[dev]"

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install base dependencies!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "  - PyTorch: May need manual installation from https://pytorch.org"
    Write-Host "  - C++ Build Tools: Install from https://visualstudio.microsoft.com/downloads/"
    exit 1
}

Write-Host "[OK] Base dependencies installed" -ForegroundColor Green

# --- Optional: Install Podcast/NeMo dependencies ---
Write-Host ""
Write-Host "[?] Do you want to install podcast transcription support (NVIDIA NeMo)?" -ForegroundColor Yellow
Write-Host "    This includes: parakeet-tdt-0.6b-v2 (transcription) + MSDD (diarization)"
Write-Host "    Requires ~5GB disk space and CUDA GPU recommended." -ForegroundColor Gray
$installPodcast = Read-Host "Install podcast support? (y/N)"

if ($installPodcast -match '^[Yy]') {
    Write-Host ""
    Write-Host "[*] Installing NeMo toolkit and podcast dependencies..." -ForegroundColor Yellow
    Write-Host "(This may take 5-10 minutes...)"
    
    & $venvPython -m pip install -e ".[podcast]"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] Failed to install podcast dependencies." -ForegroundColor Yellow
        Write-Host "    You can try again later with: pip install -e '.[podcast]'" -ForegroundColor Gray
    } else {
        Write-Host "[OK] Podcast/NeMo dependencies installed" -ForegroundColor Green
        
        # Apply Windows-specific NeMo patch
        Write-Host ""
        Write-Host "[*] Applying Windows compatibility patch for NeMo..." -ForegroundColor Yellow
        & $venvPython scripts/patch_nemo_windows.py
        
        Write-Host ""
        Write-Host "    Enable in .env: USE_NEMO=true" -ForegroundColor Gray
    }
}

# --- Install Ollama ---
Write-Host ""
Write-Host "[*] Checking for Ollama..." -ForegroundColor Yellow

$ollamaInstalled = $false
try {
    $ollamaVersion = & ollama --version 2>&1
    if ($ollamaVersion) {
        Write-Host "[OK] Ollama found: $ollamaVersion" -ForegroundColor Green
        $ollamaInstalled = $true
    }
} catch {}

if (-not $ollamaInstalled) {
    Write-Host "[!] Ollama not found. Skipping model pull." -ForegroundColor Yellow
    Write-Host "Please install Ollama later from https://ollama.com/download/windows"
}

# --- Pull default model ---
Write-Host ""
Write-Host "[*] Pulling default LLM model (llama3.2:3b)..." -ForegroundColor Yellow

try {
    # Check if Ollama service is running
    $ollamaRunning = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
    if (-not $ollamaRunning) {
        Write-Host "Starting Ollama service..."
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 3
    }
    
    & ollama pull llama3.2:3b
    Write-Host "[OK] Model ready" -ForegroundColor Green
} catch {
    Write-Host "[!] Could not pull model. You can do this later with: ollama pull llama3.2:3b" -ForegroundColor Yellow
}

# --- Create directories ---
Write-Host ""
Write-Host "[*] Creating storage directories..." -ForegroundColor Yellow

$brainStorage = Join-Path $env:USERPROFILE "_BRAIN_STORAGE"
$lancedbPath = Join-Path $brainStorage ".lancedb"

New-Item -ItemType Directory -Force -Path $brainStorage | Out-Null
New-Item -ItemType Directory -Force -Path $lancedbPath | Out-Null

Write-Host "[OK] Created $brainStorage" -ForegroundColor Green

# --- Generate config ---
Write-Host ""
Write-Host "[*] Generating configuration..." -ForegroundColor Yellow

if (-not (Test-Path ".env")) {
    # Generate API token
    $bytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $rng.GetBytes($bytes)
    $apiToken = [Convert]::ToBase64String($bytes) -replace '\+', '-' -replace '/', '_' -replace '=', ''
    
    $envContent = @"
# Zettlecast Configuration
# Generated: $(Get-Date)

# API Security
API_TOKEN=$apiToken

# Storage
STORAGE_PATH=$brainStorage
LANCEDB_PATH=$lancedbPath

# Models
EMBEDDING_MODEL=google/embeddinggemma-300m
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
WHISPER_MODEL=medium

# LLM
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b

# Features
ENABLE_CONTEXT_ENRICHMENT=false

# Server
API_PORT=8000
UI_PORT=8501
"@
    
    Set-Content -Path ".env" -Value $envContent
    Write-Host "[OK] Created .env with secure token" -ForegroundColor Green
    Write-Host ""
    Write-Host "[KEY] Your API token (save this for bookmarklet):" -ForegroundColor Cyan
    Write-Host "      $apiToken" -ForegroundColor White
} else {
    Write-Host "[OK] .env already exists" -ForegroundColor Green
}

# --- Summary ---
Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "[OK] Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To start Zettlecast:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  python -m uvicorn zettlecast.main:app --port 8000" -ForegroundColor White
Write-Host ""
Write-Host "  # In a new terminal:" -ForegroundColor Gray
Write-Host "  cd frontend && npm run dev" -ForegroundColor White
Write-Host ""
Write-Host "Or use the CLI:" -ForegroundColor Cyan
Write-Host "  zettlecast serve" -ForegroundColor White
Write-Host "=========================================" -ForegroundColor Green
