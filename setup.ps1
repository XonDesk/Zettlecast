# Zettlecast Setup Script for Windows
# Installs all dependencies including Ollama

Write-Host "🧠 Zettlecast Setup" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan

# Detect Windows version
$OSInfo = Get-WmiObject -Class Win32_OperatingSystem
Write-Host "Detected: Windows $($OSInfo.Version) ($env:PROCESSOR_ARCHITECTURE)"

# Check for Python
Write-Host ""
Write-Host "📦 Checking Python installation..." -ForegroundColor Yellow

$pythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $version = & $cmd --version 2>&1
        if ($version -match "Python 3\.1[0-9]") {
            $pythonCmd = $cmd
            Write-Host "✅ Found Python: $version" -ForegroundColor Green
            break
        }
    } catch {}
}

if (-not $pythonCmd) {
    Write-Host "❌ Python 3.10+ not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python 3.10 or later from:" -ForegroundColor Yellow
    Write-Host "  https://www.python.org/downloads/"
    Write-Host ""
    Write-Host "Make sure to check 'Add Python to PATH' during installation."
    exit 1
}

# --- Python Environment ---
Write-Host ""
Write-Host "📦 Setting up Python environment..." -ForegroundColor Yellow

if (-not (Test-Path ".venv")) {
    & $pythonCmd -m venv .venv
    Write-Host "Created virtual environment"
}

# Activate virtual environment
$activateScript = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "Activated virtual environment"
} else {
    Write-Host "⚠️  Could not find activation script at $activateScript" -ForegroundColor Yellow
}

# Upgrade pip
& python -m pip install --upgrade pip --quiet

# --- Install Zettlecast ---
Write-Host ""
Write-Host "📦 Installing Zettlecast dependencies..." -ForegroundColor Yellow
Write-Host "(This may take a few minutes...)"

& pip install -e ".[dev]"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install dependencies!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "  - PyTorch: May need manual installation from https://pytorch.org"
    Write-Host "  - C++ Build Tools: Install from https://visualstudio.microsoft.com/downloads/"
    exit 1
}

Write-Host "✅ Python dependencies installed" -ForegroundColor Green

# --- Install Ollama ---
Write-Host ""
Write-Host "🦙 Installing Ollama..." -ForegroundColor Yellow

$ollamaInstalled = $false
try {
    $ollamaVersion = & ollama --version 2>&1
    if ($ollamaVersion) {
        Write-Host "✅ Ollama already installed: $ollamaVersion" -ForegroundColor Green
        $ollamaInstalled = $true
    }
} catch {}

if (-not $ollamaInstalled) {
    Write-Host ""
    Write-Host "Ollama is not installed. Please:" -ForegroundColor Yellow
    Write-Host "  1. Download from: https://ollama.com/download/windows"
    Write-Host "  2. Run the installer"
    Write-Host "  3. Restart this script"
    Write-Host ""
    $response = Read-Host "Press 'Y' if you've already installed Ollama and want to continue, or 'N' to exit"
    
    if ($response -ne 'Y' -and $response -ne 'y') {
        Write-Host ""
        Write-Host "Please install Ollama and run setup.ps1 again."
        exit 1
    }
}

# --- Pull default model ---
Write-Host ""
Write-Host "🤖 Pulling default LLM model (llama3.2:3b)..." -ForegroundColor Yellow

try {
    # Check if Ollama service is running
    $ollamaRunning = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
    if (-not $ollamaRunning) {
        Write-Host "Starting Ollama service..."
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 3
    }
    
    & ollama pull llama3.2:3b
    Write-Host "✅ Model ready" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Could not pull model. You can do this later with: ollama pull llama3.2:3b" -ForegroundColor Yellow
}

# --- Create directories ---
Write-Host ""
Write-Host "📁 Creating storage directories..." -ForegroundColor Yellow

$brainStorage = Join-Path $env:USERPROFILE "_BRAIN_STORAGE"
$lancedbPath = Join-Path $brainStorage ".lancedb"

New-Item -ItemType Directory -Force -Path $brainStorage | Out-Null
New-Item -ItemType Directory -Force -Path $lancedbPath | Out-Null

Write-Host "✅ Created $brainStorage" -ForegroundColor Green

# --- Generate config ---
Write-Host ""
Write-Host "⚙️  Generating configuration..." -ForegroundColor Yellow

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
    Write-Host "✅ Created .env with secure token" -ForegroundColor Green
    Write-Host ""
    Write-Host "🔑 Your API token (save this for bookmarklet):" -ForegroundColor Cyan
    Write-Host "   $apiToken" -ForegroundColor White
} else {
    Write-Host "✅ .env already exists" -ForegroundColor Green
}

# --- Summary ---
Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To start Zettlecast:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  python -m uvicorn zettlecast.main:app --port 8000" -ForegroundColor White
Write-Host ""
Write-Host "  # In a new terminal:" -ForegroundColor Gray
Write-Host "  streamlit run src\zettlecast\ui\app.py" -ForegroundColor White
Write-Host ""
Write-Host "Or use the CLI:" -ForegroundColor Cyan
Write-Host "  zettlecast serve" -ForegroundColor White
Write-Host "=========================================" -ForegroundColor Green
