# Start Ollama (if needed), the Zettlecast API and the web UI.
#
#   .\start.ps1               API on :8000, web UI on :3000
#   .\start.ps1 --no-frontend API only

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PATH += ";$env:LOCALAPPDATA\Programs\Ollama"

if (-not (Get-Process ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Starting Ollama..."
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
}

& "$PSScriptRoot\.venv\Scripts\zettlecast.exe" serve @args
