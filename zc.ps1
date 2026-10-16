# Zettlecast CLI shortcut:  .\zc.ps1 podcast run   |   .\zc.ps1 search "gut training"
& "$PSScriptRoot\.venv\Scripts\zettlecast.exe" @args
exit $LASTEXITCODE
