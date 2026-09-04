# export_kg_bundle.ps1
# Export Neo4j graph + embeddings into dist/zhongyaoxue-kg-bundle.zip

$ErrorActionPreference = "Stop"
$pythonExe = & "$PSScriptRoot\scripts\resolve_python.ps1"
$env:PYTHONIOENCODING = "utf-8"

New-Item -ItemType Directory -Force -Path "dist" | Out-Null
& $pythonExe scripts\export_kg_bundle.py --output dist\zhongyaoxue-kg-bundle.zip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Bundle ready: dist\zhongyaoxue-kg-bundle.zip" -ForegroundColor Green
