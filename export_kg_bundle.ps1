# export_kg_bundle.ps1
# 把当前 Neo4j 图和向量导出为 dist/zhongyaoxue-kg-bundle.zip，便于离线分发。

$ErrorActionPreference = "Stop"
$pythonExe = & "$PSScriptRoot\scripts\resolve_python.ps1"
$env:PYTHONIOENCODING = "utf-8"

New-Item -ItemType Directory -Force -Path "dist" | Out-Null
& $pythonExe scripts\export_kg_bundle.py --output dist\zhongyaoxue-kg-bundle.zip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Bundle ready: dist\zhongyaoxue-kg-bundle.zip" -ForegroundColor Green
