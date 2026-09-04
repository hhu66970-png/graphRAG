# eval_kg.ps1
# Extraction quality evaluation + threshold calibration.

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "KG Extraction Evaluation" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$env:PYTHONIOENCODING = "utf-8"
$pythonExe = & "$PSScriptRoot\scripts\resolve_python.ps1"

& $pythonExe scripts\eval_extraction.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Threshold calibration ..." -ForegroundColor Yellow
& $pythonExe scripts\calibrate_thresholds.py
