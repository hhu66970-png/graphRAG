# eval_kg.ps1
# 跑抽取质量评测，并可选做阈值标定。

Write-Host "KG Extraction Evaluation" -ForegroundColor Cyan
Write-Host ""

$env:PYTHONIOENCODING = "utf-8"
$pythonExe = & "$PSScriptRoot\scripts\resolve_python.ps1"

& $pythonExe scripts\eval_extraction.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Threshold calibration ..." -ForegroundColor Yellow
& $pythonExe scripts\calibrate_thresholds.py
