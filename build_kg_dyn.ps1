# build_kg_dyn.ps1
# 老师基线旧建图：按名称写节点边，仅作对照，正式构建请用 build_kg_v2.ps1。

Write-Host "Building Knowledge Graph - 中药学" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path "origdata\zhongyaoxue.md")) {
    Write-Host "Error: origdata\zhongyaoxue.md not found" -ForegroundColor Red
    Write-Host "Run scripts\preprocess_pdf.py first." -ForegroundColor Yellow
    exit 1
}

$env:PYTHONIOENCODING = "utf-8"
$pythonExe = & "$PSScriptRoot\scripts\resolve_python.ps1"

Write-Host ""
Write-Host "Testing data extraction..." -ForegroundColor Yellow
& $pythonExe graphragexpr\extract\sample_data_dyn.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Building knowledge graph from 中药学..." -ForegroundColor Yellow
Write-Host ""

& $pythonExe graphragexpr\extract\build_kg_dyn.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Creating vector index..." -ForegroundColor Yellow
& $pythonExe graphragexpr\extract\create_index.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host ""
Write-Host "Knowledge graph build completed" -ForegroundColor Cyan
