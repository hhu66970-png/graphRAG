# generate_visualization.ps1
# Script to generate standalone visualization

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Generate Standalone Visualization" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Generating visualization..." -ForegroundColor Yellow
Write-Host ""

$env:PYTHONIOENCODING = "utf-8"
$pythonExe = & "$PSScriptRoot\scripts\resolve_python.ps1"
& $pythonExe graphragexpr\vis\visualize_standalone.py

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Visualization generated" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Open graphragexpr\vis\output\graph.html in your browser" -ForegroundColor Green
