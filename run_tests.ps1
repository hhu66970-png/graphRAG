# run_tests.ps1
# Run local tests that do not require Neo4j or an LLM.

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "GraphRAG Test Suite" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host ""
Write-Host "Running tests..." -ForegroundColor Yellow
Write-Host ""

$env:PYTHONIOENCODING = "utf-8"
$pythonExe = & "$PSScriptRoot\scripts\resolve_python.ps1"
& $pythonExe -m pytest -m "not integration" -v --tb=short
$testExitCode = $LASTEXITCODE

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Test run completed" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

exit $testExitCode
