# run_tests.ps1
# 跑不依赖 Neo4j/LLM 的本地 pytest。

Write-Host "GraphRAG Test Suite" -ForegroundColor Cyan
Write-Host ""
Write-Host "Running tests..." -ForegroundColor Yellow
Write-Host ""

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$pythonExe = & "$PSScriptRoot\scripts\resolve_python.ps1"
& $pythonExe -m pytest -m "not integration" -v --tb=short
$testExitCode = $LASTEXITCODE

Write-Host ""
Write-Host "Test run completed" -ForegroundColor Cyan

exit $testExitCode
