# Rebuild semantic vectors for existing Neo4j Chunk nodes without calling the LLM.

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Rebuilding Local Chinese Embeddings" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$env:PYTHONIOENCODING = "utf-8"
$pythonExe = & "$PSScriptRoot\scripts\resolve_python.ps1"
& $pythonExe "$PSScriptRoot\scripts\rebuild_embeddings.py"
exit $LASTEXITCODE
