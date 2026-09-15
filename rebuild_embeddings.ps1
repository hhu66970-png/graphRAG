# rebuild_embeddings.ps1
# 按当前嵌入模型重写已有 Chunk 向量，不调用抽取 LLM。

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Rebuilding Local Chinese Embeddings" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$env:PYTHONIOENCODING = "utf-8"
$pythonExe = & "$PSScriptRoot\scripts\resolve_python.ps1"
& $pythonExe "$PSScriptRoot\scripts\rebuild_embeddings.py"
exit $LASTEXITCODE
