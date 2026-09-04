# build_kg_v2.ps1
# Build the evidence knowledge graph (v2 pipeline).
# 旧管线见 build_kg_dyn.ps1，保留作对照基线。

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Building Evidence Knowledge Graph (v2)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$env:PYTHONIOENCODING = "utf-8"
$pythonExe = & "$PSScriptRoot\scripts\resolve_python.ps1"

# v2 从 Neo4j 里已有的 Chunk 读取语料，因此不强制要求 origdata\zhongyaoxue.md。
# 若库里还没有 Chunk，请先跑 build_kg_dyn.ps1 或导入数据包。
Write-Host "Building from chunks in Neo4j ..." -ForegroundColor Yellow
Write-Host "抽取结果会缓存在 .cache\kg_v2，重跑不会重复付费。" -ForegroundColor DarkGray
Write-Host ""

& $pythonExe scripts\build_kg_v2.py --workers 6 --activate
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "构建未通过质量门槛，run 已标记 FAILED；旧 ACTIVE 版本继续服务。" -ForegroundColor Yellow
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Verifying ..." -ForegroundColor Yellow
& $pythonExe scripts\verify_kg_v2.py --herb 麻黄

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Evidence KG build completed" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
