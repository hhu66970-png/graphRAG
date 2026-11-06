# import_kg_bundle.ps1
# 把离线 zip 导入本机 Neo4j，免 OCR、免重新抽取全书。

param(
    [string]$Bundle = "dist\zhongyaoxue-kg-bundle.zip",
    [switch]$Clear
)

$ErrorActionPreference = "Stop"
$pythonExe = & "$PSScriptRoot\scripts\resolve_python.ps1"
$env:PYTHONIOENCODING = "utf-8"

$argsList = @("scripts\import_kg_bundle.py", "--bundle", $Bundle)
if ($Clear) { $argsList += "--clear" }

& $pythonExe @argsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Import finished. Start backend with .\start_backend.ps1" -ForegroundColor Green
