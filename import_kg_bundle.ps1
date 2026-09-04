# import_kg_bundle.ps1
# Import dist/zhongyaoxue-kg-bundle.zip into the configured Neo4j database.

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
