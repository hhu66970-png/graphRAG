# start_backend.ps1
# 启动 Flask 后端：先探测 Neo4j Bolt，再运行 backend/api.py。
param(
    [switch]$Detached
)

Write-Host "GraphRAG Backend API" -ForegroundColor Cyan
Write-Host ""
$pythonExe = & "$PSScriptRoot\scripts\resolve_python.ps1"

# Check if Neo4j is running
Write-Host "Checking Neo4j connection..." -ForegroundColor Yellow
try {
    $result = & $pythonExe -c "from dotenv import load_dotenv; load_dotenv(); import os; from neo4j import GraphDatabase; d=GraphDatabase.driver(os.getenv('NEO4J_URL'), auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))); d.verify_connectivity(); print('OK')"
    if ($result -eq "OK") {
        Write-Host "✓ Neo4j connected" -ForegroundColor Green
    }
} catch {
    Write-Host "✗ Neo4j connection failed" -ForegroundColor Red
    Write-Host "Please start Neo4j first" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Application: http://localhost:5000" -ForegroundColor Green

if ($Detached) {
    $env:PYTHONIOENCODING = "utf-8"
    $outLog = Join-Path $PSScriptRoot "backend.api.out.log"
    $errLog = Join-Path $PSScriptRoot "backend.api.err.log"
    Write-Host "Starting backend in the background..." -ForegroundColor Yellow
    $proc = Start-Process -FilePath $pythonExe -ArgumentList "backend\api.py" `
        -WorkingDirectory $PSScriptRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $outLog `
        -RedirectStandardError $errLog `
        -PassThru
    if (-not $proc) {
        Write-Host "Failed to launch backend." -ForegroundColor Red
        exit 1
    }
    Write-Host "Backend started (PID $($proc.Id)). Keep Neo4j running." -ForegroundColor Green
    exit 0
}

Write-Host "Starting backend server in this window..." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

$env:PYTHONIOENCODING = "utf-8"
& $pythonExe backend\api.py
