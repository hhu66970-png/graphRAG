# 启动 Flask 后端并检查数据库连接。
param(
    [switch]$Detached,
    [ValidateRange(1, 300)]
    [int]$StartupTimeoutSeconds = 30
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$proc = $null
Push-Location $PSScriptRoot
try {
    $pythonExe = & "$PSScriptRoot\scripts\resolve_python.ps1"
    if (-not $pythonExe) { throw "No Python interpreter was found." }
    $backendScript = Join-Path $PSScriptRoot "backend/api.py"

    Write-Host "Checking Neo4j connection..." -ForegroundColor Yellow
    $probeScript = @'
import json
from backend.api import driver, _server_options
try:
    driver.verify_connectivity()
    print(json.dumps(_server_options()))
finally:
    driver.close()
'@
    $probeResult = & $pythonExe -c $probeScript
    if ($LASTEXITCODE -ne 0) {
        throw "Neo4j connection check failed. Check the service and .env settings."
    }
    $server = $probeResult | ConvertFrom-Json
    if (-not $server.host -or -not $server.port) { throw "Invalid backend configuration." }
    Write-Host "Neo4j connected" -ForegroundColor Green

    $healthHost = $server.host
    if ($healthHost -eq '0.0.0.0') { $healthHost = '127.0.0.1' }
    if ($healthHost -eq '::') { $healthHost = '::1' }
    if ($healthHost.Contains(':')) { $healthHost = "[$healthHost]" }
    $baseUrl = "http://${healthHost}:$($server.port)"

    if ($Detached) {
        $outLog = Join-Path $PSScriptRoot "backend.api.out.log"
        $errLog = Join-Path $PSScriptRoot "backend.api.err.log"
        $startOptions = @{
            FilePath = $pythonExe
            ArgumentList = ('"' + $backendScript + '"')
            WorkingDirectory = $PSScriptRoot
            RedirectStandardOutput = $outLog
            RedirectStandardError = $errLog
            PassThru = $true
        }
        if ([System.Environment]::OSVersion.Platform -eq 'Win32NT') {
            $startOptions.WindowStyle = 'Hidden'
        }
        $proc = Start-Process @startOptions
        if (-not $proc) { throw "Failed to launch backend." }

        $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
        while ((Get-Date) -lt $deadline) {
            $proc.Refresh()
            if ($proc.HasExited) {
                throw "Backend exited with code $($proc.ExitCode). See $errLog"
            }
            try {
                $health = Invoke-RestMethod -Uri "$baseUrl/api/health" -TimeoutSec 1
            } catch {
                $health = $null
            }
            # A pre-existing server on this port must not count as this launch succeeding.
            if ($health -and $health.status -eq 'ok' -and $health.pid -eq $proc.Id) {
                $proc.Refresh()
                if ($proc.HasExited) { throw "Backend exited during startup. See $errLog" }
                Write-Host "Backend ready at $baseUrl (PID $($proc.Id))." -ForegroundColor Green
                exit 0
            }
            Start-Sleep -Milliseconds 250
        }
        throw "Backend did not become healthy within $StartupTimeoutSeconds seconds. See $errLog"
    }

    Write-Host "Application: $baseUrl" -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
    & $pythonExe $backendScript
    exit $LASTEXITCODE
} catch {
    if ($proc) {
        $proc.Refresh()
        if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -ErrorAction SilentlyContinue }
    }
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}
