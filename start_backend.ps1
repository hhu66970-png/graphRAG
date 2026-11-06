# 启动 Flask 后端并检查数据库连接。
param(
    [switch]$Detached,
    [ValidateRange(1, 300)]
    [int]$StartupTimeoutSeconds = 30
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $OutputEncoding
$proc = $null
$runtimeProc = $null
$launchDirectory = $null
$launchToken = [guid]::NewGuid().ToString('N')
$launchedAt = $null

function Get-LaunchedRuntime {
    param($Launcher, [string]$Directory, [string]$Token, [datetime]$StartedAt)
    $ownerPath = Join-Path $Directory 'owner.json'
    if (-not (Test-Path -LiteralPath $ownerPath)) { return $null }
    $owner = [System.IO.File]::ReadAllText($ownerPath) | ConvertFrom-Json
    # Only our private bootstrap can identify the interpreter behind a Windows venv redirector.
    if ($owner.token -ne $Token -or $owner.pid -le 0 -or
        ($owner.pid -ne $Launcher.Id -and $owner.parent_pid -ne $Launcher.Id)) {
        throw 'Backend runtime ownership could not be verified.'
    }
    try { $runtime = [System.Diagnostics.Process]::GetProcessById([int]$owner.pid) }
    catch [System.ArgumentException] { return $null }
    if ($runtime.StartTime.ToUniversalTime() -lt $StartedAt.AddSeconds(-1)) {
        throw 'Backend runtime PID belongs to an older process.'
    }
    return $runtime
}

function Stop-LaunchedProcessTree {
    param($Process)
    if (-not $Process) { return }
    try {
        $Process.Refresh()
        if ($Process.HasExited) { return }
        if ([System.Environment]::OSVersion.Platform -eq 'Win32NT') {
            # taskkill enumerates the full Windows redirector tree before terminating its root.
            & "$env:SystemRoot\System32\taskkill.exe" /PID $Process.Id /T /F 2>$null | Out-Null
            if ($LASTEXITCODE -ne 0) {
                $Process.Refresh()
                if (-not $Process.HasExited) { throw 'Could not stop the launched backend tree.' }
            }
        } else {
            $Process.Kill($true)
        }
        $null = $Process.WaitForExit(5000)
    } catch {
        $Process.Refresh()
        if ($Process.HasExited) { return }
        $Process.Kill()
    }
}

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
        $launchDirectory = Join-Path ([System.IO.Path]::GetTempPath()) "graphrag-launch-$launchToken"
        $null = [System.IO.Directory]::CreateDirectory($launchDirectory)
        $bootstrapPath = Join-Path $launchDirectory 'launch.py'
        $bootstrap = @'
import json
import os
from pathlib import Path
import runpy
import sys

directory = Path(__file__).resolve().parent
backend = Path(sys.argv[1]).resolve()
owner = {"token": sys.argv[2], "pid": os.getpid(), "parent_pid": os.getppid()}
with (directory / "owner.tmp").open("x", encoding="utf-8") as stream:
    json.dump(owner, stream)
os.replace(directory / "owner.tmp", directory / "owner.json")
sys.argv = [str(backend)]
sys.path[0] = str(backend.parent)
runpy.run_path(str(backend), run_name="__main__")
'@
        [System.IO.File]::WriteAllText($bootstrapPath, $bootstrap, [System.Text.UTF8Encoding]::new($false))
        $outLog = Join-Path $PSScriptRoot "backend.api.out.log"
        $errLog = Join-Path $PSScriptRoot "backend.api.err.log"
        $startOptions = @{
            FilePath = $pythonExe
            ArgumentList = ('"' + $bootstrapPath + '" "' + $backendScript + '" "' + $launchToken + '"')
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
        $launchedAt = $proc.StartTime.ToUniversalTime()

        $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
        while ((Get-Date) -lt $deadline) {
            if (-not $runtimeProc) {
                $runtimeProc = Get-LaunchedRuntime $proc $launchDirectory $launchToken $launchedAt
            }
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
            if ($runtimeProc -and $health -and $health.status -eq 'ok' -and $health.pid -eq $runtimeProc.Id) {
                $proc.Refresh()
                $runtimeProc.Refresh()
                if ($proc.HasExited -or $runtimeProc.HasExited) { throw "Backend exited during startup. See $errLog" }
                Write-Host "Backend ready at $baseUrl (PID $($runtimeProc.Id))." -ForegroundColor Green
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
        if (-not $runtimeProc -and $launchDirectory -and $launchedAt) {
            try { $runtimeProc = Get-LaunchedRuntime $proc $launchDirectory $launchToken $launchedAt } catch { }
        }
        Stop-LaunchedProcessTree $proc
        Stop-LaunchedProcessTree $runtimeProc
    }
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
} finally {
    if ($launchDirectory) {
        # Remove only this launch's known temporary files, never a recursive project path.
        foreach ($name in @('launch.py', 'owner.json', 'owner.tmp')) {
            Remove-Item -LiteralPath (Join-Path $launchDirectory $name) -Force -ErrorAction SilentlyContinue
        }
        try { [System.IO.Directory]::Delete($launchDirectory, $false) } catch { }
    }
    Pop-Location
}
