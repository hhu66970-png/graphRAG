# Resolve a project Python without depending on whichever python.exe appears first in PATH.

$repoRoot = Split-Path -Parent $PSScriptRoot
$candidates = [System.Collections.Generic.List[string]]::new()

$candidates.Add((Join-Path $repoRoot '.venv\Scripts\python.exe'))

if ($env:CONDA_PREFIX -and (Split-Path -Leaf $env:CONDA_PREFIX) -eq 'zhongyao_graphrag') {
    $candidates.Add((Join-Path $env:CONDA_PREFIX 'python.exe'))
}

if ($env:CONDA_EXE) {
    $condaRoot = Split-Path -Parent (Split-Path -Parent $env:CONDA_EXE)
    $candidates.Add((Join-Path $condaRoot 'envs\zhongyao_graphrag\python.exe'))
}

if ($env:USERPROFILE) {
    $candidates.Add((Join-Path $env:USERPROFILE 'anaconda3\envs\zhongyao_graphrag\python.exe'))
    $candidates.Add((Join-Path $env:USERPROFILE 'miniconda3\envs\zhongyao_graphrag\python.exe'))
}

foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate) {
        Write-Output $candidate
        exit 0
    }
}

$pathPython = Get-Command python.exe -ErrorAction SilentlyContinue
if ($pathPython) {
    Write-Output $pathPython.Source
    exit 0
}

throw 'No usable Python was found. Create .venv or the zhongyao_graphrag Conda environment first.'
