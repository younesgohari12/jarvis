param(
    [switch]$Training,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvironmentName = if ($Training) { ".venv-training" } else { ".venv-runtime" }
$EnvironmentPath = Join-Path $ProjectRoot $EnvironmentName
$Requirements = if ($Training) { "requirements-training.txt" } else { "requirements-runtime.txt" }

Set-Location $ProjectRoot

$PythonCommand = Get-Command py -ErrorAction SilentlyContinue
if ($PythonCommand) {
    $Python = @("py", "-3.13")
} else {
    $PythonCommand = Get-Command python -ErrorAction Stop
    $Python = @($PythonCommand.Source)
}

if ($Force -and (Test-Path $EnvironmentPath)) {
    Remove-Item -Recurse -Force $EnvironmentPath
}

if (-not (Test-Path $EnvironmentPath)) {
    if ($Python.Count -eq 2) {
        & $Python[0] $Python[1] -m venv $EnvironmentPath
    } else {
        & $Python[0] -m venv $EnvironmentPath
    }
}

$EnvironmentPython = Join-Path $EnvironmentPath "Scripts\python.exe"
& $EnvironmentPython -m pip install --upgrade pip
& $EnvironmentPython -m pip install -r (Join-Path $ProjectRoot $Requirements)
& $EnvironmentPython -c "import sqlite3, tkinter, numpy; print('Runtime check: OK')"

Write-Host ""
Write-Host "JARVIS environment is ready:" -ForegroundColor Cyan
Write-Host "  $EnvironmentPath"
Write-Host "Activate it with:"
Write-Host "  .\$EnvironmentName\Scripts\Activate.ps1" -ForegroundColor Yellow
if ($Training) {
    Write-Host "Then run: python training_studio.py"
} else {
    Write-Host "Then run: python main.py"
}
