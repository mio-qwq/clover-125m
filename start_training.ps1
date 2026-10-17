param(
    [string]$OutputDirectory,
    [string]$ResumeCheckpoint
)

$ErrorActionPreference = 'Stop'
$trainPython = Join-Path $PSScriptRoot '.venv-train/Scripts/python.exe'
$trainScript = Join-Path $PSScriptRoot 'scripts/train_lm.py'
$trainConfig = Join-Path $PSScriptRoot 'configs/english_125m.json'
$trainArguments = @('-B', $trainScript, '--config', $trainConfig, '--start-training')
if ($OutputDirectory) { $trainArguments += @('--output', $OutputDirectory) }
if ($ResumeCheckpoint) { $trainArguments += @('--resume', $ResumeCheckpoint) }

Write-Host 'Starting actual English 125M pretraining. This launcher performs model updates.'
& $trainPython @trainArguments
if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE. Inspect the new run's metrics.jsonl." }
