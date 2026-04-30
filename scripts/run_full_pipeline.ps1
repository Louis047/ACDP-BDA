# ACADP Full Pipeline - PowerShell Runner
# Usage: .\scripts\run_full_pipeline.ps1 [-Input <path>] [-Epsilon <float>] [-OutputDir <path>]

param(
    [string]$Input = "data\yellow_tripdata_2023-01.parquet",
    [float]$Epsilon = 1.0,
    [string]$OutputDir = "output",
    [string]$Mechanism = "laplace",
    [string]$AllocationMethod = "inverse_sensitivity",
    [switch]$NoSweep,
    [switch]$NoML,
    [string]$RegressionTarget = "fare_amount",
    [string]$ClassificationTarget = "payment_type",
    [int]$NTrials = 5,
    [int]$RandomState = 42
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  ACADP FULL PIPELINE" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Input: $Input"
Write-Host "  Privacy Budget (epsilon): $Epsilon"
Write-Host "  Mechanism: $Mechanism"
Write-Host "  Allocation: $AllocationMethod"
Write-Host "  Output: $OutputDir"
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Activate venv if exists
$VenvPython = ".\\.venv\\Scripts\\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Virtual environment not found at .venv" -ForegroundColor Red
    Write-Host "Run: python -m venv .venv" -ForegroundColor Yellow
    Write-Host "Then: .venv\Scripts\pip.exe install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Build command
$cmd = @(
    $VenvPython,
    "-m", "src.pipeline",
    "--input", $Input,
    "--epsilon", $Epsilon,
    "--output-dir", $OutputDir,
    "--mechanism", $Mechanism,
    "--allocation-method", $AllocationMethod,
    "--n-trials", $NTrials,
    "--random-state", $RandomState,
    "--log-level", "INFO"
)

if ($RegressionTarget) {
    $cmd += @("--regression-target", $RegressionTarget)
}
if ($ClassificationTarget) {
    $cmd += @("--classification-target", $ClassificationTarget)
}
if ($NoSweep) {
    $cmd += "--no-sweep"
}
if ($NoML) {
    $cmd += "--no-ml"
}

Write-Host "Running: $($cmd -join ' ')" -ForegroundColor Gray
Write-Host ""

& $cmd[0] $cmd[1..($cmd.Length-1)]

Write-Host ""
Write-Host "Pipeline execution complete!" -ForegroundColor Green
Write-Host "Results saved to: $OutputDir" -ForegroundColor Green
