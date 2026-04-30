# Download NYC Taxi Trip Data (January 2023)
# This script downloads the dataset used for ACADP evaluation

param(
    [string]$OutputDir = "data"
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  ACADP - Dataset Download" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Write-Host "Created directory: $OutputDir" -ForegroundColor Green
}

$BaseUrl = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
$FileName = "yellow_tripdata_2023-01.parquet"
$FilePath = Join-Path $OutputDir $FileName

if (Test-Path $FilePath) {
    $fileSize = (Get-Item $FilePath).Length / 1MB
    Write-Host "Dataset already exists: $FilePath ($([math]::Round($fileSize, 1)) MB)" -ForegroundColor Yellow
    Write-Host "Delete the file and re-run to re-download." -ForegroundColor Yellow
} else {
    Write-Host "Downloading: $FileName ..." -ForegroundColor White
    Write-Host "Source: $BaseUrl$FileName" -ForegroundColor Gray

    try {
        # Use .NET WebClient for faster download with progress
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadFile("$BaseUrl$FileName", $FilePath)

        $fileSize = (Get-Item $FilePath).Length / 1MB
        Write-Host "Download complete: $([math]::Round($fileSize, 1)) MB" -ForegroundColor Green
    } catch {
        Write-Host "Download failed: $_" -ForegroundColor Red
        Write-Host "Trying with Invoke-WebRequest..." -ForegroundColor Yellow

        Invoke-WebRequest -Uri "$BaseUrl$FileName" -OutFile $FilePath
        $fileSize = (Get-Item $FilePath).Length / 1MB
        Write-Host "Download complete: $([math]::Round($fileSize, 1)) MB" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Dataset ready at: $FilePath" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
