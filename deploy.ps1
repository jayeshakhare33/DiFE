# PowerShell deployment script for multi-container setup

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Fraud Detection GNN - Multi-Container Deployment" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "Docker is not running. Please start Docker first." -ForegroundColor Red
    exit 1
}

# Check if features exist
if (-not (Test-Path "./data/features/node_features.parquet")) {
    Write-Host "Warning: Features not found. Generating features first..." -ForegroundColor Yellow
    Write-Host "Run: python main.py --mode build" -ForegroundColor Yellow
    $response = Read-Host "Continue anyway? (y/n)"
    if ($response -ne "y" -and $response -ne "Y") {
        exit 1
    }
}

# Build Docker images
Write-Host ""
Write-Host "Building Docker images..." -ForegroundColor Green
docker-compose build
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to build Docker images." -ForegroundColor Red
    exit 1
}

# Start Redis first
Write-Host ""
Write-Host "Starting Redis..." -ForegroundColor Green
docker-compose up -d redis
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to start Redis container." -ForegroundColor Red
    exit 1
}

# Wait for Redis to be healthy
Write-Host "Waiting for Redis to be ready..." -ForegroundColor Yellow
$timeout = 30
$ready = $false

while ($timeout -gt 0) {
    try {
        $result = docker exec redis redis-cli ping 2>$null
        if ($result -eq "PONG") {
            Write-Host "Redis is ready!" -ForegroundColor Green
            $ready = $true
            break
        }
    } catch {
        # continue waiting
    }
    Start-Sleep -Seconds 1
    $timeout--
}

if (-not $ready) {
    Write-Host "Redis failed to start within the timeout period." -ForegroundColor Red
    exit 1
}

# Load features into Redis
Write-Host ""
Write-Host "Loading features into Redis..." -ForegroundColor Green
docker-compose up feature-loader
if ($LASTEXITCODE -eq 0) {
    Write-Host "Features loaded successfully!" -ForegroundColor Green
} else {
    Write-Host "Feature loading failed." -ForegroundColor Red
    exit 1
}

# Start API services
Write-Host ""
Write-Host "Starting API services..." -ForegroundColor Green
docker-compose up -d api
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to start API container." -ForegroundColor Red
    exit 1
}

# Wait for API to be ready
Write-Host "Waiting for API to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check API health
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "API is ready!" -ForegroundColor Green
    } else {
        Write-Host "API responded but not healthy (Status: $($response.StatusCode))" -ForegroundColor Yellow
    }
} catch {
    Write-Host "API may not be fully ready yet." -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Services:" -ForegroundColor White
Write-Host "  - Redis:      localhost:6379" -ForegroundColor White
Write-Host "  - API:        http://localhost:8000" -ForegroundColor White
Write-Host ""
Write-Host "Useful commands:" -ForegroundColor White
Write-Host "  - View logs:    docker-compose logs -f" -ForegroundColor White
Write-Host "  - Stop all:     docker-compose down" -ForegroundColor White
Write-Host "  - Start training: docker-compose --profile training up" -ForegroundColor White
Write-Host ""
