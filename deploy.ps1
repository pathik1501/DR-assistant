# PowerShell deployment script for DR Assistant

Write-Host "🚀 DR Assistant Deployment Script" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

# Check if .env exists
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  .env file not found" -ForegroundColor Yellow
    Write-Host "Creating .env from .env.example..." -ForegroundColor Yellow
    
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host ""
        Write-Host "❌ IMPORTANT: Edit .env file and add your OPENAI_API_KEY" -ForegroundColor Red
        Write-Host "   Then run this script again" -ForegroundColor Red
        exit 1
    } else {
        Write-Host "❌ .env.example not found" -ForegroundColor Red
        exit 1
    }
}

# Check if OPENAI_API_KEY is set
$envContent = Get-Content ".env" -Raw
if ($envContent -notmatch "OPENAI_API_KEY=sk-proj-") {
    Write-Host "❌ OPENAI_API_KEY not set in .env file" -ForegroundColor Red
    Write-Host "   Please edit .env and add your OpenAI API key" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Environment file found" -ForegroundColor Green
Write-Host ""

# Check if Docker is installed
try {
    docker --version | Out-Null
    Write-Host "✅ Docker is installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not installed" -ForegroundColor Red
    Write-Host "   Install Docker Desktop from: https://www.docker.com/get-started" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Build Docker image
Write-Host "📦 Building Docker image..." -ForegroundColor Yellow
docker build -t dr-assistant:latest .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ Docker image built successfully" -ForegroundColor Green
Write-Host ""

# Start services
Write-Host "🚀 Starting services..." -ForegroundColor Yellow
docker-compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to start services" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host "✅ Deployment Complete!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host ""
Write-Host "📍 Access your application:" -ForegroundColor Cyan
Write-Host "   Frontend: http://localhost:8501" -ForegroundColor White
Write-Host "   API:      http://localhost:8080" -ForegroundColor White
Write-Host "   API Docs: http://localhost:8080/docs" -ForegroundColor White
Write-Host ""
Write-Host "📋 Useful commands:" -ForegroundColor Cyan
Write-Host "   View logs:    docker-compose logs -f" -ForegroundColor White
Write-Host "   Stop:         docker-compose down" -ForegroundColor White
Write-Host "   Restart:      docker-compose restart" -ForegroundColor White
Write-Host ""

