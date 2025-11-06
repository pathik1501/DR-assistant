# Simple DR Assistant Startup
Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "  DR Assistant - Simple Mode" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""

# Check API
Write-Host "Checking API..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -TimeoutSec 2 -UseBasicParsing
    Write-Host "✅ API is running" -ForegroundColor Green
} catch {
    Write-Host "❌ API not running. Starting it..." -ForegroundColor Red
    Write-Host ""
    Write-Host "You need to start the API server in a separate terminal:" -ForegroundColor Yellow
    Write-Host "  python src/inference.py" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Then come back and run this script again." -ForegroundColor Yellow
    Write-Host ""
    exit
}

Write-Host ""
Write-Host "Starting simple frontend..." -ForegroundColor Yellow
Write-Host ""
Write-Host "The interface will open in your browser." -ForegroundColor Cyan
Write-Host ""

# Start Streamlit
streamlit run simple_frontend.py

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
