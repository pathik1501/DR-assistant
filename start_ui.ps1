# Start the improved Streamlit UI
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting DR Assistant UI" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

cd "C:\Users\pathi\Documents\DR assistant"

# Check if API is running
Write-Host "Checking API connection..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -TimeoutSec 2 -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "[OK] API is running" -ForegroundColor Green
    }
} catch {
    Write-Host "[WARNING] API not responding!" -ForegroundColor Red
    Write-Host "Please start the API server first:" -ForegroundColor Yellow
    Write-Host "  python src/inference.py" -ForegroundColor White
    Write-Host ""
}

Write-Host ""
Write-Host "Starting Streamlit UI..." -ForegroundColor Yellow
Write-Host ""

# Start Streamlit
streamlit run frontend/app_new.py

Write-Host ""
Write-Host "UI will open in your browser at: http://localhost:8501" -ForegroundColor Cyan




