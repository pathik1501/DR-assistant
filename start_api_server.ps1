# Start the FastAPI server (Backend)
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DR Assistant - API Server (Backend)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if OpenAI API key is set
if (-not $env:OPENAI_API_KEY) {
    Write-Host "[WARNING] OPENAI_API_KEY not set. RAG features may not work." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To set it, use this command:" -ForegroundColor Cyan
    Write-Host '  $env:OPENAI_API_KEY="sk-proj-your-key-here"' -ForegroundColor White
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne 'y') {
        Write-Host "Exiting. Set the API key and try again." -ForegroundColor Yellow
        exit
    }
} else {
    Write-Host "[OK] OPENAI_API_KEY is set" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting API server..." -ForegroundColor Yellow
cd "C:\Users\pathi\Documents\DR assistant"
python src/inference.py

Write-Host ""
Write-Host "[OK] API server should be running on http://localhost:8080" -ForegroundColor Green
Write-Host ""
Write-Host "API Documentation: http://localhost:8080/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""



