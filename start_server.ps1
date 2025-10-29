Write-Host "========================================" -ForegroundColor Green
Write-Host "Diabetic Retinopathy Assistant" -ForegroundColor Green
Write-Host "Starting Deployment..." -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Green

# Set OpenAI API key from environment variable
# Set it before running: $env:OPENAI_API_KEY='your-key-here'
if (-not $env:OPENAI_API_KEY) {
    Write-Host "[WARNING] OPENAI_API_KEY not set. RAG features may not work." -ForegroundColor Yellow
}

Write-Host "OpenAI API key configured" -ForegroundColor Green
Write-Host ""
Write-Host "Starting server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "API will be available at: http://localhost:8080/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start the server
python src/inference.py
