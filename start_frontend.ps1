# Start the standalone web frontend
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DR Assistant - Standalone Frontend" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Starting frontend server..." -ForegroundColor Yellow
cd "C:\Users\pathi\Documents\DR assistant"

# Note: Set OPENAI_API_KEY in separate terminal for API server if needed
# $env:OPENAI_API_KEY="your-key-here"

python serve_frontend.py

Write-Host ""
Write-Host "[OK] Frontend server started on http://localhost:3000" -ForegroundColor Green
Write-Host ""
Write-Host "Open your browser to: http://localhost:3000/index.html" -ForegroundColor Cyan
Write-Host ""
Write-Host "Make sure the API server is running separately on localhost:8080" -ForegroundColor Yellow
Write-Host "To start API server, use: python src/inference.py" -ForegroundColor Gray
Write-Host ""

