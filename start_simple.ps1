# Stop existing servers
Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.CommandLine -like "*src/inference.py*"} | Stop-Process -Force

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DR Assistant - Simplified Version" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Start-Sleep -Seconds 2

# Start server
Write-Host "Starting server..." -ForegroundColor Yellow
cd "C:\Users\pathi\Documents\DR assistant"
# Set API key from environment variable (set before running)
if (-not $env:OPENAI_API_KEY) {
    Write-Host "[WARNING] OPENAI_API_KEY not set. RAG features may not work." -ForegroundColor Yellow
}
Start-Process python -ArgumentList "src/inference.py"

Write-Host ""
Write-Host "[OK] Server starting..." -ForegroundColor Green
Write-Host ""
Write-Host "What's fixed:" -ForegroundColor Cyan
Write-Host "  - No more blank heatmaps" -ForegroundColor White
Write-Host "  - Shows only prediction and confidence" -ForegroundColor White
Write-Host "  - Clinical hints now working" -ForegroundColor White
Write-Host "  - Clean, simple output" -ForegroundColor White
Write-Host ""
Write-Host "Access: http://localhost:8080/docs" -ForegroundColor Yellow
Write-Host ""
Write-Host "Wait 10 seconds for server to start..." -ForegroundColor Gray

