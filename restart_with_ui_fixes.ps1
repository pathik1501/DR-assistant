# Stop existing API servers
Write-Host "Stopping existing servers..." -ForegroundColor Yellow
Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.CommandLine -like "*src/inference.py*"} | Stop-Process -Force

Start-Sleep -Seconds 2

# Start new server with fixes
Write-Host ""
Write-Host "Starting API server with UI fixes..." -ForegroundColor Green
cd "C:\Users\pathi\Documents\DR assistant"
# Set API key from environment variable (set this before running)
# $env:OPENAI_API_KEY = $env:OPENAI_API_KEY
if (-not $env:OPENAI_API_KEY) {
    Write-Host "[WARNING] OPENAI_API_KEY not set. RAG features may not work." -ForegroundColor Yellow
    Write-Host "Set it with: `$env:OPENAI_API_KEY='your-key-here'" -ForegroundColor Yellow
}
Start-Process python -ArgumentList "src/inference.py"

Write-Host ""
Write-Host "[OK] API server starting..." -ForegroundColor Green
Write-Host ""
Write-Host "What's fixed:" -ForegroundColor Cyan
Write-Host "  - Clinical hints ALWAYS generated" -ForegroundColor White
Write-Host "  - User-friendly recommendations with emojis" -ForegroundColor White
Write-Host "  - Fallback templates if RAG fails" -ForegroundColor White
Write-Host ""
Write-Host "Wait 10 seconds, then:" -ForegroundColor Yellow
Write-Host "  1. Start UI: streamlit run frontend/app_new.py" -ForegroundColor Cyan
Write-Host "  2. Or use: powershell -ExecutionPolicy Bypass -File start_ui.ps1" -ForegroundColor Cyan
Write-Host ""

