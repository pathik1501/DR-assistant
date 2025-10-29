# Stop existing servers
Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.CommandLine -like "*src/inference.py*"} | Stop-Process -Force

Write-Host "Stopped existing servers" -ForegroundColor Yellow

# Wait a moment
Start-Sleep -Seconds 2

# Start new server
Write-Host "Starting new server..." -ForegroundColor Green
cd "C:\Users\pathi\Documents\DR assistant"
# Set API key from environment variable (set before running)
if (-not $env:OPENAI_API_KEY) {
    Write-Host "[WARNING] OPENAI_API_KEY not set. RAG features may not work." -ForegroundColor Yellow
}
Start-Process python -ArgumentList "src/inference.py" -PassThru

Write-Host ""
Write-Host "Server restarted with explanations ALWAYS enabled!" -ForegroundColor Green
Write-Host "Go to: http://localhost:8080/docs" -ForegroundColor Cyan
Write-Host "And try uploading an image again" -ForegroundColor Cyan

