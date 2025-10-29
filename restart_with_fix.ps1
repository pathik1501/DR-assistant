# Stop existing servers
Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.CommandLine -like "*src/inference.py*"} | Stop-Process -Force

Write-Host "Stopped existing servers" -ForegroundColor Yellow

# Wait a moment
Start-Sleep -Seconds 2

# Start new server
Write-Host "Starting new server with fixed Grad-CAM layers..." -ForegroundColor Green
cd "C:\Users\pathi\Documents\DR assistant"
# Set API key from environment variable (set before running)
if (-not $env:OPENAI_API_KEY) {
    Write-Host "[WARNING] OPENAI_API_KEY not set. RAG features may not work." -ForegroundColor Yellow
}
Start-Process python -ArgumentList "src/inference.py"

Write-Host ""
Write-Host "Server restarted with FIXED Grad-CAM!" -ForegroundColor Green
Write-Host "Fixed layer names: backbone.blocks.5.0, backbone.blocks.6.0" -ForegroundColor Cyan
Write-Host ""
Write-Host "Go to: http://localhost:8080/docs" -ForegroundColor Cyan
Write-Host "And try uploading an image again" -ForegroundColor Cyan
Write-Host ""
Write-Host "Wait 10 seconds for server to start..." -ForegroundColor Yellow

