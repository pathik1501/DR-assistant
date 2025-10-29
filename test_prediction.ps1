# Test the prediction endpoint
Write-Host "Testing DR Assistant API..." -ForegroundColor Cyan

# Test health
Write-Host "`n1. Checking health endpoint..." -ForegroundColor Yellow
try {
    $health = Invoke-WebRequest -Uri "http://localhost:8080/health" -Method GET
    Write-Host "✓ Health check passed: $($health.StatusCode)" -ForegroundColor Green
    Write-Host "Response: $($health.Content)"
} catch {
    Write-Host "✗ Health check failed: $_" -ForegroundColor Red
    exit 1
}

# Test stats
Write-Host "`n2. Checking prediction stats..." -ForegroundColor Yellow
try {
    $stats = Invoke-WebRequest -Uri "http://localhost:8080/predictions/stats" -Method GET
    Write-Host "✓ Stats endpoint working: $($stats.StatusCode)" -ForegroundColor Green
    Write-Host "Response: $($stats.Content)"
} catch {
    Write-Host "✗ Stats endpoint failed: $_" -ForegroundColor Red
}

Write-Host "`n✓ API is running successfully!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Open browser: http://localhost:8080/docs"
Write-Host "2. Use the predict endpoint to test with an image"
Write-Host "3. Check the explanation field for Grad-CAM visualizations"
