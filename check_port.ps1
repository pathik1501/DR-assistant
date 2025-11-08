# Check what's using port 8080
Write-Host ""
Write-Host "Checking port 8080..." -ForegroundColor Cyan

$connections = Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue

if ($connections) {
    Write-Host "[FOUND] Port 8080 is in use:" -ForegroundColor Yellow
    Write-Host ""
    foreach ($conn in $connections) {
        $process = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "  Process: $($process.ProcessName)" -ForegroundColor White
            Write-Host "  PID: $($conn.OwningProcess)" -ForegroundColor White
            Write-Host "  Command: $($process.Path)" -ForegroundColor Gray
            Write-Host ""
        }
    }
    Write-Host "To stop these processes, run: .\stop_api_server.ps1" -ForegroundColor Cyan
} else {
    Write-Host "[OK] Port 8080 is available" -ForegroundColor Green
}

Write-Host ""



