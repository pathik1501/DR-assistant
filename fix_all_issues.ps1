# Comprehensive fix script - stops processes and provides guidance
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DR Assistant - Complete Fix Script" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Kill processes on port 8080
Write-Host "Step 1: Stopping processes on port 8080..." -ForegroundColor Yellow
Write-Host ""

# Method 1: Find and kill via netstat
$netstatOutput = netstat -ano | findstr ":8080"
$pids = @()

foreach ($line in $netstatOutput) {
    if ($line -match '\s+(\d+)$') {
        $pid = $matches[1]
        if ($pid -and $pid -ne '0') {
            $pids += $pid
        }
    }
}

$uniquePids = $pids | Select-Object -Unique

if ($uniquePids) {
    Write-Host "Found processes on port 8080:" -ForegroundColor Cyan
    foreach ($pid in $uniquePids) {
        $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "  - $($process.ProcessName) (PID: $pid)" -ForegroundColor White
            try {
                taskkill /F /PID $pid 2>$null | Out-Null
                Write-Host "    [OK] Killed" -ForegroundColor Green
            } catch {
                Write-Host "    [WARNING] Could not kill" -ForegroundColor Yellow
            }
        }
    }
} else {
    Write-Host "[INFO] No processes found on port 8080" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Waiting 3 seconds for ports to release..." -ForegroundColor Gray
Start-Sleep -Seconds 3

# Step 2: Verify port is free
Write-Host ""
Write-Host "Step 2: Verifying port 8080 is free..." -ForegroundColor Yellow
$check = netstat -ano | findstr ":8080"
if ($check) {
    Write-Host "[WARNING] Port 8080 is still in use!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Remaining processes:" -ForegroundColor Yellow
    Write-Host $check
    Write-Host ""
    Write-Host "Try running: .\kill_port_8080.ps1" -ForegroundColor Cyan
} else {
    Write-Host "[OK] Port 8080 is free!" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Set OpenAI API key (if needed):" -ForegroundColor White
Write-Host '   $env:OPENAI_API_KEY="sk-proj-your-key-here"' -ForegroundColor Gray
Write-Host ""
Write-Host "2. Start API server:" -ForegroundColor White
Write-Host "   python src/inference.py" -ForegroundColor Gray
Write-Host ""
Write-Host "3. In a NEW terminal, start frontend:" -ForegroundColor White
Write-Host "   python serve_frontend.py" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Open browser:" -ForegroundColor White
Write-Host "   http://localhost:3000/index.html" -ForegroundColor Cyan
Write-Host ""


