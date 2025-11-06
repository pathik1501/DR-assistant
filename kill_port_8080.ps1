# Aggressive method to kill processes on port 8080
Write-Host ""
Write-Host "========================================" -ForegroundColor Red
Write-Host "  Killing ALL processes on port 8080" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Red
Write-Host ""

# Method 1: Using netstat and taskkill (most reliable)
Write-Host "Method 1: Using netstat and taskkill..." -ForegroundColor Cyan

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
    foreach ($pid in $uniquePids) {
        $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "Found process on port 8080: $($process.ProcessName) (PID: $pid)" -ForegroundColor Yellow
            
            try {
                # Try graceful stop first
                Stop-Process -Id $pid -Force -ErrorAction Stop
                Write-Host "  [OK] Process $pid killed" -ForegroundColor Green
            } catch {
                # If that fails, use taskkill
                Write-Host "  Trying taskkill for PID $pid..." -ForegroundColor Yellow
                taskkill /F /PID $pid 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "  [OK] Process $pid killed via taskkill" -ForegroundColor Green
                } else {
                    Write-Host "  [ERROR] Could not kill process $pid" -ForegroundColor Red
                }
            }
        }
    }
} else {
    Write-Host "[INFO] No processes found on port 8080 via netstat" -ForegroundColor Gray
}

# Method 2: Kill all Python processes (nuclear option - use carefully)
Write-Host ""
$killPython = Read-Host "Kill ALL Python processes? This will stop ALL Python scripts. (y/n)"
if ($killPython -eq 'y') {
    Write-Host "Killing all Python processes..." -ForegroundColor Yellow
    Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
    Write-Host "[OK] All Python processes killed" -ForegroundColor Green
}

Write-Host ""
Write-Host "Waiting 3 seconds for ports to release..." -ForegroundColor Gray
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "[DONE] Port 8080 should now be free" -ForegroundColor Green
Write-Host ""


