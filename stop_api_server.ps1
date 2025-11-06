# Stop existing API servers on port 8080
Write-Host ""
Write-Host "Stopping existing API servers on port 8080..." -ForegroundColor Yellow

# Find processes using port 8080
$processes = Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique

if ($processes) {
    foreach ($processId in $processes) {
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "Stopping process: $($process.ProcessName) (PID: $processId)" -ForegroundColor Cyan
            try {
                Stop-Process -Id $processId -Force -ErrorAction Stop
                Write-Host "  [OK] Process $processId stopped" -ForegroundColor Green
            } catch {
                Write-Host "  [WARNING] Failed to stop process $processId : $_" -ForegroundColor Yellow
            }
        }
    }
    Write-Host ""
    Write-Host "[OK] Stopped processes on port 8080" -ForegroundColor Green
    Start-Sleep -Seconds 3
} else {
    Write-Host "[INFO] No processes found on port 8080" -ForegroundColor Gray
}

# Also try to kill any Python processes running inference.py
Write-Host ""
Write-Host "Checking for Python processes running inference.py..." -ForegroundColor Yellow
$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*inference.py*" -or $_.Path -like "*python*"
}

if ($pythonProcesses) {
    foreach ($proc in $pythonProcesses) {
        try {
            $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($proc.Id)").CommandLine
            if ($cmdLine -like "*inference.py*") {
                Write-Host "Stopping Python process: PID $($proc.Id)" -ForegroundColor Cyan
                Stop-Process -Id $proc.Id -Force -ErrorAction Stop
                Write-Host "  [OK] Process $($proc.Id) stopped" -ForegroundColor Green
            }
        } catch {
            Write-Host "  [WARNING] Could not check/stop process $($proc.Id)" -ForegroundColor Yellow
        }
    }
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "Done. Port 8080 should now be available." -ForegroundColor Green
Write-Host ""

