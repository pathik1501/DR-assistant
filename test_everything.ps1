# Complete diagnostic script
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DR Assistant - Complete Diagnostic" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check 1: Port 8080
Write-Host "1. Checking port 8080..." -ForegroundColor Yellow
$port8080 = netstat -ano | findstr ":8080"
if ($port8080) {
    Write-Host "   [WARNING] Port 8080 is in use!" -ForegroundColor Red
    Write-Host $port8080
} else {
    Write-Host "   [OK] Port 8080 is free" -ForegroundColor Green
}

# Check 2: Port 3000
Write-Host ""
Write-Host "2. Checking port 3000..." -ForegroundColor Yellow
$port3000 = netstat -ano | findstr ":3000"
if ($port3000) {
    Write-Host "   [WARNING] Port 3000 is in use!" -ForegroundColor Red
    Write-Host $port3000
} else {
    Write-Host "   [OK] Port 3000 is free" -ForegroundColor Green
}

# Check 3: Files
Write-Host ""
Write-Host "3. Checking required files..." -ForegroundColor Yellow
$files = @(
    "src\inference.py",
    "serve_frontend.py",
    "frontend\index.html",
    "frontend\app.js",
    "frontend\style.css"
)

foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "   [OK] $file" -ForegroundColor Green
    } else {
        Write-Host "   [MISSING] $file" -ForegroundColor Red
    }
}

# Check 4: Model checkpoint
Write-Host ""
Write-Host "4. Checking model checkpoint..." -ForegroundColor Yellow
$checkpoint = "1\7d0928bb87954a739123ca35fa03cccf\checkpoints\dr-model-epoch=11-val_qwk=0.769.ckpt"
if (Test-Path $checkpoint) {
    Write-Host "   [OK] Model checkpoint exists" -ForegroundColor Green
} else {
    Write-Host "   [WARNING] Model checkpoint not found" -ForegroundColor Yellow
    Write-Host "   Server will use pretrained model (less accurate)" -ForegroundColor Gray
}

# Check 5: Python dependencies
Write-Host ""
Write-Host "5. Checking Python dependencies..." -ForegroundColor Yellow
try {
    $fastapi = python -c "import fastapi; print('FastAPI OK')" 2>&1
    if ($fastapi -like "*OK*") {
        Write-Host "   [OK] FastAPI installed" -ForegroundColor Green
    } else {
        Write-Host "   [ERROR] FastAPI check failed" -ForegroundColor Red
    }
} catch {
    Write-Host "   [ERROR] FastAPI not installed" -ForegroundColor Red
}

try {
    $streamlit = python -c "import streamlit; print('Streamlit OK')" 2>&1
    if ($streamlit -like "*OK*") {
        Write-Host "   [OK] Streamlit installed" -ForegroundColor Green
    } else {
        Write-Host "   [WARNING] Streamlit check failed (not needed for standalone frontend)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   [WARNING] Streamlit not installed (not needed for standalone frontend)" -ForegroundColor Yellow
}

# Check 6: API health
Write-Host ""
Write-Host "6. Testing API connection..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "   [OK] API server is running!" -ForegroundColor Green
    Write-Host "   Response: $($response.Content)" -ForegroundColor Gray
} catch {
    Write-Host "   [INFO] API server is not running (this is OK if you haven't started it)" -ForegroundColor Gray
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Summary & Next Steps" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "To start the system:" -ForegroundColor White
Write-Host ""
Write-Host "1. Start API server:" -ForegroundColor Cyan
Write-Host '   $env:OPENAI_API_KEY="sk-proj-your-key-here"' -ForegroundColor Gray
Write-Host "   python src/inference.py" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Start Frontend (NEW terminal):" -ForegroundColor Cyan
Write-Host "   python serve_frontend.py" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Open browser:" -ForegroundColor Cyan
Write-Host "   http://localhost:3000/index.html" -ForegroundColor Gray
Write-Host ""



