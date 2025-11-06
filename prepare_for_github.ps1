# Prepare for GitHub Upload
# This script helps ensure no secrets are committed

Write-Host "=== Preparing for GitHub Upload ===" -ForegroundColor Cyan
Write-Host ""

# 1. Check for API keys in files
Write-Host "1. Checking for API keys in code files..." -ForegroundColor Yellow
$apiKeys = Select-String -Path "*.py","*.ps1","*.yaml" -Pattern "sk-proj-[a-zA-Z0-9]{20,}" -ErrorAction SilentlyContinue
if ($apiKeys) {
    Write-Host "   ⚠️  WARNING: Found potential API keys in:" -ForegroundColor Red
    $apiKeys | ForEach-Object { Write-Host "      $($_.Path):$($_.LineNumber)" -ForegroundColor Red }
    Write-Host "   Please remove these before committing!" -ForegroundColor Red
} else {
    Write-Host "   ✅ No API keys found in code files" -ForegroundColor Green
}

# 2. Check documentation files
Write-Host ""
Write-Host "2. Checking documentation files..." -ForegroundColor Yellow
$docKeys = Select-String -Path "*.md" -Pattern "sk-proj-[a-zA-Z0-9]{20,}" -ErrorAction SilentlyContinue
if ($docKeys) {
    Write-Host "   ⚠️  WARNING: Found API keys in documentation:" -ForegroundColor Red
    $docKeys | ForEach-Object { Write-Host "      $($_.Path):$($_.LineNumber)" -ForegroundColor Red }
    Write-Host "   Please replace with placeholders!" -ForegroundColor Red
} else {
    Write-Host "   ✅ Documentation files are clean" -ForegroundColor Green
}

# 3. Verify .gitignore
Write-Host ""
Write-Host "3. Verifying .gitignore..." -ForegroundColor Yellow
if (Test-Path ".gitignore") {
    $gitignore = Get-Content ".gitignore" -Raw
    if ($gitignore -match "\.env") {
        Write-Host "   ✅ .env is in .gitignore" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  .env is NOT in .gitignore!" -ForegroundColor Red
    }
} else {
    Write-Host "   ⚠️  .gitignore file not found!" -ForegroundColor Red
}

# 4. Check if .env exists
Write-Host ""
Write-Host "4. Checking for .env file..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "   ✅ .env file exists (will be ignored by git)" -ForegroundColor Green
} else {
    Write-Host "   ℹ️  .env file not found (create from .env.example)" -ForegroundColor Cyan
}

# 5. Check if .env.example exists
Write-Host ""
Write-Host "5. Checking for .env.example..." -ForegroundColor Yellow
if (Test-Path ".env.example") {
    Write-Host "   ✅ .env.example exists" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  .env.example not found! Creating..." -ForegroundColor Yellow
    # Create .env.example
    @"
# Diabetic Retinopathy Assistant Environment Variables
# Copy this file to .env and fill in your actual values

# OpenAI API Key (required for RAG pipeline)
# Get your key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your-openai-api-key-here

# MLflow Configuration
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_DEFAULT_ARTIFACT_ROOT=./outputs/mlflow_artifacts

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
"@ | Out-File -FilePath ".env.example" -Encoding UTF8
    Write-Host "   ✅ Created .env.example" -ForegroundColor Green
}

# 6. Summary
Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Before pushing to GitHub:" -ForegroundColor Yellow
Write-Host "1. Remove any API keys from code/documentation" -ForegroundColor White
Write-Host "2. Verify .gitignore includes .env" -ForegroundColor White
Write-Host "3. Ensure .env.example exists" -ForegroundColor White
Write-Host "4. Test that application works" -ForegroundColor White
Write-Host ""
Write-Host "Ready to commit? Run:" -ForegroundColor Cyan
Write-Host "  git add ." -ForegroundColor White
Write-Host "  git commit -m 'Your commit message'" -ForegroundColor White
Write-Host "  git push origin main" -ForegroundColor White

