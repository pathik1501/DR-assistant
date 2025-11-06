# PowerShell script to verify repository is safe to push to GitHub
# Checks for sensitive files and hardcoded API keys

Write-Host "🔍 Checking repository safety before GitHub push..." -ForegroundColor Cyan
Write-Host ""

# Check if .env exists and is in .gitignore
Write-Host "1. Checking .env file..." -ForegroundColor Yellow
if (Test-Path ".env") {
    $envInGitignore = git check-ignore .env 2>$null
    if ($envInGitignore) {
        Write-Host "   ✅ .env exists and is in .gitignore" -ForegroundColor Green
    } else {
        Write-Host "   ❌ WARNING: .env exists but is NOT in .gitignore!" -ForegroundColor Red
        Write-Host "   Fix: Add .env to .gitignore" -ForegroundColor Red
    }
} else {
    Write-Host "   ℹ️  .env file not found (this is OK)" -ForegroundColor Gray
}

# Check for hardcoded API keys in source files
Write-Host ""
Write-Host "2. Checking for hardcoded API keys..." -ForegroundColor Yellow
$apiKeyPattern = "sk-proj-[a-zA-Z0-9]{48,}"
$foundKeys = @()

Get-ChildItem -Path "src", "frontend" -Recurse -File -Include "*.py" | ForEach-Object {
    $content = Get-Content $_.FullName -Raw
    if ($content -match $apiKeyPattern) {
        $foundKeys += $_.FullName
    }
}

if ($foundKeys.Count -eq 0) {
    Write-Host "   ✅ No hardcoded API keys found in source files" -ForegroundColor Green
} else {
    Write-Host "   ❌ WARNING: Found potential API keys in:" -ForegroundColor Red
    $foundKeys | ForEach-Object {
        Write-Host "      - $_" -ForegroundColor Red
    }
    Write-Host "   Fix: Remove hardcoded keys and use environment variables" -ForegroundColor Red
}

# Check if .env.example exists
Write-Host ""
Write-Host "3. Checking for .env.example..." -ForegroundColor Yellow
if (Test-Path ".env.example") {
    Write-Host "   ✅ .env.example exists (good for deployment)" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  .env.example not found (recommended for deployment)" -ForegroundColor Yellow
}

# Check git status
Write-Host ""
Write-Host "4. Checking git status..." -ForegroundColor Yellow
try {
    $gitStatus = git status --short 2>$null
    if ($gitStatus) {
        Write-Host "   Modified/New files:" -ForegroundColor Cyan
        $gitStatus | ForEach-Object {
            Write-Host "      $_" -ForegroundColor Gray
        }
        
        # Check if .env is in status
        $envInStatus = $gitStatus | Select-String "\.env$"
        if ($envInStatus) {
            Write-Host ""
            Write-Host "   ❌ WARNING: .env file is staged/tracked!" -ForegroundColor Red
            Write-Host "   Fix: Run 'git reset HEAD .env' and ensure .env is in .gitignore" -ForegroundColor Red
        } else {
            Write-Host ""
            Write-Host "   ✅ .env is not in git status" -ForegroundColor Green
        }
    } else {
        Write-Host "   ℹ️  No changes to commit" -ForegroundColor Gray
    }
} catch {
    Write-Host "   ⚠️  Not a git repository (run 'git init' first)" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📋 Summary" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan

$allSafe = $true

if (-not (Test-Path ".env") -or (git check-ignore .env 2>$null)) {
    Write-Host "✅ .env is properly ignored" -ForegroundColor Green
} else {
    Write-Host "❌ .env needs to be added to .gitignore" -ForegroundColor Red
    $allSafe = $false
}

if ($foundKeys.Count -eq 0) {
    Write-Host "✅ No hardcoded API keys found" -ForegroundColor Green
} else {
    Write-Host "❌ Hardcoded API keys found - REMOVE THEM!" -ForegroundColor Red
    $allSafe = $false
}

if ($allSafe) {
    Write-Host ""
    Write-Host "✅ Repository appears safe to push!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Review files: git status" -ForegroundColor White
    Write-Host "2. Add files: git add src/ frontend/ configs/ requirements.txt .gitignore .env.example" -ForegroundColor White
    Write-Host "3. Commit: git commit -m 'Your message'" -ForegroundColor White
    Write-Host "4. Push: git push origin main" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "❌ Repository is NOT safe to push!" -ForegroundColor Red
    Write-Host "Please fix the issues above before pushing." -ForegroundColor Red
}

Write-Host ""

