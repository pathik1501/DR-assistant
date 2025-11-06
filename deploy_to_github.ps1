# Complete GitHub Deployment Script
# Safely uploads all code and model checkpoint to GitHub

Write-Host "🚀 GitHub Deployment Script" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

# Check if git is initialized
if (-not (Test-Path ".git")) {
    Write-Host "⚠️  Git repository not initialized" -ForegroundColor Yellow
    Write-Host "Initializing git repository..." -ForegroundColor Yellow
    git init
    Write-Host "✅ Git repository initialized" -ForegroundColor Green
    Write-Host ""
}

# Check for remote
$remote = git remote -v 2>&1
if ($remote -match "origin") {
    Write-Host "✅ Git remote 'origin' found" -ForegroundColor Green
    Write-Host "   $remote" -ForegroundColor Gray
} else {
    Write-Host "⚠️  No remote repository configured" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To add a remote repository, run:" -ForegroundColor Cyan
    Write-Host "   git remote add origin https://github.com/yourusername/dr-assistant.git" -ForegroundColor White
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit 1
    }
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📋 Step 1: Safety Checks" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

# Check for .env file
if (Test-Path ".env") {
    $envInGitignore = git check-ignore .env 2>$null
    if ($envInGitignore) {
        Write-Host "✅ .env file is in .gitignore (safe)" -ForegroundColor Green
    } else {
        Write-Host "❌ WARNING: .env file is NOT in .gitignore!" -ForegroundColor Red
        Write-Host "   This is dangerous - API keys will be exposed!" -ForegroundColor Red
        Write-Host ""
        Write-Host "   Fix: Add .env to .gitignore before continuing" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "ℹ️  .env file not found (OK)" -ForegroundColor Gray
}

# Check for hardcoded API keys
Write-Host "Checking for hardcoded API keys..." -ForegroundColor Yellow
$apiKeyPattern = "sk-proj-[a-zA-Z0-9]{48,}"
$foundKeys = @()

Get-ChildItem -Path "src", "frontend" -Recurse -File -Include "*.py" -ErrorAction SilentlyContinue | ForEach-Object {
    $content = Get-Content $_.FullName -Raw -ErrorAction SilentlyContinue
    if ($content -and $content -match $apiKeyPattern) {
        $foundKeys += $_.FullName
    }
}

if ($foundKeys.Count -eq 0) {
    Write-Host "✅ No hardcoded API keys found" -ForegroundColor Green
} else {
    Write-Host "❌ WARNING: Found potential API keys in:" -ForegroundColor Red
    $foundKeys | ForEach-Object {
        Write-Host "   - $_" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "   Fix: Remove hardcoded keys before continuing" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📦 Step 2: Adding Files" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

# Add core application files
Write-Host "Adding source code..." -ForegroundColor Yellow
git add src/
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Added src/" -ForegroundColor Green
} else {
    Write-Host "⚠️  Warning adding src/" -ForegroundColor Yellow
}

Write-Host "Adding frontend..." -ForegroundColor Yellow
git add frontend/
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Added frontend/" -ForegroundColor Green
} else {
    Write-Host "⚠️  Warning adding frontend/" -ForegroundColor Yellow
}

Write-Host "Adding configs..." -ForegroundColor Yellow
git add configs/
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Added configs/" -ForegroundColor Green
} else {
    Write-Host "⚠️  Warning adding configs/" -ForegroundColor Yellow
}

# Add model checkpoint
Write-Host "Adding model checkpoint..." -ForegroundColor Yellow
$checkpointPath = "1/7b2108ed09bf401fa06ff1b7d8c1e949/checkpoints/dr-model-epoch=60-val_qwk=0.853.ckpt"
if (Test-Path $checkpointPath) {
    git add -f $checkpointPath
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Added model checkpoint (53.87 MB)" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Warning adding checkpoint" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  Checkpoint not found: $checkpointPath" -ForegroundColor Yellow
}

# Add configuration files
Write-Host "Adding configuration files..." -ForegroundColor Yellow
git add requirements.txt .gitignore .env.example
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Added config files" -ForegroundColor Green
} else {
    Write-Host "⚠️  Warning adding config files" -ForegroundColor Yellow
}

# Add documentation
Write-Host "Adding documentation..." -ForegroundColor Yellow
git add README.md DEPLOYMENT_GUIDE.md QUICK_DEPLOY.md GITHUB_READY.md ADD_MODEL_CHECKPOINT.md MODEL_DEPLOYMENT.md
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Added documentation" -ForegroundColor Green
} else {
    Write-Host "⚠️  Warning adding documentation" -ForegroundColor Yellow
}

# Add Docker files
Write-Host "Adding Docker files..." -ForegroundColor Yellow
if (Test-Path "Dockerfile") {
    git add Dockerfile
}
if (Test-Path "docker-compose.yml") {
    git add docker-compose.yml
}
Write-Host "✅ Added Docker files" -ForegroundColor Green

# Add deployment scripts
Write-Host "Adding deployment scripts..." -ForegroundColor Yellow
git add verify_safe_to_push.ps1 include_model_checkpoint.ps1 deploy_to_github.ps1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Added deployment scripts" -ForegroundColor Green
} else {
    Write-Host "⚠️  Warning adding scripts" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📊 Step 3: Review Changes" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

# Show status
Write-Host "Files staged for commit:" -ForegroundColor Yellow
git status --short

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "💾 Step 4: Commit Changes" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

$commitMessage = "Add DR Assistant: RAG pipeline, improved frontend, best model checkpoint (QWK=0.853), and deployment config"

Write-Host "Commit message:" -ForegroundColor Yellow
Write-Host "   $commitMessage" -ForegroundColor Gray
Write-Host ""

$confirm = Read-Host "Commit these changes? (y/n)"
if ($confirm -eq "y") {
    git commit -m $commitMessage
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Changes committed successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Commit failed" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Commit cancelled" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "🚀 Step 5: Push to GitHub" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

# Check current branch
$branch = git branch --show-current 2>&1
if (-not $branch) {
    Write-Host "No branch found. Creating 'main' branch..." -ForegroundColor Yellow
    git branch -M main
    $branch = "main"
}

Write-Host "Current branch: $branch" -ForegroundColor Cyan
Write-Host ""

$push = Read-Host "Push to GitHub? (y/n)"
if ($push -eq "y") {
    Write-Host "Pushing to origin/$branch..." -ForegroundColor Yellow
    
    # Try to push
    git push -u origin $branch 2>&1 | ForEach-Object {
        if ($_ -match "error" -or $_ -match "fatal") {
            Write-Host $_ -ForegroundColor Red
        } else {
            Write-Host $_ -ForegroundColor Gray
        }
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Successfully pushed to GitHub!" -ForegroundColor Green
        Write-Host ""
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
        Write-Host "🎉 Deployment Complete!" -ForegroundColor Green
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Cyan
        Write-Host "1. Set OPENAI_API_KEY in deployment platform" -ForegroundColor White
        Write-Host "2. Deploy using Docker or cloud platform" -ForegroundColor White
        Write-Host "3. See DEPLOYMENT_GUIDE.md for details" -ForegroundColor White
    } else {
        Write-Host ""
        Write-Host "⚠️  Push may have failed or remote not configured" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "To set up remote:" -ForegroundColor Cyan
        Write-Host "   git remote add origin https://github.com/yourusername/dr-assistant.git" -ForegroundColor White
        Write-Host "   git push -u origin $branch" -ForegroundColor White
    }
} else {
    Write-Host "Push cancelled. Run 'git push' manually when ready." -ForegroundColor Yellow
}

Write-Host ""

