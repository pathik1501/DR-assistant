# Push DR Assistant to GitHub
# Usage: .\push_to_github.ps1 -RepositoryUrl "https://github.com/username/repo-name.git"

param(
    [Parameter(Mandatory=$true)]
    [string]$RepositoryUrl
)

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Pushing to GitHub" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

cd "C:\Users\pathi\Documents\DR assistant"

# Check if remote already exists
$existingRemote = git remote get-url origin 2>$null
if ($existingRemote) {
    Write-Host "Remote 'origin' already exists: $existingRemote" -ForegroundColor Yellow
    $update = Read-Host "Update to new URL? (y/n)"
    if ($update -eq 'y') {
        git remote set-url origin $RepositoryUrl
        Write-Host "[OK] Remote updated" -ForegroundColor Green
    } else {
        $RepositoryUrl = $existingRemote
        Write-Host "Using existing remote: $RepositoryUrl" -ForegroundColor Cyan
    }
} else {
    Write-Host "Adding remote repository..." -ForegroundColor Yellow
    git remote add origin $RepositoryUrl
    Write-Host "[OK] Remote added" -ForegroundColor Green
}

# Check current branch
$currentBranch = git branch --show-current
if ($currentBranch -ne "main") {
    Write-Host "Renaming branch to 'main'..." -ForegroundColor Yellow
    git branch -M main
}

# Verify what will be pushed
Write-Host ""
Write-Host "Files ready to push:" -ForegroundColor Cyan
$fileCount = (git ls-files | Measure-Object).Count
Write-Host "  Total files: $fileCount" -ForegroundColor White

Write-Host ""
Write-Host "Commit info:" -ForegroundColor Cyan
git log --oneline -1

Write-Host ""
$confirm = Read-Host "Push to GitHub? (y/n)"
if ($confirm -eq 'y') {
    Write-Host ""
    Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
    git push -u origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "[SUCCESS] Code pushed to GitHub!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Repository URL: $RepositoryUrl" -ForegroundColor Cyan
    } else {
        Write-Host ""
        Write-Host "[ERROR] Push failed. Check the error above." -ForegroundColor Red
        Write-Host ""
        Write-Host "Common issues:" -ForegroundColor Yellow
        Write-Host "  1. Repository doesn't exist - create it on GitHub first" -ForegroundColor White
        Write-Host "  2. Authentication required - use GitHub credentials or SSH key" -ForegroundColor White
        Write-Host "  3. Branch name mismatch - ensure repository uses 'main' branch" -ForegroundColor White
    }
} else {
    Write-Host "Push cancelled." -ForegroundColor Yellow
}

Write-Host ""



