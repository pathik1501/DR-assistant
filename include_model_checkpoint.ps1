# Script to include model checkpoint in Git repository
# Handles both small files (<100MB) and large files (Git LFS)

param(
    [string]$CheckpointPath = "1/7b2108ed09bf401fa06ff1b7d8c1e949/checkpoints/dr-model-epoch=60-val_qwk=0.853.ckpt"
)

Write-Host "🔍 Checking model checkpoint..." -ForegroundColor Cyan
Write-Host ""

# Check if checkpoint exists
if (-not (Test-Path $CheckpointPath)) {
    Write-Host "❌ Checkpoint not found at: $CheckpointPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Searching for checkpoints..." -ForegroundColor Yellow
    
    # Search for checkpoints
    $checkpoints = Get-ChildItem -Path "." -Recurse -Filter "*.ckpt" -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "*qwk*" -or $_.Name -like "*best*" }
    
    if ($checkpoints) {
        Write-Host "Found checkpoints:" -ForegroundColor Green
        $checkpoints | ForEach-Object {
            $sizeMB = [math]::Round($_.Length/1MB, 2)
            Write-Host "  - $($_.FullName) ($sizeMB MB)" -ForegroundColor Gray
        }
        Write-Host ""
        Write-Host "Please specify the correct path." -ForegroundColor Yellow
    } else {
        Write-Host "No checkpoints found." -ForegroundColor Red
    }
    exit 1
}

# Get file size
$file = Get-Item $CheckpointPath
$sizeMB = [math]::Round($file.Length/1MB, 2)

Write-Host "✅ Checkpoint found: $CheckpointPath" -ForegroundColor Green
Write-Host "   Size: $sizeMB MB" -ForegroundColor Cyan
Write-Host ""

# Check size and recommend approach
if ($sizeMB -lt 50) {
    Write-Host "✅ File size is under 50MB - Can upload directly to GitHub" -ForegroundColor Green
    $useLFS = $false
} elseif ($sizeMB -lt 100) {
    Write-Host "⚠️  File size is 50-100MB - Can upload but GitHub will warn" -ForegroundColor Yellow
    Write-Host "   Recommendation: Use Git LFS for better performance" -ForegroundColor Yellow
    $useLFS = $true
} else {
    Write-Host "❌ File size is over 100MB - Must use Git LFS" -ForegroundColor Red
    $useLFS = $true
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📋 Setup Instructions" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

if ($useLFS) {
    Write-Host "Option 1: Use Git LFS (Recommended for large files)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. Install Git LFS:" -ForegroundColor White
    Write-Host "   Download from: https://git-lfs.github.com/" -ForegroundColor Gray
    Write-Host "   Or: winget install Git.GitLFS" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Initialize Git LFS:" -ForegroundColor White
    Write-Host "   git lfs install" -ForegroundColor Gray
    Write-Host ""
    Write-Host "3. Track .ckpt files:" -ForegroundColor White
    Write-Host "   git lfs track `"*.ckpt`"" -ForegroundColor Gray
    Write-Host "   git lfs track `"1/**/*.ckpt`"" -ForegroundColor Gray
    Write-Host ""
    Write-Host "4. Add .gitattributes:" -ForegroundColor White
    Write-Host "   git add .gitattributes" -ForegroundColor Gray
    Write-Host ""
    Write-Host "5. Add checkpoint:" -ForegroundColor White
    Write-Host "   git add -f `"$CheckpointPath`"" -ForegroundColor Gray
    Write-Host ""
    Write-Host "6. Commit and push:" -ForegroundColor White
    Write-Host "   git commit -m `"Add best model checkpoint via Git LFS`"" -ForegroundColor Gray
    Write-Host "   git push origin main" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host "Option 1: Direct Upload (File < 50MB)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. Update .gitignore to allow this specific file:" -ForegroundColor White
    Write-Host "   Add exception: !$CheckpointPath" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Add checkpoint:" -ForegroundColor White
    Write-Host "   git add -f `"$CheckpointPath`"" -ForegroundColor Gray
    Write-Host ""
    Write-Host "3. Commit and push:" -ForegroundColor White
    Write-Host "   git commit -m `"Add best model checkpoint`"" -ForegroundColor Gray
    Write-Host "   git push origin main" -ForegroundColor Gray
    Write-Host ""
}

Write-Host "Option 2: Host Separately (Alternative)" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Upload checkpoint to cloud storage (Google Drive, S3, etc.)" -ForegroundColor White
Write-Host "2. Create download_model.py script" -ForegroundColor White
Write-Host "3. Download during deployment" -ForegroundColor White
Write-Host ""
Write-Host "See MODEL_DEPLOYMENT.md for details" -ForegroundColor Gray
Write-Host ""

