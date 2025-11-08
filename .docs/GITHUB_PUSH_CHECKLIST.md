# GitHub Push Checklist

## ✅ Pre-Push Security Check

### 1. Verify No Sensitive Files
```bash
# Check if .env is tracked (should return nothing)
git ls-files | grep -E "\.env$|\.env\.|secrets|api.*key"

# Check for hardcoded API keys in code
grep -r "sk-proj-" src/ frontend/ --exclude-dir=__pycache__
```

### 2. Files to Commit

#### ✅ Safe to Commit
- `src/` - All source code (no hardcoded keys)
- `frontend/` - Frontend code
- `configs/config.yaml` - Configuration (no secrets)
- `requirements.txt` - Dependencies
- `.gitignore` - Git ignore rules
- `.env.example` - Environment template
- `README.md` - Documentation
- `DEPLOYMENT_GUIDE.md` - Deployment instructions
- `Dockerfile` - Docker configuration
- `docker-compose.yml` - Docker compose config

#### ❌ Never Commit
- `.env` - Contains API keys (in .gitignore ✅)
- `data/vector_db/` - Vector database (in .gitignore ✅)
- `*.ckpt`, `*.pth` - Model checkpoints (in .gitignore ✅)
- `mlflow.db`, `mlruns/` - MLflow data (in .gitignore ✅)
- `__pycache__/` - Python cache (in .gitignore ✅)
- `outputs/` - Training outputs (in .gitignore ✅)

## 📝 Git Commands

### Check Status
```bash
git status
```

### Add Files
```bash
# Add all safe files
git add src/
git add frontend/
git add configs/
git add requirements.txt
git add .gitignore
git add .env.example
git add README.md
git add DEPLOYMENT_GUIDE.md
git add Dockerfile
git add docker-compose.yml
```

### Commit
```bash
git commit -m "Add DR Assistant: RAG pipeline, frontend improvements, and deployment config"
```

### Push
```bash
git push origin main
# or
git push origin master
```

## 🔍 Final Verification

Before pushing, verify:
1. ✅ `.env` is NOT in `git status` output
2. ✅ No API keys in source code (checked with grep)
3. ✅ `.env.example` exists as template
4. ✅ `.gitignore` includes all sensitive files
5. ✅ All relevant code files are staged

## 🚨 If You Accidentally Committed .env

If `.env` was committed:
```bash
# Remove from git (but keep local file)
git rm --cached .env

# Add to .gitignore (already there)
# Then commit the removal
git commit -m "Remove .env from tracking"

# If already pushed, you need to:
# 1. Rotate your API key immediately
# 2. Force push (dangerous - coordinate with team)
# 3. Or use git filter-branch to remove from history
```

## 📦 What Gets Deployed

When you push to GitHub:
- ✅ All source code
- ✅ Configuration files (no secrets)
- ✅ Documentation
- ✅ Docker files
- ❌ No API keys
- ❌ No model checkpoints (too large)
- ❌ No vector database (can be regenerated)

## 🔐 Environment Variables for Deployment

For deployment platforms, set these environment variables:
- `OPENAI_API_KEY` - Required for RAG features
- `API_PORT` - Optional (default: 8080)
- `FRONTEND_PORT` - Optional (default: 8501)
