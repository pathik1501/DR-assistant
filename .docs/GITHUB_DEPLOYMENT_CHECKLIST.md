# GitHub Upload & Deployment Checklist

## ✅ Security Checks (CRITICAL - Do First!)

### 1. Remove API Keys from Documentation
- [ ] Check `START_API_SERVER.md` - Remove any real API keys
- [ ] Check `QUICK_START_API.md` - Remove any real API keys  
- [ ] Check `START_INSTRUCTIONS.md` - Remove any real API keys
- [ ] Search all `.md` files for `sk-proj-` pattern
- [ ] Replace with placeholder: `sk-proj-your-key-here`

### 2. Verify .gitignore
- [x] `.env` is in .gitignore ✅
- [x] `*.env` files are ignored ✅
- [x] `data/vector_db/` is ignored ✅
- [x] Model checkpoints are ignored ✅
- [x] Logs and outputs are ignored ✅

### 3. Check Code Files
- [x] No hardcoded API keys in Python files ✅
- [x] All use `os.getenv('OPENAI_API_KEY')` ✅
- [x] No API keys in PowerShell scripts ✅

## 📦 Files to Commit

### Core Application Files
- [x] `src/` - All Python source files
- [x] `frontend/` - Frontend application files
- [x] `configs/config.yaml` - Configuration (no secrets)
- [x] `requirements.txt` - Dependencies
- [x] `.env.example` - Environment template (NEW)

### Documentation
- [x] `README.md` - Main documentation
- [x] `TRAINING_GUIDE.md` - Training instructions
- [x] `DEPLOYMENT_GUIDE.md` - Deployment guide
- [ ] Clean documentation files (remove real API keys)

### Scripts
- [x] `*.ps1` - PowerShell scripts (no API keys)
- [x] `setup.py` - Setup script
- [x] `simple_setup.py` - Simple setup

### Configuration
- [x] `.gitignore` - Git ignore rules
- [x] `Dockerfile` - Docker configuration
- [x] `docker-compose.yml` - Docker compose
- [x] `.env.example` - Environment template

### Tests
- [x] `tests/` - Test files

## ❌ Files NOT to Commit

- [ ] `.env` - Contains secrets (already in .gitignore)
- [ ] `data/` - Large data files (already in .gitignore)
- [ ] `models/` - Model checkpoints (already in .gitignore)
- [ ] `outputs/` - Output files (already in .gitignore)
- [ ] `mlruns/` - MLflow runs (already in .gitignore)
- [ ] `__pycache__/` - Python cache (already in .gitignore)
- [ ] `*.ckpt`, `*.pth` - Model files (already in .gitignore)

## 🚀 Pre-Deployment Steps

### 1. Clean Documentation
```powershell
# Search for real API keys in documentation
Select-String -Path "*.md" -Pattern "sk-proj-[a-zA-Z0-9]{20,}" | Select-Object Path, LineNumber
```

### 2. Verify No Secrets
```powershell
# Check for any remaining API keys
Select-String -Path "*.py","*.ps1","*.yaml","*.md" -Pattern "sk-proj-[a-zA-Z0-9]{20,}"
```

### 3. Test Locally
- [ ] API server starts without errors
- [ ] Frontend connects to API
- [ ] Predictions work
- [ ] RAG features work (if API key set)

## 📤 GitHub Upload Commands

```powershell
# 1. Check git status
git status

# 2. Add files (excluding .gitignore items)
git add .

# 3. Commit
git commit -m "Initial commit: DR Assistant with RAG pipeline and compact UI"

# 4. Push to GitHub
git push origin main
```

## 🌐 Deployment Options

### Option 1: Docker Deployment
```bash
# Build and run with Docker
docker-compose up -d
```

### Option 2: Cloud Platform (Heroku, Railway, etc.)
- Set environment variables in platform dashboard
- Deploy using platform-specific commands

### Option 3: VPS/Server
- Clone repository
- Install dependencies: `pip install -r requirements.txt`
- Set environment variables
- Run: `python src/inference.py`

## 🔐 Environment Variables for Deployment

Set these in your deployment platform:
- `OPENAI_API_KEY` - Your OpenAI API key
- `API_HOST` - Usually `0.0.0.0` for servers
- `API_PORT` - Usually `8080`

## ✅ Final Checklist Before Push

- [ ] All API keys removed from code
- [ ] All API keys removed from documentation
- [ ] `.env.example` created
- [ ] `.gitignore` verified
- [ ] No secrets in committed files
- [ ] README.md updated
- [ ] All tests pass
- [ ] Application works locally

