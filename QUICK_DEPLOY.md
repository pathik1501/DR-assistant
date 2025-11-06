# Quick Deployment Guide

## 🚀 Quick Start for Deployment

### 1. Verify Safety (Run First!)
```powershell
.\verify_safe_to_push.ps1
```

This checks:
- ✅ `.env` is in `.gitignore`
- ✅ No hardcoded API keys
- ✅ `.env.example` exists
- ✅ No sensitive files staged

### 2. Initialize Git (if not already done)
```bash
git init
git remote add origin https://github.com/yourusername/dr-assistant.git
```

### 3. Add Safe Files
```bash
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

### 4. Commit
```bash
git commit -m "Add DR Assistant: RAG pipeline, frontend, and deployment config"
```

### 5. Push to GitHub
```bash
git push -u origin main
```

## 🔐 Environment Variables for Deployment

### For Local Development
1. Copy `.env.example` to `.env`
2. Add your OpenAI API key

### For Production Deployment

**Heroku:**
```bash
heroku config:set OPENAI_API_KEY=your-key-here
```

**Railway:**
- Add in dashboard: Settings → Environment Variables

**Docker:**
```bash
docker run -e OPENAI_API_KEY=your-key-here ...
```

**VPS/Server:**
- Create `.env` file with production values
- Never commit it!

## 📦 What Gets Deployed

✅ **Included:**
- Source code (`src/`, `frontend/`)
- Configuration files (no secrets)
- Documentation
- Docker files
- Requirements

❌ **Excluded (in .gitignore):**
- `.env` files
- API keys
- Model checkpoints
- Vector database
- Training outputs

## 🐳 Docker Deployment

```bash
# Build
docker build -t dr-assistant .

# Run
docker run -p 8080:8080 -p 8501:8501 -e OPENAI_API_KEY=your-key dr-assistant
```

## ☁️ Cloud Platform Deployment

### Heroku
```bash
heroku create dr-assistant
heroku config:set OPENAI_API_KEY=your-key
git push heroku main
```

### Railway
1. Connect GitHub repo
2. Add `OPENAI_API_KEY` in environment variables
3. Deploy automatically

### AWS/GCP/Azure
- Use container services
- Set environment variables
- Deploy container image

## ✅ Post-Deployment Checklist

- [ ] API health check: `curl https://your-domain.com/health`
- [ ] Frontend accessible
- [ ] Test image upload works
- [ ] RAG pipeline initializes
- [ ] No errors in logs

## 🔒 Security Reminders

1. **Never commit `.env`** - Already in `.gitignore` ✅
2. **Rotate keys if exposed** - Regenerate immediately
3. **Use environment variables** - No hardcoded keys ✅
4. **Review commits** - Check `git diff` before pushing

## 📚 Full Documentation

See `DEPLOYMENT_GUIDE.md` for detailed instructions.

