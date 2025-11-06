# Deployment Guide for DR Assistant

## 🚀 Pre-Deployment Checklist

### 1. Security Check ✅
- [x] `.env` file is in `.gitignore`
- [x] No API keys hardcoded in source files
- [x] `.env.example` created for reference
- [x] All sensitive files excluded

### 2. Files to Commit

#### Core Application Files
```
src/
├── inference.py          # FastAPI backend
├── rag_pipeline.py      # RAG pipeline for explanations
├── model.py             # Model architecture
├── explainability.py    # Grad-CAM visualizations
├── train.py             # Training script
└── ...

frontend/
├── app_new.py           # Streamlit frontend
└── ...

configs/
└── config.yaml          # Configuration (no secrets)

requirements.txt         # Python dependencies
.gitignore              # Git ignore rules
.env.example            # Environment template
README.md               # Project documentation
```

#### Files to EXCLUDE (already in .gitignore)
- `.env` - Contains API keys
- `data/vector_db/` - Vector database (can be regenerated)
- `*.ckpt`, `*.pth` - Model checkpoints (too large)
- `mlflow.db`, `mlruns/` - MLflow data
- `__pycache__/` - Python cache
- `outputs/` - Training outputs

### 3. Environment Setup

#### For Local Development
1. Copy `.env.example` to `.env`
2. Add your OpenAI API key to `.env`
3. Install dependencies: `pip install -r requirements.txt`
4. Run API: `python src/inference.py`
5. Run Frontend: `streamlit run frontend/app_new.py`

#### For Production Deployment

**Option 1: Docker (Recommended)**
```bash
# Build and run with docker-compose
docker-compose up -d
```

**Option 2: Cloud Platform (Heroku, Railway, etc.)**
1. Set environment variables in platform dashboard
2. Deploy using platform's deployment method
3. Ensure `OPENAI_API_KEY` is set as environment variable

**Option 3: VPS/Server**
1. Clone repository
2. Create `.env` file with production values
3. Use systemd or supervisor to run services
4. Set up reverse proxy (nginx) for production

## 📋 Deployment Steps

### Step 1: Prepare Repository
```bash
# Check what will be committed
git status

# Ensure .env is not tracked
git check-ignore .env

# Add files
git add src/ frontend/ configs/ requirements.txt .gitignore .env.example README.md
```

### Step 2: Commit Changes
```bash
git commit -m "Add DR Assistant application with RAG pipeline and frontend"
```

### Step 3: Push to GitHub
```bash
git push origin main
```

### Step 4: Set Up Environment Variables

**On GitHub (for CI/CD):**
- Go to Settings → Secrets and variables → Actions
- Add `OPENAI_API_KEY` as a secret

**On Deployment Platform:**
- Add `OPENAI_API_KEY` in environment variables section
- Never commit `.env` file

## 🔒 Security Best Practices

1. **Never commit `.env` files**
   - Already in `.gitignore` ✅
   - Use `.env.example` as template ✅

2. **Use Environment Variables**
   - All sensitive data loaded from environment
   - No hardcoded keys in source code ✅

3. **Rotate API Keys**
   - If a key is exposed, regenerate it immediately
   - Update `.env` file with new key

4. **Review Commits**
   - Before pushing, check: `git diff` to ensure no secrets

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t dr-assistant:latest .
```

### Run with Environment Variables
```bash
docker run -d \
  -p 8080:8080 \
  -p 8501:8501 \
  -e OPENAI_API_KEY=your-key-here \
  dr-assistant:latest
```

### Or use docker-compose.yml
```bash
# Edit docker-compose.yml to add environment variables
# Then run:
docker-compose up -d
```

## ☁️ Cloud Platform Deployment

### Heroku
```bash
heroku create dr-assistant
heroku config:set OPENAI_API_KEY=your-key-here
git push heroku main
```

### Railway
1. Connect GitHub repository
2. Add `OPENAI_API_KEY` in environment variables
3. Deploy automatically on push

### AWS/GCP/Azure
1. Use container services (ECS, Cloud Run, Container Instances)
2. Set environment variables in platform
3. Deploy container image

## 📝 Post-Deployment

1. **Test API Health**
   ```bash
   curl https://your-domain.com/health
   ```

2. **Test Frontend**
   - Open frontend URL
   - Upload test image
   - Verify predictions work

3. **Monitor Logs**
   - Check for errors
   - Verify RAG pipeline initializes
   - Monitor API response times

## 🔧 Troubleshooting

### API Key Not Working
- Verify `.env` file exists and has correct key
- Check environment variables are loaded
- Restart server after changing `.env`

### Vector Database Missing
- RAG pipeline will create it automatically
- Requires OpenAI API key with quota
- First run may take a few minutes

### Port Conflicts
- Change ports in `configs/config.yaml`
- Or set `API_PORT` and `FRONTEND_PORT` in `.env`

## 📚 Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Streamlit Deployment](https://docs.streamlit.io/deploy)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
