# Simple Deployment Guide

## 🎯 Choose Your Method

### Method 1: Docker (Easiest - 3 Steps) ⭐
### Method 2: Cloud Platform (No Server Setup) ☁️
### Method 3: Local Python (For Development) 💻

---

## 🐳 Method 1: Docker Deployment (Recommended)

### Prerequisites
- Docker Desktop installed: https://www.docker.com/get-started
- OpenAI API key: https://platform.openai.com/api-keys

### Step 1: Clone Repository
```bash
git clone https://github.com/pathik1501/DR-assistant.git
cd DR-assistant
```

### Step 2: Setup Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your OpenAI API key
# Windows: notepad .env
# Mac/Linux: nano .env
```

Add this line to `.env`:
```
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

### Step 3: Deploy
```bash
# Windows PowerShell
.\deploy.ps1

# Mac/Linux
chmod +x deploy.sh
./deploy.sh

# Or manually:
docker-compose up -d
```

### ✅ Done!
- **Frontend:** http://localhost:8501
- **API:** http://localhost:8080
- **API Docs:** http://localhost:8080/docs

### Useful Commands
```bash
# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Restart services
docker-compose restart

# Rebuild after code changes
docker-compose up -d --build
```

---

## ☁️ Method 2: Cloud Platform (Railway)

### Step 1: Sign Up
1. Go to https://railway.app
2. Click "Start a New Project"
3. Sign up with GitHub

### Step 2: Deploy
1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose: `pathik1501/DR-assistant`
4. Railway will auto-detect and deploy

### Step 3: Add API Key
1. Click on your project
2. Go to "Variables" tab
3. Click "New Variable"
4. Add:
   - **Name:** `OPENAI_API_KEY`
   - **Value:** `sk-proj-your-actual-key-here`
5. Click "Add"

### Step 4: Access
Railway provides a public URL automatically (e.g., `https://dr-assistant.up.railway.app`)

### ✅ Done!

---

## ☁️ Method 2b: Cloud Platform (Render)

### Step 1: Sign Up
1. Go to https://render.com
2. Sign up with GitHub

### Step 2: Create Web Service
1. Click "New +" → "Web Service"
2. Connect GitHub repository: `pathik1501/DR-assistant`
3. Configure:
   - **Name:** `dr-assistant`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python src/inference.py`
   - **Port:** `8080`

### Step 3: Add Environment Variable
1. Scroll to "Environment Variables"
2. Add:
   - **Key:** `OPENAI_API_KEY`
   - **Value:** `sk-proj-your-actual-key-here`
3. Click "Save Changes"

### Step 4: Deploy
Click "Create Web Service"
Render will build and deploy automatically

### ✅ Done!
Render provides a URL like: `https://dr-assistant.onrender.com`

---

## 💻 Method 3: Local Python (Development)

### Step 1: Install Python
- Python 3.8+ required
- Download: https://www.python.org/downloads/

### Step 2: Clone Repository
```bash
git clone https://github.com/pathik1501/DR-assistant.git
cd DR-assistant
```

### Step 3: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Setup Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
# Windows: notepad .env
# Mac/Linux: nano .env
```

Add:
```
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

### Step 6: Run API Server
```bash
python src/inference.py
```

API will run on: http://localhost:8080

### Step 7: Run Frontend (New Terminal)
```bash
# Activate virtual environment again
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

streamlit run frontend/app_new.py
```

Frontend will run on: http://localhost:8501

### ✅ Done!

---

## 🔍 Verify Deployment

### Check API Health
```bash
# Browser
http://localhost:8080/health

# Command line
curl http://localhost:8080/health

# Should return: {"status": "healthy", "timestamp": ...}
```

### Test Application
1. Open frontend: http://localhost:8501
2. Upload a retinal fundus image
3. Click "Analyze Image"
4. Verify predictions work

---

## 🐛 Troubleshooting

### Issue: Port Already in Use
```bash
# Find what's using the port
# Windows
netstat -ano | findstr :8080

# Mac/Linux
lsof -i :8080

# Kill the process or change port in configs/config.yaml
```

### Issue: API Key Not Working
1. Verify `.env` file has correct key
2. Restart services after changing `.env`
3. Check logs for errors:
   ```bash
   docker-compose logs -f
   ```

### Issue: Model Not Loading
1. Verify checkpoint exists:
   ```bash
   ls 1/7b2108ed09bf401fa06ff1b7d8c1e949/checkpoints/
   ```
2. If missing, pull from GitHub:
   ```bash
   git pull origin main
   ```

### Issue: RAG Pipeline Not Working
1. Check OpenAI API key has quota/credits
2. Verify key is set correctly in `.env`
3. Check logs for quota errors
4. Wait 5-10 minutes if you just added billing

---

## 📋 Quick Reference

### Docker Commands
```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f

# Restart
docker-compose restart

# Rebuild
docker-compose up -d --build
```

### Local Python Commands
```bash
# Activate virtual environment
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# Run API
python src/inference.py

# Run Frontend (new terminal)
streamlit run frontend/app_new.py
```

---

## 🎯 Next Steps After Deployment

1. **Test the application** - Upload test images
2. **Monitor logs** - Check for errors
3. **Set up SSL** - For production (HTTPS)
4. **Configure domain** - Point your domain to the server
5. **Set up monitoring** - Use Prometheus/Grafana (included in docker-compose)

---

## 📚 More Information

- **Detailed Guide:** `STEP_BY_STEP_DEPLOYMENT.md`
- **Quick Steps:** `QUICK_DEPLOY_STEPS.md`
- **Full Guide:** `DEPLOYMENT_GUIDE.md`

---

**🎉 Your DR Assistant is ready to use!**

