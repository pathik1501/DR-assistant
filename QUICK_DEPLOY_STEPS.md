# Quick Deployment Steps

## 🚀 Fastest Way: Docker (5 minutes)

### Step 1: Clone & Setup
```bash
git clone https://github.com/pathik1501/DR-assistant.git
cd DR-assistant
cp .env.example .env
```

### Step 2: Add API Key
Edit `.env` file:
```
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

### Step 3: Deploy
```bash
docker-compose up -d
```

### Step 4: Access
- Frontend: http://localhost:8501
- API: http://localhost:8080/health

**Done! ✅**

---

## ☁️ Cloud Platform: Railway (3 minutes)

### Step 1: Sign Up
1. Go to https://railway.app
2. Sign up with GitHub

### Step 2: Deploy
1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose: `pathik1501/DR-assistant`

### Step 3: Add API Key
1. Go to Settings → Variables
2. Add: `OPENAI_API_KEY` = `sk-proj-your-key`

### Step 4: Access
Railway provides a public URL automatically

**Done! ✅**

---

## 🖥️ VPS/Server (10 minutes)

### Step 1: Install
```bash
sudo apt update
sudo apt install python3 python3-pip git -y
```

### Step 2: Clone
```bash
git clone https://github.com/pathik1501/DR-assistant.git
cd DR-assistant
```

### Step 3: Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add OPENAI_API_KEY
```

### Step 4: Run
```bash
python src/inference.py
```

**Done! ✅**

---

## 📋 Required: Environment Variable

**OPENAI_API_KEY** - Get from: https://platform.openai.com/api-keys

Add to:
- `.env` file (for local/Docker)
- Platform environment variables (for cloud)
- System environment (for VPS)

---

## ✅ Verify Deployment

```bash
# Check API
curl http://localhost:8080/health

# Should return: {"status": "healthy"}
```

---

**See `STEP_BY_STEP_DEPLOYMENT.md` for detailed instructions!**

