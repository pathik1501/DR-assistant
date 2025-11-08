# Step-by-Step Deployment Guide

## 🎯 Choose Your Deployment Method

### Option 1: Docker (Recommended - Easiest)
### Option 2: Cloud Platform (Heroku, Railway, etc.)
### Option 3: VPS/Server (Full Control)

---

## 🐳 Option 1: Docker Deployment (Recommended)

### Prerequisites
- Docker installed: https://www.docker.com/get-started
- Docker Compose installed (usually comes with Docker Desktop)

### Step 1: Clone Repository
```bash
git clone https://github.com/pathik1501/DR-assistant.git
cd DR-assistant
```

### Step 2: Create Environment File
```bash
# Copy the example file
cp .env.example .env

# Edit .env file and add your OpenAI API key
# Windows: notepad .env
# Linux/Mac: nano .env
```

Add this line to `.env`:
```
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

### Step 3: Build Docker Image
```bash
docker build -t dr-assistant:latest .
```

This will:
- Install all Python dependencies
- Copy all source code
- Set up the application

**Time:** 5-10 minutes (first time)

### Step 4: Run with Docker Compose
```bash
docker-compose up -d
```

This starts:
- API server on port 8080
- Frontend on port 8501

### Step 5: Verify Deployment
```bash
# Check API health
curl http://localhost:8080/health

# Should return: {"status": "healthy", "timestamp": ...}

# Open frontend in browser
# http://localhost:8501
```

### Step 6: Access Application
- **Frontend:** http://localhost:8501
- **API:** http://localhost:8080
- **API Docs:** http://localhost:8080/docs

### Step 7: Stop Services
```bash
docker-compose down
```

### Troubleshooting Docker
```bash
# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Rebuild after code changes
docker-compose up -d --build
```

---

## ☁️ Option 2: Cloud Platform Deployment

### A. Heroku Deployment

#### Step 1: Install Heroku CLI
Download from: https://devcenter.heroku.com/articles/heroku-cli

#### Step 2: Login to Heroku
```bash
heroku login
```

#### Step 3: Create Heroku App
```bash
heroku create dr-assistant
```

#### Step 4: Set Environment Variables
```bash
heroku config:set OPENAI_API_KEY=sk-proj-your-actual-key-here
```

#### Step 5: Deploy
```bash
git push heroku main
```

#### Step 6: Open Application
```bash
heroku open
```

Your app will be at: `https://dr-assistant.herokuapp.com`

#### Step 7: View Logs
```bash
heroku logs --tail
```

---

### B. Railway Deployment

#### Step 1: Sign Up
Go to: https://railway.app
Sign up with GitHub

#### Step 2: Create New Project
1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose your repository: `pathik1501/DR-assistant`

#### Step 3: Add Environment Variables
1. Go to Project Settings
2. Click "Variables"
3. Add: `OPENAI_API_KEY` = `sk-proj-your-actual-key-here`

#### Step 4: Configure Services
Railway will auto-detect:
- Python application
- Requirements.txt
- Port 8080 for API

#### Step 5: Deploy
Railway automatically deploys on every push to main branch

#### Step 6: Access Application
Railway provides a public URL like: `https://dr-assistant.up.railway.app`

---

### C. Render Deployment

#### Step 1: Sign Up
Go to: https://render.com
Sign up with GitHub

#### Step 2: Create New Web Service
1. Click "New +" → "Web Service"
2. Connect your GitHub repository
3. Select: `pathik1501/DR-assistant`

#### Step 3: Configure Service
- **Name:** dr-assistant
- **Environment:** Python 3
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `python src/inference.py`
- **Port:** 8080

#### Step 4: Add Environment Variables
Add: `OPENAI_API_KEY` = `sk-proj-your-actual-key-here`

#### Step 5: Deploy
Click "Create Web Service"
Render will build and deploy automatically

#### Step 6: Access Application
Render provides a URL like: `https://dr-assistant.onrender.com`

---

## 🖥️ Option 3: VPS/Server Deployment

### Prerequisites
- Ubuntu/Debian server (or similar Linux)
- SSH access
- Root or sudo access

### Step 1: Connect to Server
```bash
ssh user@your-server-ip
```

### Step 2: Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.8+
sudo apt install python3 python3-pip python3-venv -y

# Install Git
sudo apt install git -y

# Install Nginx (for reverse proxy)
sudo apt install nginx -y
```

### Step 3: Clone Repository
```bash
cd /opt
sudo git clone https://github.com/pathik1501/DR-assistant.git
cd DR-assistant
sudo chown -R $USER:$USER .
```

### Step 4: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 5: Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6: Create Environment File
```bash
cp .env.example .env
nano .env
```

Add:
```
OPENAI_API_KEY=sk-proj-your-actual-key-here
API_HOST=0.0.0.0
API_PORT=8080
```

Save and exit (Ctrl+X, Y, Enter)

### Step 7: Test API Server
```bash
# Test if it starts
python src/inference.py
```

Press Ctrl+C to stop

### Step 8: Create Systemd Service for API
```bash
sudo nano /etc/systemd/system/dr-assistant-api.service
```

Add this content:
```ini
[Unit]
Description=DR Assistant API Service
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/opt/DR-assistant
Environment="PATH=/opt/DR-assistant/venv/bin"
ExecStart=/opt/DR-assistant/venv/bin/python src/inference.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Replace `your-username` with your actual username.

Save and exit.

### Step 9: Start API Service
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable dr-assistant-api

# Start service
sudo systemctl start dr-assistant-api

# Check status
sudo systemctl status dr-assistant-api
```

### Step 10: Configure Nginx (Reverse Proxy)
```bash
sudo nano /etc/nginx/sites-available/dr-assistant
```

Add this content:
```nginx
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain or IP

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Save and exit.

### Step 11: Enable Nginx Site
```bash
# Create symbolic link
sudo ln -s /etc/nginx/sites-available/dr-assistant /etc/nginx/sites-enabled/

# Test Nginx configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

### Step 12: Configure Firewall
```bash
# Allow HTTP and HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Enable firewall
sudo ufw enable
```

### Step 13: Access Application
- **API:** http://your-server-ip
- **Health Check:** http://your-server-ip/health
- **API Docs:** http://your-server-ip/docs

### Step 14: Set Up SSL (Optional but Recommended)
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal is set up automatically
```

---

## 🔧 Common Deployment Issues & Solutions

### Issue 1: API Key Not Working
**Solution:**
```bash
# Verify environment variable is set
echo $OPENAI_API_KEY

# For Docker
docker-compose exec api env | grep OPENAI_API_KEY

# Restart service after setting
docker-compose restart
# or
sudo systemctl restart dr-assistant-api
```

### Issue 2: Port Already in Use
**Solution:**
```bash
# Find what's using the port
sudo lsof -i :8080

# Kill the process
sudo kill -9 <PID>

# Or change port in configs/config.yaml
```

### Issue 3: Model Checkpoint Not Found
**Solution:**
```bash
# Verify checkpoint exists
ls -lh 1/7b2108ed09bf401fa06ff1b7d8c1e949/checkpoints/

# If missing, download from GitHub
git pull origin main
```

### Issue 4: RAG Pipeline Not Initializing
**Solution:**
1. Check OpenAI API key is set correctly
2. Verify API key has quota/credits
3. Check logs for errors:
   ```bash
   docker-compose logs api
   # or
   sudo journalctl -u dr-assistant-api -f
   ```

### Issue 5: Frontend Can't Connect to API
**Solution:**
1. Verify API is running: `curl http://localhost:8080/health`
2. Check CORS settings in `src/inference.py`
3. Update frontend API URL if needed

---

## 📋 Post-Deployment Checklist

- [ ] API health check returns `{"status": "healthy"}`
- [ ] Frontend accessible in browser
- [ ] Can upload test image
- [ ] Predictions work correctly
- [ ] RAG pipeline initializes (check logs)
- [ ] Scan explanations generate
- [ ] Heatmaps display correctly
- [ ] No errors in logs

---

## 🔍 Monitoring & Maintenance

### View Logs

**Docker:**
```bash
docker-compose logs -f api
docker-compose logs -f frontend
```

**Systemd:**
```bash
sudo journalctl -u dr-assistant-api -f
```

**Heroku:**
```bash
heroku logs --tail
```

### Restart Services

**Docker:**
```bash
docker-compose restart
```

**Systemd:**
```bash
sudo systemctl restart dr-assistant-api
```

### Update Application

**Docker:**
```bash
git pull origin main
docker-compose up -d --build
```

**Systemd:**
```bash
cd /opt/DR-assistant
git pull origin main
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart dr-assistant-api
```

---

## 🎯 Quick Start Commands

### Docker (Fastest)
```bash
git clone https://github.com/pathik1501/DR-assistant.git
cd DR-assistant
cp .env.example .env
# Edit .env and add OPENAI_API_KEY
docker-compose up -d
# Access: http://localhost:8501
```

### Cloud Platform (Easiest)
1. Sign up for Railway/Render
2. Connect GitHub repo
3. Set `OPENAI_API_KEY` environment variable
4. Deploy automatically

### VPS (Most Control)
```bash
git clone https://github.com/pathik1501/DR-assistant.git
cd DR-assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add OPENAI_API_KEY
python src/inference.py
```

---

## 📞 Need Help?

- Check logs for error messages
- Verify environment variables are set
- Ensure ports are not in use
- Check firewall settings
- Review `DEPLOYMENT_GUIDE.md` for more details

---

**🎉 Your DR Assistant is ready to deploy!**

