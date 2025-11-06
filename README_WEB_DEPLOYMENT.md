# 🌐 Web Deployment - Complete Setup

Your DR Assistant is now ready for web deployment! Here's everything you need to know.

## ✅ What's Been Set Up

### Frontend (Netlify-Ready)
- ✅ `frontend/index.html` - Main HTML file
- ✅ `frontend/app.js` - JavaScript (uses config.js for API URL)
- ✅ `frontend/style.css` - Styling
- ✅ `frontend/config.js` - API URL configuration
- ✅ `netlify.toml` - Netlify configuration
- ✅ `frontend/_redirects` - Netlify routing

### API (Railway/Render-Ready)
- ✅ `src/inference.py` - Updated to use PORT environment variable
- ✅ CORS middleware - Already configured
- ✅ `railway.json` - Railway deployment config
- ✅ `Procfile` - Heroku/Render deployment config
- ✅ `runtime.txt` - Python version specification

## 🚀 Quick Start

### Option 1: Railway + Netlify (Recommended)

**Deploy API to Railway:**
1. Go to https://railway.app
2. Sign up with GitHub
3. New Project → Deploy from GitHub
4. Select your repo
5. Add environment variable: `OPENAI_API_KEY`
6. Copy your API URL

**Deploy Frontend to Netlify:**
1. Go to https://app.netlify.com
2. Sign up with GitHub
3. Add new site → Import from GitHub
4. Base directory: `frontend`
5. Publish directory: `frontend`
6. Add environment variable: `API_URL` = Your Railway API URL
7. Deploy!

### Option 2: Render + Netlify

**Deploy API to Render:**
1. Go to https://render.com
2. New → Web Service
3. Connect GitHub repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `python src/inference.py`
6. Add environment variables:
   - `OPENAI_API_KEY`
   - `PORT` = `8080`
7. Deploy

**Deploy Frontend to Netlify:** (Same as Option 1)

## 📋 Files Created

### Deployment Configs
- `netlify.toml` - Netlify build settings
- `railway.json` - Railway deployment config
- `Procfile` - Heroku/Render deployment
- `runtime.txt` - Python version

### Frontend Updates
- `frontend/config.js` - API URL configuration
- `frontend/_redirects` - Netlify routing rules
- Updated `frontend/app.js` - Uses config.js for API URL
- Updated `frontend/index.html` - Loads config.js

### API Updates
- Updated `src/inference.py` - Supports PORT environment variable

### Documentation
- `WEB_DEPLOYMENT.md` - Comprehensive deployment guide
- `QUICK_WEB_DEPLOY.md` - Quick reference
- `NETLIFY_DEPLOYMENT.md` - Netlify-specific guide

## 🔧 How It Works

### Frontend (Netlify)
1. `config.js` sets `window.API_URL` from Netlify environment variable
2. `app.js` reads `window.API_URL` or defaults to `localhost:8080`
3. All API calls use the configured URL

### API (Railway/Render)
1. Reads `PORT` environment variable (set by platform)
2. Falls back to config.yaml port (8080) for local development
3. CORS is enabled to allow frontend requests

## 🎯 Environment Variables

### Railway/Render (API)
- `OPENAI_API_KEY` - Your OpenAI API key (required for RAG)
- `PORT` - Automatically set by platform

### Netlify (Frontend)
- `API_URL` - Your deployed API URL (e.g., `https://your-api.up.railway.app`)

## ✅ Testing

### Local Development
1. Start API: `python src/inference.py`
2. Open `frontend/index.html` in browser
3. Should connect to `http://localhost:8080`

### Production
1. Test API: `curl https://your-api.up.railway.app/health`
2. Test Frontend: Open your Netlify URL
3. Upload an image and test analysis

## 🐛 Troubleshooting

### Frontend Can't Connect to API
- ✅ Check `API_URL` environment variable in Netlify
- ✅ Verify API is running (check Railway/Render logs)
- ✅ Test API health endpoint directly
- ✅ Check browser console (F12) for CORS errors

### API Not Starting
- ✅ Check Railway/Render logs
- ✅ Verify `OPENAI_API_KEY` is set
- ✅ Verify model checkpoint exists
- ✅ Check Python version (3.11 recommended)

### CORS Errors
- ✅ CORS is already configured in `src/inference.py`
- ✅ If issues persist, check `allow_origins` in CORS middleware

## 📚 Documentation

- **Quick Start**: See `QUICK_WEB_DEPLOY.md`
- **Full Guide**: See `WEB_DEPLOYMENT.md`
- **Netlify Details**: See `NETLIFY_DEPLOYMENT.md`

## 🎉 You're Ready!

Your DR Assistant is now configured for web deployment. Just follow the steps above to deploy to Railway/Netlify and you'll have a public URL!

---

**Questions?** Check the deployment logs in Railway/Netlify dashboards.

