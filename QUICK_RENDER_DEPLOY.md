# 🚀 Quick Render.com Deployment Guide

## Why Render.com?

- ✅ **No image size limits** (solves your 9.3 GB problem!)
- ✅ **Free tier available**
- ✅ **Auto-deploy from GitHub**
- ✅ **Easy setup** (5 minutes)
- ✅ **Better for ML apps** than Railway

---

## Step-by-Step Deployment (5 minutes)

### Step 1: Sign Up
1. Go to **https://render.com**
2. Click **"Get Started for Free"**
3. Sign up with **GitHub** (recommended)
4. Authorize Render to access your repositories

### Step 2: Create Web Service
1. Click **"New +"** → **"Web Service"**
2. Select **"Connect a repository"**
3. Find and select: **`pathik1501/DR-assistant`**
4. Click **"Connect"**

### Step 3: Configure Service
1. **Name**: `dr-assistant-api` (or any name)
2. **Region**: Choose closest to you (e.g., `Oregon`)
3. **Branch**: `main`
4. **Root Directory**: (leave empty)
5. **Runtime**: `Python 3`
6. **Build Command**: 
   ```
   pip install -r requirements.txt
   ```
7. **Start Command**:
   ```
   python src/inference.py
   ```

### Step 4: Add Environment Variables
Click **"Advanced"** → **"Add Environment Variable"**:

- **Key**: `OPENAI_API_KEY`
- **Value**: Your OpenAI API key
- Click **"Add"**

(Optional - Render sets PORT automatically):
- **Key**: `PORT`
- **Value**: `8080`

### Step 5: Deploy!
1. Click **"Create Web Service"**
2. Wait 5-10 minutes for first build
3. Your API will be live at: `https://dr-assistant-api.onrender.com`

### Step 6: Get Your API URL
1. Once deployed, you'll see your service URL
2. Test it: `https://your-service.onrender.com/health`
3. Should return: `{"status":"healthy"}`

---

## Update Frontend (Netlify)

1. Go to **Netlify dashboard**
2. **Site settings** → **Environment variables**
3. Update `API_URL` to your Render URL:
   ```
   https://dr-assistant-api.onrender.com
   ```
4. **Trigger redeploy**

---

## ✅ That's It!

**Your DR Assistant is now live:**
- **Frontend**: `https://your-app.netlify.app`
- **API**: `https://your-service.onrender.com`

---

## 🎯 Advantages Over Railway

| Feature | Railway | Render |
|---------|---------|--------|
| Image Size Limit | ❌ 4 GB | ✅ No limit |
| Free Tier | ✅ Yes | ✅ Yes |
| Auto Deploy | ✅ Yes | ✅ Yes |
| ML App Support | ⚠️ Limited | ✅ Better |
| Setup Time | 5 min | 5 min |

---

## 💡 Pro Tips

1. **Free tier limitations:**
   - Service spins down after 15 min inactivity
   - First request after spin-down takes ~30 seconds
   - Upgrade to paid ($7/month) for always-on

2. **Monitor logs:**
   - Go to **"Logs"** tab in Render dashboard
   - See real-time build and runtime logs

3. **Auto-deploy:**
   - Render auto-deploys on every Git push
   - Just push to GitHub and it updates!

4. **Custom domain:**
   - Add custom domain in **"Settings"**
   - Free SSL certificate included

---

## 🐛 Troubleshooting

### Service won't start?
- Check **"Logs"** tab for errors
- Verify `OPENAI_API_KEY` is set
- Check if model checkpoint exists

### Timeout on first request?
- Free tier spins down after inactivity
- First request after spin-down is slow (~30s)
- This is normal for free tier

### Build fails?
- Check build logs
- Verify `requirements.txt` is correct
- Make sure Python 3 is selected

---

**Render.com is the easiest alternative to Railway!** 🚀

