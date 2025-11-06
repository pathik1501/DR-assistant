# 🚀 Quick Web Deployment Guide

Deploy your DR Assistant to the web in **5 minutes**!

## 🎯 Two-Step Deployment

### Step 1: Deploy API (Railway) - 3 minutes

1. Go to **https://railway.app**
2. Sign up with **GitHub**
3. Click **"New Project"** → **"Deploy from GitHub repo"**
4. Select your repository: `pathik1501/DR-assistant`
5. Add environment variable:
   - **Key**: `OPENAI_API_KEY`
   - **Value**: Your OpenAI API key
6. Railway will auto-deploy!
7. **Copy your API URL** (e.g., `https://dr-assistant-api.up.railway.app`)

### Step 2: Deploy Frontend (Netlify) - 2 minutes

1. Go to **https://app.netlify.com**
2. Sign up with **GitHub**
3. Click **"Add new site"** → **"Import an existing project"**
4. Select your repository: `pathik1501/DR-assistant`
5. **Build settings**:
   - **Base directory**: `frontend`
   - **Publish directory**: `frontend`
   - **Build command**: (leave empty)
6. **Environment variables**:
   - **Key**: `API_URL`
   - **Value**: Your Railway API URL (from Step 1)
7. Click **"Deploy site"**

## ✅ Done!

Your DR Assistant is now live at:
- **Frontend**: `https://your-app.netlify.app`
- **API**: `https://your-api.up.railway.app`

## 🔧 Update API for Production

Before deploying, make sure your API supports the `PORT` environment variable:

The `src/inference.py` file has been updated to automatically use the `PORT` environment variable if set (for Railway/Render).

## 📝 Notes

- **Railway** provides free tier with 500 hours/month
- **Netlify** provides free tier with 100GB bandwidth/month
- Both auto-deploy on Git push
- Custom domains available

## 🐛 Troubleshooting

### API not working?
- Check Railway logs
- Verify `OPENAI_API_KEY` is set
- Test API health: `https://your-api.up.railway.app/health`

### Frontend can't connect?
- Verify `API_URL` environment variable in Netlify
- Check browser console (F12) for errors
- Make sure CORS is enabled in API (already configured)

---

**That's it!** Your DR Assistant is now accessible worldwide! 🌍

