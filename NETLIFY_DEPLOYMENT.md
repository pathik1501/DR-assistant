# Netlify Deployment Guide

## ⚠️ Important: Netlify Limitations

**Netlify is NOT ideal for this application** because:

1. **Serverless Functions Timeout:**
   - Free tier: 10 seconds max
   - Pro tier: 26 seconds max
   - Model loading + inference can take 5-15+ seconds

2. **Cold Start Issues:**
   - Functions start cold (no persistent state)
   - Model needs to load on each request
   - First request can take 30+ seconds

3. **Memory Limits:**
   - Free tier: 128MB RAM
   - Pro tier: 1GB RAM
   - PyTorch model + dependencies need 2-4GB+ RAM

4. **No Persistent Storage:**
   - Model checkpoint needs to be loaded each time
   - Vector database needs to be recreated

## 🎯 Alternative: Hybrid Approach

**Best Option:** Deploy frontend to Netlify, API elsewhere

### Architecture:
```
┌─────────────┐         ┌─────────────┐
│   Netlify   │  ────>  │  API Server │
│  (Frontend) │         │  (Railway/  │
│  (Static)   │         │   Render)   │
└─────────────┘         └─────────────┘
```

---

## ✅ Option 1: Frontend on Netlify + API on Railway/Render

### Step 1: Deploy API to Railway/Render

**Railway:**
1. Go to https://railway.app
2. Deploy from GitHub: `pathik1501/DR-assistant`
3. Set `OPENAI_API_KEY` environment variable
4. Railway provides API URL: `https://dr-assistant-api.up.railway.app`

**Render:**
1. Go to https://render.com
2. Create Web Service from GitHub repo
3. Set `OPENAI_API_KEY` environment variable
4. Render provides API URL: `https://dr-assistant-api.onrender.com`

### Step 2: Create Static Frontend for Netlify

Create a simple HTML/JavaScript frontend that calls your API:

```html
<!-- index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>DR Assistant</title>
    <style>
        /* Add your styles */
    </style>
</head>
<body>
    <h1>Diabetic Retinopathy Detection Assistant</h1>
    <input type="file" id="imageInput" accept="image/*">
    <button onclick="analyzeImage()">Analyze Image</button>
    <div id="results"></div>
    
    <script>
        const API_URL = 'https://your-api-url.up.railway.app'; // Your API URL
        
        async function analyzeImage() {
            const file = document.getElementById('imageInput').files[0];
            if (!file) return;
            
            // Convert to base64
            const reader = new FileReader();
            reader.onload = async (e) => {
                const base64 = e.target.result.split(',')[1];
                
                // Call API
                const response = await fetch(`${API_URL}/predict_base64`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_base64: base64,
                        include_explanation: true,
                        include_hint: true
                    })
                });
                
                const result = await response.json();
                displayResults(result);
            };
            reader.readAsDataURL(file);
        }
        
        function displayResults(result) {
            // Display results
            document.getElementById('results').innerHTML = `
                <h2>Grade ${result.prediction}: ${result.grade_description}</h2>
                <p>Confidence: ${(result.confidence * 100).toFixed(1)}%</p>
                <p>${result.clinical_hint || ''}</p>
            `;
        }
    </script>
</body>
</html>
```

### Step 3: Deploy Frontend to Netlify

1. Create a new folder: `netlify-frontend/`
2. Put your `index.html` and assets there
3. Push to GitHub
4. Go to https://app.netlify.com
5. Click "New site from Git"
6. Select your repository
7. Set build directory: `netlify-frontend/`
8. Deploy

### Step 4: Update API URL

Update the `API_URL` in your frontend to point to your Railway/Render API.

---

## ⚠️ Option 2: Netlify Functions (Not Recommended)

If you really want to try Netlify Functions, here's how (but it will be slow and may timeout):

### Step 1: Create Netlify Function

Create `netlify/functions/predict.py`:

```python
import json
import base64
import requests
import os

def handler(event, context):
    """Netlify serverless function for DR prediction."""
    
    # Get API URL from environment (deployed elsewhere)
    api_url = os.environ.get('API_URL', 'https://your-api-url.up.railway.app')
    
    try:
        # Parse request
        body = json.loads(event['body'])
        image_base64 = body.get('image_base64')
        
        # Forward to actual API
        response = requests.post(
            f'{api_url}/predict_base64',
            json={
                'image_base64': image_base64,
                'include_explanation': body.get('include_explanation', True),
                'include_hint': body.get('include_hint', True)
            },
            timeout=25  # Netlify Pro limit
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps(response.json())
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### Step 2: Configure Netlify

Create `netlify.toml`:

```toml
[build]
  functions = "netlify/functions"
  publish = "dist"

[[redirects]]
  from = "/api/*"
  to = "/.netlify/functions/:splat"
  status = 200
```

### Step 3: Deploy

```bash
netlify deploy --prod
```

**⚠️ Warning:** This will be slow and may timeout on free tier!

---

## 🎯 Recommended: Better Alternatives

### Option 1: Railway (Best for Full Stack)
- ✅ Deploys both API and frontend
- ✅ No timeout limits
- ✅ Persistent storage
- ✅ Easy setup
- **Cost:** Free tier available

### Option 2: Render (Good Alternative)
- ✅ Full-stack deployment
- ✅ No timeout limits
- ✅ Persistent storage
- **Cost:** Free tier available

### Option 3: Heroku (Classic)
- ✅ Full-stack deployment
- ✅ Well-documented
- **Cost:** Paid plans only

### Option 4: Vercel (Better than Netlify for Python)
- ✅ Better Python support
- ✅ Serverless functions (26s timeout)
- ⚠️ Still has timeout limits
- **Cost:** Free tier available

---

## 📋 Comparison Table

| Platform | Timeout | Memory | Full Stack | Best For |
|----------|---------|--------|------------|----------|
| **Netlify** | 10-26s | 128MB-1GB | ❌ | Static sites |
| **Railway** | None | Unlimited | ✅ | Full stack apps |
| **Render** | None | 512MB-2GB | ✅ | Full stack apps |
| **Vercel** | 26s | 1GB | ⚠️ | Next.js, limited Python |
| **Heroku** | None | 512MB-14GB | ✅ | Full stack apps |

---

## ✅ Recommended Deployment Strategy

### For Production:
1. **API:** Deploy to Railway or Render
2. **Frontend:** Deploy to Netlify (static HTML/JS) OR keep on same platform

### Quick Setup:
```bash
# 1. Deploy API to Railway
# - Connect GitHub repo
# - Set OPENAI_API_KEY
# - Get API URL

# 2. Deploy frontend to Netlify
# - Create static frontend
# - Point to Railway API URL
# - Deploy
```

---

## 🚀 Quick Start: Railway (Recommended)

Instead of Netlify, use Railway for full deployment:

1. Go to https://railway.app
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select: `pathik1501/DR-assistant`
5. Add environment variable: `OPENAI_API_KEY`
6. Deploy automatically

**Done!** Railway handles everything.

---

## 📚 See Also

- `SIMPLE_DEPLOY.md` - Simple deployment guide
- `STEP_BY_STEP_DEPLOYMENT.md` - Detailed deployment steps
- `DEPLOYMENT_GUIDE.md` - Full deployment guide

---

## 💡 Summary

**Netlify is NOT recommended** for this application because:
- ❌ Timeout limits too short
- ❌ Memory limits too small
- ❌ Cold start issues
- ❌ Not designed for ML inference

**Better alternatives:**
- ✅ **Railway** - Best for full stack
- ✅ **Render** - Good alternative
- ✅ **Heroku** - Classic option

**If you must use Netlify:**
- Deploy frontend to Netlify (static)
- Deploy API to Railway/Render
- Connect them together

---

**Recommendation: Use Railway or Render instead of Netlify for this application.**

