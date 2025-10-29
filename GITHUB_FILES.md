# Files to Push to GitHub

## ✅ Should Be Included (Core Project Files)

### Source Code
```
src/
├── model.py                 ✅
├── data_processing.py        ✅
├── train.py                  ✅
├── enhanced_train.py         ✅
├── inference.py              ✅
├── explainability.py         ✅
├── rag_pipeline.py           ✅
└── eval.py                   ✅
```

### Configuration
```
configs/
└── config.yaml               ✅ (Remove API keys if any!)
```

### Frontend
```
frontend/
├── app.py                    ✅ (original)
└── app_new.py                ✅ (new improved version)
```

### Tests
```
tests/
└── test_dr_system.py         ✅
```

### Documentation
```
README.md                     ✅
*.md                          ✅ (All markdown docs)
```

### Deployment Files
```
Dockerfile                    ✅
docker-compose.yml            ✅
requirements.txt              ✅ (or requirements_simple.txt)
setup.py                      ✅
simple_setup.py               ✅
```

### Configuration/Infrastructure
```
.gitignore                    ✅
monitoring/
├── prometheus.yml            ✅
└── grafana/                  ✅
```

### Scripts
```
deploy.py                     ✅
launch_monitoring.py          ✅
download_datasets.py          ✅ (optional)
*.ps1                         ✅ (PowerShell scripts)
```

## ❌ Should NOT Be Included

### Data (Too Large)
```
data/                         ❌
├── aptos2019/                ❌
└── eyepacs/                  ❌
```

### Model Checkpoints (Too Large)
```
1/                            ❌ (MLflow runs)
models/                       ❌
*.ckpt                        ❌
*.pth                         ❌
```

### Logs & Outputs
```
logs/                         ❌
outputs/                      ❌
mlflow.db                     ❌
mlruns/                       ❌
```

### Environment/Secrets
```
.env                          ❌
*.env                         ❌
secrets.yaml                  ❌
config_secrets.yaml           ❌
```

### Cache/Temporary
```
__pycache__/                  ❌
*.pyc                         ❌
*.pyo                         ❌
*.log                         ❌
*.tmp                         ❌
api_response.json             ❌
test_output.txt               ❌
```

### Virtual Environments
```
venv/                         ❌
env/                          ❌
.venv/                        ❌
```

## 📋 Quick Checklist

Before pushing, ensure:

- [ ] `.gitignore` is created and correct
- [ ] No API keys in code (use environment variables)
- [ ] No large data files (>100MB)
- [ ] No model checkpoints
- [ ] No `.env` files with secrets
- [ ] README.md is updated
- [ ] All source code is included
- [ ] Config files don't contain secrets

## 🚀 Recommended Files Structure for GitHub

```
DR-assistant/
├── .gitignore                 ✅
├── README.md                  ✅
├── requirements.txt           ✅
├── Dockerfile                ✅
├── docker-compose.yml         ✅
├── setup.py                  ✅
│
├── src/                      ✅
│   ├── __init__.py           ✅
│   ├── model.py              ✅
│   ├── data_processing.py    ✅
│   ├── train.py              ✅
│   ├── inference.py          ✅
│   ├── explainability.py     ✅
│   ├── rag_pipeline.py      ✅
│   └── eval.py               ✅
│
├── frontend/                 ✅
│   ├── app.py                ✅
│   └── app_new.py            ✅
│
├── configs/                  ✅
│   └── config.yaml           ✅
│
├── tests/                    ✅
│   └── test_dr_system.py     ✅
│
├── monitoring/               ✅
│   ├── prometheus.yml        ✅
│   └── grafana/              ✅
│
├── scripts/                  ✅ (optional, put .ps1 here)
│
└── docs/                     ✅ (optional, put .md here)
```

## 📝 Files to Review Before Pushing

### Check These Files for Secrets:

1. **`configs/config.yaml`**
   - Remove any API keys
   - Use environment variables instead

2. **`src/inference.py`**
   - Check for hardcoded API keys (line 285)
   - Should use environment variables

3. **`*.ps1` scripts**
   - Remove API keys from scripts
   - Or use environment variables

4. **`README.md`**
   - Update with installation instructions
   - Remove any sensitive info

## 🔒 Security Reminder

**NEVER PUSH:**
- API keys
- Passwords
- Private credentials
- Large datasets (>100MB)
- Personal information

**DO PUSH:**
- Source code
- Configuration templates
- Documentation
- Setup scripts
- Infrastructure files

## 📦 Example `.gitignore` is Provided

The `.gitignore` file has been created with all necessary exclusions.

## 🎯 Summary

**Push these:**
- ✅ All Python source code (`src/`)
- ✅ Frontend UI (`frontend/`)
- ✅ Configuration templates (`configs/`)
- ✅ Documentation (`*.md`)
- ✅ Docker files
- ✅ Tests
- ✅ Setup scripts
- ✅ `.gitignore`

**Don't push:**
- ❌ Data files (`data/`)
- ❌ Model checkpoints (`*.ckpt`, `1/`)
- ❌ Logs (`logs/`, `outputs/`)
- ❌ Secrets (API keys, `.env`)
- ❌ Large files (>100MB)

