# Files to Push to GitHub - Complete List

## ✅ MUST PUSH (Core Project Files)

### Source Code (`src/`)
```
src/
├── __init__.py              ✅ (create if missing)
├── model.py                 ✅
├── data_processing.py        ✅
├── train.py                  ✅
├── enhanced_train.py         ✅
├── inference.py              ✅
├── explainability.py         ✅
├── rag_pipeline.py           ✅ (remove API key reference)
└── eval.py                   ✅
```

### Frontend (`frontend/`)
```
frontend/
├── app.py                    ✅
└── app_new.py                ✅
```

### Configuration (`configs/`)
```
configs/
└── config.yaml               ✅ (template only, no secrets)
```

### Tests (`tests/`)
```
tests/
└── test_dr_system.py         ✅
```

### Infrastructure
```
Dockerfile                    ✅
docker-compose.yml            ✅
requirements.txt              ✅
requirements_simple.txt       ✅ (if using)
setup.py                      ✅
simple_setup.py               ✅
```

### Scripts
```
deploy.py                     ✅
launch_monitoring.py           ✅
download_datasets.py           ✅
test_api.py                   ✅
test_data.py                  ✅
test_gpu.py                   ✅
quick_test.py                 ✅
```

### Monitoring
```
monitoring/
├── prometheus.yml            ✅
└── grafana/
    ├── dashboards/           ✅
    └── datasources/          ✅
```

### Documentation (All `.md` files)
```
README.md                     ✅
*.md                          ✅ (20+ documentation files)
```

## ⚠️ MODIFY BEFORE PUSHING

### PowerShell Scripts (Remove API Keys)
```
restart_with_ui_fixes.ps1     ⚠️ Remove API key from line 14
start_simple.ps1              ⚠️ Remove API key if present
restart_with_fix.ps1          ⚠️ Remove API key if present
restart_server.ps1            ⚠️ Remove API key if present
start_server.ps1              ⚠️ Remove API key if present
start_ui.ps1                  ✅ (should be safe)
```

**Fix:** Replace hardcoded keys with:
```powershell
$env:OPENAI_API_KEY = $env:OPENAI_API_KEY  # Use environment variable
```

### Config Files
```
configs/config.yaml           ⚠️ Ensure no API keys hardcoded
```

## ❌ DO NOT PUSH (Excluded by .gitignore)

### Data Files (Too Large)
```
data/                         ❌ All excluded
├── aptos2019/                ❌
└── eyepacs/                  ❌
```

### Model Checkpoints (Too Large)
```
1/                            ❌ MLflow runs
models/                       ❌ 
*.ckpt                        ❌ All checkpoints
```

### Logs & Outputs
```
logs/                         ❌
outputs/                      ❌
mlflow.db                     ❌
mlruns/                       ❌
```

### Cache & Temporary
```
__pycache__/                  ❌
*.pyc                         ❌
*.log                         ❌
*.tmp                         ❌
api_response.json             ❌
test_output.txt               ❌
```

### Test Files (Optional - can exclude)
```
test_*.py                     ⚠️ (individual test scripts - optional)
test_*.ps1                     ⚠️ (optional)
check_*.py                    ⚠️ (optional)
```

## 📋 Quick Command to Add Files

```bash
# Add core source code
git add src/
git add frontend/
git add configs/
git add tests/

# Add infrastructure
git add Dockerfile
git add docker-compose.yml
git add requirements*.txt
git add setup.py
git add simple_setup.py

# Add scripts (after removing keys)
git add deploy.py
git add launch_monitoring.py
git add download_datasets.py
git add *.py  # Be careful - check test files first

# Add monitoring
git add monitoring/

# Add documentation
git add *.md

# Add .gitignore
git add .gitignore

# Review before committing
git status
```

## 🔒 Security: Files to Clean

**Before pushing, clean these files:**

1. **PowerShell Scripts** (5 files)
   - Remove: `sk-proj-...` API keys
   - Replace with: Environment variable reference

2. **Python Files**
   - Check `src/rag_pipeline.py` for hardcoded keys
   - Use `os.getenv("OPENAI_API_KEY")` instead

3. **Config Files**
   - Ensure `configs/config.yaml` has no secrets
   - Use environment variable placeholders

## 📊 File Count Summary

**Total files to push:** ~60-80 files

**By category:**
- Python source: ~10 files
- Frontend: 2 files
- Documentation: ~25 files
- Config/scripts: ~15 files
- Tests: ~1 file
- Infrastructure: ~5 files

**Estimated size:** <5MB (excluding data/models)

## ✅ Final Checklist

Before `git push`:

- [ ] ✅ `.gitignore` created and committed
- [ ] ✅ API keys removed from PowerShell scripts
- [ ] ✅ API keys removed from Python files
- [ ] ✅ Config files have no secrets
- [ ] ✅ Large files excluded (data/, logs/, outputs/)
- [ ] ✅ Model checkpoints excluded
- [ ] ✅ README.md updated with setup instructions
- [ ] ✅ All source code added
- [ ] ✅ Reviewed `git status` output

## 🚀 Ready to Push!

Once all sensitive data is removed, you're ready to push! 🎉



