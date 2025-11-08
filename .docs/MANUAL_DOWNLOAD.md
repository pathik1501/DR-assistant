# 📊 Manual Dataset Download Guide

Since Kaggle API setup requires manual steps, here's how to download the datasets manually:

## Method 1: Manual Download (Recommended)

### APTOS 2019 Dataset
1. **Go to**: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
2. **Download**:
   - `train.csv` (labels file)
   - `train_images.zip` (image archive)
3. **Extract to**: `data/aptos2019/`
   - Place `train.csv` directly in `data/aptos2019/`
   - Extract `train_images.zip` to create `data/aptos2019/train_images/`

### EyePACS Dataset
1. **Go to**: https://www.kaggle.com/datasets/sovitrath/eyepacs-dataset
2. **Download**:
   - `trainLabels.csv` (labels file)
   - `train.zip` (image archive)
3. **Extract to**: `data/eyepacs/`
   - Place `trainLabels.csv` directly in `data/eyepacs/`
   - Extract `train.zip` to create `data/eyepacs/train/`

## Method 2: Use Kaggle API (After Setup)

Once you've placed `kaggle.json` in `C:\Users\pathi\.kaggle\`:

```bash
python download_datasets.py
```

## Expected File Structure

After download, you should have:

```
data/
├── aptos2019/
│   ├── train.csv
│   └── train_images/
│       ├── 000c1434d8d7.png
│       ├── 000d6e816238.png
│       └── ... (3,661 images)
└── eyepacs/
    ├── trainLabels.csv
    └── train/
        ├── 1_left.jpeg
        ├── 1_right.jpeg
        └── ... (35,126 images)
```

## Verification

After downloading, run:
```bash
python -c "from pathlib import Path; aptos_count = len(list(Path('data/aptos2019/train_images').glob('*.png'))); eyepacs_count = len(list(Path('data/eyepacs/train').glob('*.jpeg'))); print(f'APTOS: {aptos_count} images'); print(f'EyePACS: {eyepacs_count} images'); print(f'Total: {aptos_count + eyepacs_count} images')"
```

## Dataset Sizes
- **APTOS 2019**: ~3GB (3,661 images)
- **EyePACS**: ~10GB (35,126 images)
- **Total**: ~13GB

## Next Steps

Once datasets are downloaded:
1. **Test data loading**: `python -c "from src.data_processing import DataProcessor; processor = DataProcessor(); print('Data processor ready!')"`
2. **Start training**: `python src/train.py`
3. **Monitor progress**: Check MLflow at http://localhost:5000
