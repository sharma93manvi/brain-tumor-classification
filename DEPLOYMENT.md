# Streamlit Cloud Deployment Guide

## Prerequisites

1. **GitHub Account**: Your code must be pushed to a GitHub repository
2. **Streamlit Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Model Files**: The trained model files need to be available

## Deployment Steps

### Step 1: Prepare Your Repository

Ensure your code is pushed to GitHub:
```bash
git push origin main
```

### Step 2: Model Files Setup

**Important**: Model files are large (especially `efficientnet_b3_finetuned.pth` at ~123MB) and are not included in the repository.

**Option A: Use Git LFS (Recommended for large files)**
```bash
# Install Git LFS
git lfs install

# Track large model files
git lfs track "models/*.pth"
git lfs track "models/*.pkl"

# Add and commit models
git add .gitattributes models/
git commit -m "Add model files with Git LFS"
git push origin main
```

**Option B: Host models externally and download at runtime**
- Upload models to Google Drive, Dropbox, or cloud storage
- Modify `app.py` to download models on first run
- See `scripts/download_models.py` for example

**Option C: Use only Feature Extraction models**
- The feature extraction models are smaller (~50-65KB)
- Fine-tuned model can be optional
- App will work with available models

### Step 3: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**: Visit [share.streamlit.io](https://share.streamlit.io)

2. **Sign in**: Use your GitHub account to sign in

3. **New App**: Click "New app" button

4. **Configure Deployment**:
   - **Repository**: Select `sharma93manvi/brain-tumor-classification`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **Python version**: 3.9 or 3.10 (recommended)

5. **Advanced Settings** (if needed):
   - **Secrets**: If using external model storage, add API keys/secrets here
   - **Python version**: 3.9 or 3.10

6. **Deploy**: Click "Deploy"

### Step 4: Post-Deployment

After deployment, if models are missing:

1. **Option 1**: Upload models via Streamlit Cloud's file manager (if available)
2. **Option 2**: Use Git LFS and redeploy
3. **Option 3**: Set up model downloading script

## Troubleshooting

### Models Not Found Error

If you see "Model Not Available" errors:

1. **Check file paths**: Ensure model files are in `models/` directory
2. **Check file names**: Verify filenames match what `app.py` expects:
   - `models/efficientnet_b3_finetuned.pth` (fine-tuned model)
   - `models/brain_tumor_classifier_efficientnet_b3.pkl` (EfficientNet classifier)
   - `models/brain_tumor_classifier_resnet50.pkl` (ResNet50 classifier)
   - `models/scaler_efficientnet_b3.pkl` (EfficientNet scaler)
   - `models/scaler_resnet50.pkl` (ResNet50 scaler)

### Memory Issues

If deployment fails due to memory:
- Streamlit Cloud free tier has limited memory
- Consider using only feature extraction models (smaller)
- Or upgrade to Streamlit Cloud Pro

### Build Failures

- Check `requirements.txt` is complete
- Verify Python version compatibility
- Check build logs in Streamlit Cloud dashboard

## Model File Sizes

- `efficientnet_b3_finetuned.pth`: ~123 MB (requires Git LFS or external storage)
- `brain_tumor_classifier_efficientnet_b3.pkl`: ~49 KB
- `brain_tumor_classifier_resnet50.pkl`: ~65 KB
- `scaler_efficientnet_b3.pkl`: ~37 KB
- `scaler_resnet50.pkl`: ~49 KB

## Alternative: Use Hugging Face Spaces

You can also deploy on Hugging Face Spaces:
1. Create account at [huggingface.co](https://huggingface.co)
2. Create a new Space
3. Select Streamlit SDK
4. Upload your code and models
5. Deploy

## Support

For issues, check:
- Streamlit Cloud documentation: [docs.streamlit.io/cloud](https://docs.streamlit.io/cloud)
- GitHub Issues: [github.com/sharma93manvi/brain-tumor-classification/issues](https://github.com/sharma93manvi/brain-tumor-classification/issues)

