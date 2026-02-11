# Streamlit Web Application Guide

## Overview

The Streamlit web application provides an interactive interface for brain tumor classification from MRI scans. Users can upload images and get instant predictions using our trained deep learning models.

## Features

- **Multiple Model Support**: Choose between EfficientNet-B3 Fine-Tuned (97.64% accuracy), EfficientNet-B3 Feature Extraction (91.34%), or ResNet50 Feature Extraction (91.34%)
- **Real-time Preprocessing Visualization**: See brain contour cropping and CLAHE enhancement steps
- **Detailed Predictions**: View class probabilities, confidence scores, and probability distributions
- **User-Friendly Interface**: Clean, intuitive design with clear visualizations

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Model Files Exist**:
   - `models/efficientnet_b3_finetuned.pth` (for fine-tuned model)
   - `models/brain_tumor_classifier_efficientnet_b3.pkl` and `models/scaler_efficientnet_b3.pkl` (for EfficientNet feature extraction)
   - `models/brain_tumor_classifier_resnet50.pkl` and `models/scaler_resnet50.pkl` (for ResNet50 feature extraction)

## Running the App

### Option 1: Using the Script
```bash
./scripts/run_app.sh
```

### Option 2: Direct Command
```bash
streamlit run app.py
```

### Option 3: Custom Port
```bash
streamlit run app.py --server.port 8501
```

## Usage

1. **Upload Image**: Click the upload area and select a brain MRI scan (PNG, JPG, or JPEG)

2. **Select Model**: Choose your preferred model from the sidebar:
   - **EfficientNet-B3 Fine-Tuned**: Best accuracy (97.64%)
   - **EfficientNet-B3 Feature Extraction**: Fast inference (91.34%)
   - **ResNet50 Feature Extraction**: Alternative approach (91.34%)

3. **View Results**: 
   - See preprocessing steps (cropping and enhancement)
   - View class probabilities with progress bars
   - Check the predicted class and confidence score
   - Examine probability distribution chart

## Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository and branch
6. Set main file to `app.py`
7. Click "Deploy"

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t brain-tumor-app .
docker run -p 8501:8501 brain-tumor-app
```

### Local Network Access

To access the app from other devices on your network:
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Then access via: `http://YOUR_IP_ADDRESS:8501`

## Troubleshooting

### Model Files Not Found
- Ensure model files are in the `models/` directory
- Check file names match exactly: 
  - `efficientnet_b3_finetuned.pth` (fine-tuned model)
  - `brain_tumor_classifier_efficientnet_b3.pkl` and `scaler_efficientnet_b3.pkl` (EfficientNet feature extraction)
  - `brain_tumor_classifier_resnet50.pkl` and `scaler_resnet50.pkl` (ResNet50 feature extraction)
- Run the training notebook to generate model files if missing

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that `src/` directory is accessible
- Verify Python version (3.8+)

### Image Upload Issues
- Supported formats: PNG, JPG, JPEG
- Ensure image is a valid brain MRI scan
- Check image file size (very large images may cause issues)

### Low Confidence Predictions
- Model confidence below 70% indicates uncertainty
- Try different preprocessing or model
- Consult medical professionals for clinical decisions

## Performance Tips

- **CPU Mode**: App runs on CPU by default (slower but works everywhere)
- **GPU Support**: For faster inference, ensure CUDA is available and PyTorch is GPU-enabled
- **Caching**: Models are cached using `@st.cache_resource` for faster reloads
- **Batch Processing**: Currently supports single image upload (batch support can be added)

## Security Notes

- This is a research/educational tool
- Not intended for clinical diagnosis
- Always consult medical professionals
- Model predictions should not replace expert medical judgment

## Future Enhancements

- Batch image processing
- Model comparison side-by-side
- Export results as PDF/CSV
- Historical prediction tracking
- API endpoint for programmatic access


