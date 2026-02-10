# Quick Start Guide - Streamlit App

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
cd brain-tumor-classification
pip install -r requirements.txt
```

### Step 2: Ensure Model Files Exist

Make sure you have trained models in the `models/` directory:

- `models/efficientnet_b3_finetuned.pth` (for fine-tuned model)
- `models/model_effnet.pkl` and `models/scaler_effnet.pkl` (for EfficientNet feature extraction)
- `models/model_resnet.pkl` and `models/scaler_resnet.pkl` (for ResNet50 feature extraction)

**Note**: If you don't have these files, run the training cells in `notebooks/brain_tumor_classification.ipynb` first.

### Step 3: Run the App

```bash
streamlit run app.py
```

Or use the script:
```bash
./scripts/run_app.sh
```

The app will open automatically in your browser at `http://localhost:8501`

## 📸 Using the App

1. **Upload an Image**: Click the upload area and select a brain MRI scan
2. **Select Model**: Choose from the sidebar (Fine-Tuned recommended for best accuracy)
3. **View Results**: See preprocessing steps, probabilities, and predictions

## 🐛 Troubleshooting

### "Model file not found" Error
- Run the training notebook first to generate model files
- Check that files are in the `models/` directory
- Verify file names match exactly

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+)

### App Won't Start
- Check if port 8501 is already in use
- Try: `streamlit run app.py --server.port 8502`

## 🌐 Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select repository and set main file to `app.py`
6. Click "Deploy"

## 📚 More Information

- Full documentation: `docs/STREAMLIT_APP.md`
- Project README: `README.md`


