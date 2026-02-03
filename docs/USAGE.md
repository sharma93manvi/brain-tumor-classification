# Usage Guide

## Quick Start

### 1. Prepare Your Data

Place your MRI images in the following structure:
```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    └── ...
```

### 2. Run the Notebook

Open `notebooks/brain_tumor_classification.ipynb` and run cells sequentially.

### 3. Train Models

The notebook includes:
- **Feature Extraction**: Extract features using pretrained models
- **Training**: Train classifiers on extracted features
- **Fine-Tuning**: End-to-end training with epochs
- **Evaluation**: Compare different approaches

## Using Trained Models

### Load and Use Saved Models

```python
import joblib
import numpy as np
from src.feature_extractor import MedicalFeatureExtractor
import cv2

# Load model and scaler
model = joblib.load('models/brain_tumor_classifier_efficientnet_b3.pkl')
scaler = joblib.load('models/scaler_efficientnet_b3.pkl')

# Initialize feature extractor
extractor = MedicalFeatureExtractor(model_name='efficientnet', device='cpu')

# Load and preprocess image
image = cv2.imread('path/to/image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Extract features
features = extractor.extract_features(image_rgb)
features_scaled = scaler.transform([features])

# Predict
prediction = model.predict(features_scaled)
probabilities = model.predict_proba(features_scaled)

# Class mapping
classes = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
print(f"Predicted: {classes[prediction[0]]}")
print(f"Confidence: {probabilities[0].max():.2%}")
```

## Model Comparison

The notebook includes comprehensive comparisons:
- Baseline vs Deep Learning
- ResNet50 vs EfficientNet-B3
- Feature Extraction vs Fine-Tuning

## Performance Metrics

Models are evaluated using:
- Accuracy
- Precision, Recall, F1-Score (per class)
- Confusion Matrix
- Cross-Validation

## Tips

1. **GPU Acceleration**: Use GPU for faster training (especially fine-tuning)
2. **Batch Size**: Adjust based on available memory
3. **Epochs**: Fine-tuning typically needs 10-20 epochs
4. **Data Augmentation**: Can improve generalization (future enhancement)

