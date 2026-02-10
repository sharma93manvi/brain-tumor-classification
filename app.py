"""
Streamlit App for Brain Tumor Classification

A web application for classifying brain tumors from MRI scans using deep learning.
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import os
import sys
import time

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Force reload modules to avoid caching issues
if 'src.preprocessing' in sys.modules:
    import importlib
    importlib.reload(sys.modules['src.preprocessing'])
if 'src.feature_extractor' in sys.modules:
    import importlib
    importlib.reload(sys.modules['src.feature_extractor'])

from src.preprocessing import ImagePreprocessor
from src.feature_extractor import MedicalFeatureExtractor
from torchvision import models, transforms
from torchvision.models import EfficientNet_B3_Weights
import joblib

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<h1 class="main-header">Brain Tumor Classification from MRI Scans</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    Upload an MRI scan to classify brain tumors using deep learning models
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("Configuration")
st.sidebar.markdown("---")

# Check which models are available
fine_tuned_available = os.path.exists('models/efficientnet_b3_finetuned.pth')
effnet_available = (os.path.exists('models/model_effnet.pkl') or os.path.exists('models/brain_tumor_classifier_efficientnet_b3.pkl')) and (os.path.exists('models/scaler_effnet.pkl') or os.path.exists('models/scaler_efficientnet_b3.pkl'))
resnet_available = (os.path.exists('models/model_resnet.pkl') or os.path.exists('models/brain_tumor_classifier_resnet50.pkl')) and (os.path.exists('models/scaler_resnet.pkl') or os.path.exists('models/scaler_resnet50.pkl'))

# Build model options list
model_options = []
if fine_tuned_available:
    model_options.append("EfficientNet-B3 Fine-Tuned (Best - 97.20%)")
if effnet_available:
    model_options.append("EfficientNet-B3 Feature Extraction (91.34%)")
if resnet_available:
    model_options.append("ResNet50 Feature Extraction (91.34%)")

if not model_options:
    st.sidebar.error("No models available. Please train models first.")
    model_options = ["No models available"]

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model",
    model_options,
    help="Choose the model for classification. Fine-tuned model offers best accuracy."
)

# Confidence interpretation guide
st.sidebar.markdown("---")
st.sidebar.markdown("### Confidence Levels")
st.sidebar.info("""
**Understanding Model Confidence:**

- **≥90%**: High confidence
  - Model is very certain about prediction
  - Still requires professional verification

- **70-90%**: Moderate confidence
  - Model is reasonably certain
  - Professional consultation recommended

- **<70%**: Low confidence
  - Model is uncertain
  - **Consult medical professional**
  - May indicate ambiguous image or edge case
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This tool classifies brain MRI scans into:
- **No Tumor**
- **Glioma**
- **Meningioma**
- **Pituitary Tumor**

**Note**: This is a research tool. For clinical use, consult medical professionals.
""")

# Class names
CLASS_NAMES = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]

@st.cache_resource(show_spinner=False)
def load_fine_tuned_model():
    """Load the fine-tuned EfficientNet-B3 model"""
    try:
        model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        
        # Replace classifier for 4 classes
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, 4)
        )
        
        # Load saved weights
        model_path = 'models/efficientnet_b3_finetuned.pth'
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            return model
        else:
            # Model file not found - return None (will show helpful message in UI)
            return None
    except Exception as e:
        st.error(f"Error loading fine-tuned model: {str(e)}")
        return None

@st.cache_resource(show_spinner=False)
def load_feature_extraction_model(model_name):
    """Load feature extraction model and classifier"""
    try:
        # Load feature extractor
        extractor = MedicalFeatureExtractor(model_name=model_name, device='cpu')
        
        # Load classifier and scaler
        if model_name == 'efficientnet':
            # Try new naming first, fallback to old naming
            if os.path.exists('models/brain_tumor_classifier_efficientnet_b3.pkl'):
                classifier_path = 'models/brain_tumor_classifier_efficientnet_b3.pkl'
                scaler_path = 'models/scaler_efficientnet_b3.pkl'
            else:
                classifier_path = 'models/model_effnet.pkl'
                scaler_path = 'models/scaler_effnet.pkl'
        else:  # resnet50
            # Try new naming first, fallback to old naming
            if os.path.exists('models/brain_tumor_classifier_resnet50.pkl'):
                classifier_path = 'models/brain_tumor_classifier_resnet50.pkl'
                scaler_path = 'models/scaler_resnet50.pkl'
            else:
                classifier_path = 'models/model_resnet.pkl'
                scaler_path = 'models/scaler_resnet.pkl'
        
        if os.path.exists(classifier_path) and os.path.exists(scaler_path):
            classifier = joblib.load(classifier_path)
            scaler = joblib.load(scaler_path)
            return extractor, classifier, scaler
        else:
            st.error(f"Model files not found: {classifier_path} or {scaler_path}")
            return None, None, None
    except Exception as e:
        st.error(f"Error loading feature extraction model: {str(e)}")
        return None, None, None

def preprocess_for_fine_tuning(image_array):
    """Preprocess image for fine-tuned model (300x300)"""
    preprocessor = ImagePreprocessor()
    
    # Medical preprocessing
    cropped = preprocessor.crop_brain_contour(image_array)
    enhanced = preprocessor.apply_clahe(cropped)
    
    # Convert to RGB if grayscale
    if len(enhanced.shape) == 2:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    elif len(enhanced.shape) == 3 and enhanced.shape[2] == 1:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    # Resize to 300x300 (as used in training)
    resized = cv2.resize(enhanced, (300, 300))
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    pil_image = Image.fromarray(resized)
    tensor = transform(pil_image).unsqueeze(0)
    
    return tensor, enhanced

def predict_fine_tuned(model, image_array):
    """Predict using fine-tuned model"""
    tensor, enhanced = preprocess_for_fine_tuning(image_array)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    return probabilities[0].numpy(), enhanced

def predict_feature_extraction(extractor, classifier, scaler, image_array):
    """Predict using feature extraction approach"""
    features = extractor.extract_features(image_array)
    features_scaled = scaler.transform(features.reshape(1, -1))
    probabilities = classifier.predict_proba(features_scaled)[0]
    
    # Get preprocessed image for visualization
    preprocessor = ImagePreprocessor()
    cropped = preprocessor.crop_brain_contour(image_array)
    enhanced = preprocessor.apply_clahe(cropped)
    
    return probabilities, enhanced

# File uploader
st.markdown("### Upload MRI Image")
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['png', 'jpg', 'jpeg'],
    help="Upload a brain MRI scan image (PNG, JPG, or JPEG format)"
)

if uploaded_file is not None:
    # Load and display image
    try:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Handle different image formats
        if len(image_array.shape) == 3:
            # RGB or RGBA image
            if image_array.shape[2] == 4:
                # RGBA - convert to RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            elif image_array.shape[2] == 1:
                # Single channel - convert to grayscale (2D)
                image_array = image_array[:, :, 0]
        # If already 2D (grayscale), keep as is
        
        # Display original image
        st.markdown("### Original Image")
        st.image(image, use_container_width=True, caption=f"Uploaded MRI Scan | Size: {image.size[0]}×{image.size[1]} pixels | Format: {image.format}")
        
        # Preprocessing visualization in expander
        with st.expander("🔍 View Preprocessing Steps", expanded=False):
            try:
                preprocessor = ImagePreprocessor()
                cropped = preprocessor.crop_brain_contour(image_array)
                enhanced = preprocessor.apply_clahe(cropped)
            except Exception as e:
                st.error(f"Preprocessing error: {str(e)}")
                st.info(f"Image shape: {image_array.shape}, dtype: {image_array.dtype}")
                # Fallback: try to convert to RGB if grayscale
                if len(image_array.shape) == 2:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
                    preprocessor = ImagePreprocessor()
                    cropped = preprocessor.crop_brain_contour(image_array)
                    enhanced = preprocessor.apply_clahe(cropped)
                else:
                    raise
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(cropped, caption="1. Brain Contour Cropping", use_container_width=True)
            with col2:
                st.image(enhanced, caption="2. CLAHE Enhancement", use_container_width=True)
        
        # Make prediction
        st.markdown("---")
        st.markdown("### Prediction Results")
        
        # Start timing
        start_time = time.time()
        
        with st.spinner("Loading model and making prediction..."):
            if "Fine-Tuned" in model_type:
                model = load_fine_tuned_model()
                if model is not None:
                    probabilities, processed_img = predict_fine_tuned(model, image_array)
                else:
                    st.error("**Fine-Tuned Model Not Available**")
                    st.warning("""
                    The fine-tuned EfficientNet-B3 model file (`models/efficientnet_b3_finetuned.pth`) is not found.
                    
                    **To use the fine-tuned model:**
                    1. Train the model using the notebook: `notebooks/brain_tumor_classification.ipynb`
                    2. Or copy the model file from your Colab/Google Drive to `models/efficientnet_b3_finetuned.pth`
                    
                    **Alternative:** Use the Feature Extraction models which are available:
                    - EfficientNet-B3 Feature Extraction (91.34% accuracy)
                    - ResNet50 Feature Extraction (91.34% accuracy)
                    """)
                    st.info("Please select a different model from the sidebar to make predictions.")
                    st.stop()
            elif "EfficientNet-B3" in model_type:
                extractor, classifier, scaler = load_feature_extraction_model('efficientnet')
                if extractor is not None and classifier is not None:
                    probabilities, processed_img = predict_feature_extraction(
                        extractor, classifier, scaler, image_array
                    )
                else:
                    st.error("**EfficientNet-B3 Model Not Available**")
                    st.warning("""
                    Model files not found. Please ensure these files exist:
                    - `models/brain_tumor_classifier_efficientnet_b3.pkl` or `models/model_effnet.pkl`
                    - `models/scaler_efficientnet_b3.pkl` or `models/scaler_effnet.pkl`
                    
                    **To generate these files:**
                    Run the training cells in `notebooks/brain_tumor_classification.ipynb`
                    """)
                    st.stop()
            else:  # ResNet50
                extractor, classifier, scaler = load_feature_extraction_model('resnet50')
                if extractor is not None and classifier is not None:
                    probabilities, processed_img = predict_feature_extraction(
                        extractor, classifier, scaler, image_array
                    )
                else:
                    st.error("**ResNet50 Model Not Available**")
                    st.warning("""
                    Model files not found. Please ensure these files exist:
                    - `models/brain_tumor_classifier_resnet50.pkl` or `models/model_resnet.pkl`
                    - `models/scaler_resnet50.pkl` or `models/scaler_resnet.pkl`
                    
                    **To generate these files:**
                    Run the training cells in `notebooks/brain_tumor_classification.ipynb`
                    """)
                    st.stop()
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Find predicted class
        predicted_idx = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        # Display results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Class Probabilities")
            
            # Display probabilities with progress bars
            for i, (cls, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
                st.markdown(f"**{cls}**")
                # Convert numpy float32 to Python float for Streamlit progress bar
                prob_float = float(prob)
                st.progress(prob_float, text=f"{prob_float*100:.2f}%")
                st.markdown("")
        
        with col2:
            st.markdown("#### Prediction")
            
            # Display prediction with confidence
            if confidence >= 0.9:
                st.success(f"**{predicted_class}**")
            elif confidence >= 0.7:
                st.info(f"**{predicted_class}**")
            else:
                st.warning(f"**{predicted_class}**")
            
            # Display metrics
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("Confidence", f"{confidence*100:.2f}%")
            with col_metric2:
                st.metric("Inference Time", f"{inference_time:.2f}s")
            
            # Warning for low confidence
            if confidence < 0.7:
                st.markdown("""
                <div class="warning-box">
                    <strong>Low Confidence Prediction</strong><br>
                    The model's confidence is below 70%. Please consult a medical professional for accurate diagnosis.
                </div>
                """, unsafe_allow_html=True)
            
            # Additional info - only show if second prediction is close
            top2_indices = np.argsort(probabilities)[-2:][::-1]
            if probabilities[top2_indices[1]] > 0.15:  # Show if second prediction is >15%
                st.markdown("---")
                st.markdown("**Top 2 Predictions:**")
                for idx in top2_indices:
                    st.markdown(f"- {CLASS_NAMES[idx]}: {probabilities[idx]*100:.2f}%")
        
        # Visualization
        st.markdown("---")
        st.markdown("### Probability Distribution")
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2ecc71' if i == predicted_idx else '#95a5a6' for i in range(len(CLASS_NAMES))]
        bars = ax.barh(CLASS_NAMES, probabilities, color=colors)
        ax.set_xlabel("Probability", fontsize=12)
        ax.set_title("Class Probability Distribution", fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add value labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{prob*100:.2f}%', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.exception(e)

else:
    # Show instructions when no file is uploaded
    st.info("Please upload an MRI image to get started")
    
    st.markdown("---")
    st.markdown("### How to Use")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 1. Upload Image
        Click the upload area above and select a brain MRI scan image (PNG, JPG, or JPEG format)
        """)
    
    with col2:
        st.markdown("""
        #### 2. Select Model
        Choose your preferred model from the sidebar dropdown menu.
        """)
    
    with col3:
        st.markdown("""
        #### 3. View Results
        See preprocessing steps, class probabilities, and the final prediction with confidence scores
        """)
    
    st.markdown("---")
    st.markdown("### Model Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Fine-Tuned EfficientNet-B3** (Recommended)
        
        **Performance:**
        - Accuracy: 97.20%
        - F1-Score: 0.972
        
        **Best for:**
        - Clinical decision-making
        - Highest accuracy required
        """)
        if fine_tuned_available:
            st.success("Available")
        else:
            st.warning("Not available")
    
    with col2:
        st.markdown("""
        **EfficientNet-B3 Feature Extraction**
        
        **Performance:**
        - Accuracy: 91.34%
        - F1-Score: 0.913
        
        **Best for:**
        - Fast inference
        - Resource-constrained environments
        """)
        if effnet_available:
            st.success("Available")
        else:
            st.warning("Not available")
    
    with col3:
        st.markdown("""
        **ResNet50 Feature Extraction**
        
        **Performance:**
        - Accuracy: 91.34%
        - F1-Score: 0.913
        
        **Best for:**
        - Baseline comparison
        - Fast inference
        """)
        if resnet_available:
            st.success("Available")
        else:
            st.warning("Not available")
    
    st.markdown("---")
    st.markdown("### Classification Classes")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **No Tumor**
        - Normal brain tissue
        """)
    
    with col2:
        st.markdown("""
        **Glioma**
        - Most common primary brain tumor
        """)
    
    with col3:
        st.markdown("""
        **Meningioma**
        - Usually benign tumor
        """)
    
    with col4:
        st.markdown("""
        **Pituitary**
        - Pituitary gland tumor
        """)
    
    st.markdown("---")
    st.markdown("### Important Disclaimer")
    st.warning("""
    **This is a research tool for educational purposes only.**
    
    - Not intended for clinical diagnosis
    - Always consult qualified medical professionals
    - Results should not be used as a substitute for professional medical advice
    - Model performance may vary with different image qualities and acquisition protocols
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>Brain Tumor Classification System | Built with Streamlit & PyTorch</small>
</div>
""", unsafe_allow_html=True)

