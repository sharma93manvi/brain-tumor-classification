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
from PIL import UnidentifiedImageError
import os
import sys
import time
import subprocess

# Ensure Git LFS files are pulled (for Streamlit Cloud deployment)
@st.cache_resource
def ensure_lfs_files():
    """Ensure Git LFS files are downloaded on Streamlit Cloud"""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    
    # Check if we're on Streamlit Cloud and files are missing
    model_files = [
        'models/efficientnet_b3_finetuned.pth',
        'models/brain_tumor_classifier_efficientnet_b3.pkl',
        'models/brain_tumor_classifier_resnet50.pkl',
        'models/scaler_efficientnet_b3.pkl',
        'models/scaler_resnet50.pkl'
    ]
    
    missing_files = [f for f in model_files if not os.path.exists(f)]
    
    if missing_files:
        # Try to pull Git LFS files
        try:
            # First, try to initialize Git LFS if not already done
            subprocess.run(['git', 'lfs', 'install'], capture_output=True, timeout=10)
            
            # Then pull LFS files
            result = subprocess.run(
                ['git', 'lfs', 'pull'],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=os.path.dirname(__file__) if '__file__' in globals() else '.'
            )
            if result.returncode == 0:
                st.info("Git LFS files pulled successfully")
            else:
                st.warning(f"Git LFS pull may have failed. Check logs: {result.stderr[:200] if result.stderr else 'No error message'}")
        except FileNotFoundError:
            st.warning("Git LFS not found. Models may not be available. Please ensure Git LFS is installed on Streamlit Cloud.")
        except (subprocess.TimeoutExpired, Exception) as e:
            st.warning(f"Could not pull Git LFS files: {str(e)[:200]}")
    
    return True

# Run on startup (only show warnings, don't block)
try:
    ensure_lfs_files()
except Exception:
    pass  # Fail silently to not break the app

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

# Custom CSS for medical professional styling
st.markdown("""
<style>
    /* Medical Professional Color Scheme */
    :root {
        --medical-blue: #0066cc;
        --medical-light-blue: #e6f2ff;
        --medical-dark-blue: #004499;
        --medical-green: #28a745;
        --medical-red: #dc3545;
        --medical-gray: #6c757d;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0066cc;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .prediction-box {
        padding: 1.5rem;
        border-radius: 8px;
        background-color: #f8f9fa;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    
    .warning-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .clinical-info-box {
        padding: 1.2rem;
        border-radius: 8px;
        background-color: #e6f2ff;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
    }
    
    .disclaimer-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    /* Hide Streamlit branding for professional look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Make sidebar toggle button more visible */
    button[kind="header"] {
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    /* Ensure sidebar toggle is always accessible */
    [data-testid="stSidebar"] {
        position: relative;
    }
    
    /* Make the collapse button more visible */
    .css-1d391kg {
        visibility: visible !important;
    }
    
    /* Professional spacing */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Section headers */
    h2 {
        color: #0066cc;
        font-weight: 600;
        border-bottom: 2px solid #e6f2ff;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    
    h3 {
        color: #004499;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Title and header with medical professional styling
col_title, col_toggle = st.columns([11, 1])
with col_title:
    st.markdown('<h1 class="main-header">🧠 Brain Tumor Classification System</h1>', unsafe_allow_html=True)
with col_toggle:
    # Sidebar toggle button using custom HTML component
    import streamlit.components.v1 as components
    components.html("""
    <style>
        .sidebar-toggle-btn {
            background-color: transparent;
            color: #6c757d;
            border: none;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 18px;
            width: 100%;
            transition: color 0.3s ease, background-color 0.3s ease;
        }
        .sidebar-toggle-btn:hover {
            color: #495057;
            background-color: rgba(108, 117, 125, 0.1);
        }
        .sidebar-toggle-btn:active {
            color: #343a40;
            background-color: rgba(108, 117, 125, 0.2);
        }
    </style>
    <button class="sidebar-toggle-btn" id="sidebarToggleBtn">☰</button>
    <script>
        (function() {
            const btn = document.getElementById('sidebarToggleBtn');
            if (btn) {
                btn.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    
                    // Get the parent window (where Streamlit's UI actually lives)
                    const parentWindow = window.parent;
                    const parentDoc = parentWindow.document;
                    
                    // Try multiple methods to find and click the sidebar toggle
                    const methods = [
                        // Method 1: Direct selector for collapsed control
                        () => {
                            const toggle = parentDoc.querySelector('[data-testid="collapsedControl"]');
                            if (toggle) { toggle.click(); return true; }
                            return false;
                        },
                        // Method 2: Find button in header
                        () => {
                            const header = parentDoc.querySelector('header');
                            if (header) {
                                const buttons = header.querySelectorAll('button');
                                for (let b of buttons) {
                                    if (b.getAttribute('aria-label') && 
                                        (b.getAttribute('aria-label').toLowerCase().includes('sidebar') ||
                                         b.getAttribute('aria-label').toLowerCase().includes('menu'))) {
                                        b.click();
                                        return true;
                                    }
                                }
                            }
                            return false;
                        },
                        // Method 3: Find first button in header (usually the toggle)
                        () => {
                            const header = parentDoc.querySelector('header');
                            if (header) {
                                const firstBtn = header.querySelector('button');
                                if (firstBtn) {
                                    firstBtn.click();
                                    return true;
                                }
                            }
                            return false;
                        },
                        // Method 4: Simulate keyboard shortcut
                        () => {
                            const event = new KeyboardEvent('keydown', {
                                key: 's',
                                code: 'KeyS',
                                ctrlKey: true,
                                shiftKey: true,
                                bubbles: true,
                                cancelable: true
                            });
                            parentDoc.dispatchEvent(event);
                            return true;
                        }
                    ];
                    
                    // Try each method
                    for (let method of methods) {
                        try {
                            if (method()) break;
                        } catch (err) {
                            console.log('Toggle method failed:', err);
                        }
                    }
                });
            }
        })();
    </script>
    """, height=40)

st.markdown("""
<div class="subtitle">
    AI-Powered Clinical Decision Support Tool for MRI Analysis
</div>
""", unsafe_allow_html=True)

# Medical disclaimer banner
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Medical Disclaimer:</strong> This tool is for research and educational purposes only. 
    It is not intended to replace professional medical judgment, diagnosis, or treatment. 
    All predictions should be verified by qualified radiologists and medical professionals.
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

**Important Limitations:**
- The model only recognizes these 4 categories
- Images from other categories will still be assigned to one of these classes
- Very low confidence (<40%) or evenly distributed probabilities indicate the image may not belong to any trained category

**Note**: This is a research tool. For clinical use, consult medical professionals.
""")

# Class names
CLASS_NAMES = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]

# Clinical information for each tumor type
CLINICAL_INFO = {
    "No Tumor": {
        "description": "Normal brain tissue with no evidence of tumor",
        "prevalence": "Most common finding",
        "next_steps": "Routine follow-up as per clinical protocol",
        "significance": "Normal finding"
    },
    "Glioma": {
        "description": "Primary brain tumor arising from glial cells (astrocytes, oligodendrocytes, or ependymal cells)",
        "prevalence": "Most common primary brain tumor (~30% of all brain tumors)",
        "characteristics": "Can be low-grade (benign) or high-grade (malignant). Location and grade determine prognosis.",
        "next_steps": "Requires neurosurgical consultation, biopsy for grading, and multidisciplinary tumor board review",
        "significance": "High clinical significance - requires immediate evaluation"
    },
    "Meningioma": {
        "description": "Tumor arising from the meninges (protective layers surrounding the brain)",
        "prevalence": "Most common benign brain tumor (~37% of all primary brain tumors)",
        "characteristics": "Usually slow-growing and benign. More common in women and older adults.",
        "next_steps": "Neurosurgical evaluation recommended. Small, asymptomatic tumors may be monitored.",
        "significance": "Moderate to high significance - requires neurosurgical assessment"
    },
    "Pituitary": {
        "description": "Tumor of the pituitary gland, often benign adenoma",
        "prevalence": "Common (~15% of all brain tumors)",
        "characteristics": "Can cause hormonal imbalances. May be microadenoma (<1cm) or macroadenoma (>1cm).",
        "next_steps": "Endocrinology and neurosurgery consultation. Hormone level testing recommended.",
        "significance": "Moderate to high significance - requires endocrinological and neurosurgical evaluation"
    }
}

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
        # Try to open with PIL first
        try:
            image = Image.open(uploaded_file)
            image_array = np.array(image)
        except Exception as pil_error:
            # If PIL fails, try reading with OpenCV as fallback
            uploaded_file.seek(0)  # Reset file pointer
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image_array is None:
                raise ValueError("Could not decode image file. The file may be corrupted or in an unsupported format.")
            
            # Convert BGR to RGB for display
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            # Convert numpy array to PIL Image for display
            image = Image.fromarray(image_array)
        
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
        
        # Display original image in a compact size
        st.markdown("### Original Image")
        # Display image at fixed width (400px) for better layout
        st.image(image, width=400, caption=f"Uploaded MRI Scan | Size: {image.size[0]}×{image.size[1]} pixels | Format: {image.format}")
        
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
        
        # Detect out-of-distribution images
        # Calculate entropy to detect if probabilities are too evenly distributed
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(len(CLASS_NAMES))  # Maximum entropy for uniform distribution
        entropy_ratio = entropy / max_entropy
        
        # Flag for potential out-of-distribution
        # High entropy (close to uniform) or very low confidence suggests OOD
        is_potentially_ood = entropy_ratio > 0.85 or confidence < 0.4
        
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
            if is_potentially_ood:
                st.error(f"**{predicted_class}** (Uncertain)")
                st.error("""
                **⚠️ Out-of-Distribution Warning**
                
                The model's prediction is highly uncertain. This image may:
                - Not belong to any of the 4 trained categories
                - Be a different type of medical image (not a brain MRI)
                - Be corrupted or of poor quality
                
                **The model was only trained on:**
                - No Tumor
                - Glioma
                - Meningioma
                - Pituitary Tumor
                
                **Recommendation:** Verify that this is a brain MRI scan. If the image is from a different category or modality, the prediction may not be reliable.
                """)
            elif confidence >= 0.9:
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
            
            # Warning for low confidence (but not OOD)
            if confidence < 0.7 and not is_potentially_ood:
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
        
        # Clinical Information Section
        st.markdown("---")
        st.markdown("### 📋 Clinical Information")
        
        if not is_potentially_ood:
            clinical_info = CLINICAL_INFO[predicted_class]
            
            col_info1, col_info2 = st.columns([1, 1])
            
            with col_info1:
                st.markdown(f"""
                <div class="clinical-info-box">
                    <h4 style="color: #0066cc; margin-top: 0;">About {predicted_class}</h4>
                    <p><strong>Description:</strong> {clinical_info['description']}</p>
                    <p><strong>Prevalence:</strong> {clinical_info['prevalence']}</p>
                    {f"<p><strong>Characteristics:</strong> {clinical_info.get('characteristics', 'N/A')}</p>" if 'characteristics' in clinical_info else ""}
                </div>
                """, unsafe_allow_html=True)
            
            with col_info2:
                st.markdown(f"""
                <div class="clinical-info-box">
                    <h4 style="color: #0066cc; margin-top: 0;">Clinical Significance</h4>
                    <p><strong>Significance:</strong> {clinical_info['significance']}</p>
                    <p><strong>Recommended Next Steps:</strong></p>
                    <p>{clinical_info['next_steps']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional clinical notes
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-top: 1rem;">
                <strong>Important Clinical Notes:</strong>
                <ul style="margin-top: 0.5rem;">
                    <li>This AI prediction is a screening tool and should not replace clinical judgment</li>
                    <li>All findings require correlation with patient history, physical examination, and other imaging studies</li>
                    <li>Final diagnosis should be confirmed by board-certified radiologists</li>
                    <li>Treatment decisions should be made by qualified medical professionals</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("""
            **Clinical Information Not Available**
            
            Due to high uncertainty in the prediction, clinical information cannot be reliably provided. 
            Please verify the image quality and ensure it is a brain MRI scan before proceeding with clinical interpretation.
            """)
    
    except UnidentifiedImageError as e:
        st.error("**Image Format Error**")
        st.warning(f"""
        The uploaded file '{uploaded_file.name}' could not be identified as a valid image file.
        
        **Possible causes:**
        - File is corrupted or incomplete
        - File extension doesn't match the actual format
        - Unsupported image format
        
        **Please try:**
        - Re-uploading the image file
        - Converting the image to PNG or JPEG format
        - Checking if the file is not corrupted
        """)
        st.exception(e)
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
        <div style="padding: 1rem; background-color: #e6f2ff; border-radius: 5px;">
            <strong>No Tumor</strong>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Normal brain tissue with no evidence of tumor. Most common finding in routine screening</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="padding: 1rem; background-color: #ffe6e6; border-radius: 5px;">
            <strong>Glioma</strong>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Primary brain tumor from glial cells. Most common primary brain tumor (~30%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="padding: 1rem; background-color: #fff3cd; border-radius: 5px;">
            <strong>Meningioma</strong>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Tumor from meninges. Most common benign brain tumor (~37%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="padding: 1rem; background-color: #d4edda; border-radius: 5px;">
            <strong>Pituitary</strong>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Pituitary gland tumor. Common (~15%), often benign adenoma</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Professional Footer
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 2rem; border-radius: 8px; margin-top: 3rem; text-align: center;">
        <h4 style="color: #0066cc; margin-bottom: 1rem;">Medical AI Classification System</h4>
        <p style="color: #6c757d; font-size: 0.9rem; margin-bottom: 0.5rem;">
            <strong>Intended Use:</strong> Research and educational purposes | Clinical decision support tool
        </p>
        <p style="color: #6c757d; font-size: 0.85rem; margin-bottom: 1rem;">
            This system is designed to assist qualified healthcare professionals and should not replace clinical judgment.
        </p>
        <div style="border-top: 1px solid #dee2e6; padding-top: 1rem; margin-top: 1rem;">
            <p style="color: #6c757d; font-size: 0.8rem; margin: 0;">
                <strong>Disclaimer:</strong> All predictions require verification by board-certified radiologists. 
                This tool does not provide medical diagnosis or treatment recommendations.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

