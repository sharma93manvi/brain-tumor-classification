"""
Streamlit App for Brain Tumor Classification

A web application for classifying brain tumors from MRI scans using deep learning.
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from PIL import UnidentifiedImageError
import os
import sys
import time
import subprocess
import requests
from pathlib import Path

# Download model files from GitHub if not available locally (for Streamlit Cloud)
@st.cache_resource
def download_model_if_missing(filepath, github_url=None):
    """Download model file from GitHub if it doesn't exist locally"""
    if os.path.exists(filepath):
        return True
    
    # If GitHub URL provided, download from there
    if github_url:
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            response = requests.get(github_url, stream=True, timeout=300)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
        except Exception as e:
            st.warning(f"Could not download {os.path.basename(filepath)}: {str(e)[:100]}")
    
    return False

# Ensure models directory exists
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir, exist_ok=True)

# Note: Models should be available via Git LFS or need to be hosted externally
# For now, the app will work with whatever models are available locally

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
        display: block;
        width: 100%;
        margin-left: auto;
        margin-right: auto;
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
    st.markdown("""
    <div class="subtitle">AI-Powered Clinical Decision Support Tool for MRI Analysis</div>
    """, unsafe_allow_html=True)
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
effnet_available = os.path.exists('models/brain_tumor_classifier_efficientnet_b3.pkl') and os.path.exists('models/scaler_efficientnet_b3.pkl')
resnet_available = os.path.exists('models/brain_tumor_classifier_resnet50.pkl') and os.path.exists('models/scaler_resnet50.pkl')

# Build model options list
model_options = []
if fine_tuned_available:
    model_options.append("EfficientNet-B3 Fine-Tuned (Best - 97.64%)")
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
            classifier_path = 'models/brain_tumor_classifier_efficientnet_b3.pkl'
            scaler_path = 'models/scaler_efficientnet_b3.pkl'
        else:  # resnet50
            classifier_path = 'models/brain_tumor_classifier_resnet50.pkl'
            scaler_path = 'models/scaler_resnet50.pkl'
        
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

def generate_gradcam(model, tensor, predicted_class_idx, enhanced_image):
    """Generate Grad-CAM heatmap for tumor localization"""
    try:
        # Ensure model is in eval mode but allows gradients
        model.eval()
        
        # Register hook to get activations from the last convolutional layer
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                gradients.append(grad_output[0])
        
        # Find the last convolutional layer in EfficientNet
        target_layer = None
        
        # Try to find in features submodule (EfficientNet structure)
        if hasattr(model, 'features'):
            for name, module in model.features.named_modules():
                if isinstance(module, nn.Conv2d):
                    target_layer = module
        
        # Fallback: search all modules
        if target_layer is None:
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    target_layer = module
        
        if target_layer is None:
            return None
        
        # Register hooks
        handle_forward = target_layer.register_forward_hook(forward_hook)
        handle_backward = target_layer.register_backward_hook(backward_hook)
        
        # Forward pass with gradients (tensor should already have requires_grad=True)
        # Ensure model allows gradients
        for param in model.parameters():
            param.requires_grad = True
        
        outputs = model(tensor)
        
        # Backward pass
        model.zero_grad()
        # Ensure the output requires grad
        if not outputs.requires_grad:
            outputs = outputs.requires_grad_(True)
        outputs[0, predicted_class_idx].backward()
        
        # Get gradients and activations
        if len(gradients) == 0 or len(activations) == 0:
            handle_forward.remove()
            handle_backward.remove()
            return None
        
        grads = gradients[0]
        acts = activations[0]
        
        # Compute weights (global average pooling of gradients)
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)  # Apply ReLU
        
        # Normalize
        cam = cam.squeeze().cpu().detach().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to match original image size
        cam_resized = cv2.resize(cam, (enhanced_image.shape[1], enhanced_image.shape[0]))
        
        # Remove hooks
        handle_forward.remove()
        handle_backward.remove()
        
        return cam_resized
        
    except Exception as e:
        # Log error for debugging
        import traceback
        error_msg = f"Grad-CAM error: {str(e)}\n{traceback.format_exc()}"
        # Store error in session state for debugging (only first error)
        if 'gradcam_error' not in st.session_state:
            st.session_state.gradcam_error = error_msg
        # Print to console for Streamlit Cloud logs
        print(f"Grad-CAM Error: {str(e)}")
        print(traceback.format_exc())
        return None

def overlay_heatmap(image, heatmap, alpha=0.4, threshold_percentile=75):
    """Overlay heatmap on image with improved contrast for tumor regions"""
    # Apply thresholding to highlight only high-attention regions
    threshold = np.percentile(heatmap, threshold_percentile)
    
    # Create a masked heatmap that emphasizes high-attention regions
    masked_heatmap = np.where(heatmap >= threshold, heatmap, heatmap * 0.3)
    
    # Normalize the masked heatmap
    if masked_heatmap.max() > 0:
        masked_heatmap = (masked_heatmap - masked_heatmap.min()) / (masked_heatmap.max() - masked_heatmap.min() + 1e-8)
    
    # Convert heatmap to 0-255 range
    heatmap_uint8 = (masked_heatmap * 255).astype(np.uint8)
    
    # Apply colormap (jet: blue=low, red=high)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Ensure image is RGB
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
    
    # Resize heatmap if needed
    if heatmap_colored.shape[:2] != image_rgb.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, (image_rgb.shape[1], image_rgb.shape[0]))
    
    # Overlay with higher alpha for better visibility of tumor regions
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay

def predict_fine_tuned(model, image_array, return_gradcam=False):
    """Predict using fine-tuned model"""
    tensor, enhanced = preprocess_for_fine_tuning(image_array)
    
    # For Grad-CAM, we need a separate tensor with gradients enabled
    if return_gradcam:
        # Clone and detach, then enable gradients
        tensor_gradcam = tensor.clone().detach()
        tensor_gradcam.requires_grad_(True)
        # Ensure it's on the same device as the model
        if next(model.parameters()).is_cuda:
            tensor_gradcam = tensor_gradcam.cuda()
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    predicted_idx = np.argmax(probabilities[0].numpy())
    
    gradcam_heatmap = None
    if return_gradcam:
        # Use the tensor with gradients enabled for Grad-CAM
        try:
            gradcam_heatmap = generate_gradcam(model, tensor_gradcam, predicted_idx, enhanced)
            if gradcam_heatmap is None:
                print("Warning: generate_gradcam returned None")
        except Exception as e:
            print(f"Error in generate_gradcam: {str(e)}")
            import traceback
            print(traceback.format_exc())
            gradcam_heatmap = None
    
    return probabilities[0].numpy(), enhanced, gradcam_heatmap

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
        
        # Display original image centered with small size
        st.markdown("### Original Image")
        col_left, col_img, col_right = st.columns([1, 2, 1])
        
        with col_img:
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
            
            st.info("""
            **Note:** Additional preprocessing steps (resize to 300×300, tensor conversion, and ImageNet normalization) 
            are performed automatically before model inference but are not shown here for clarity.
            """)
        
        # Make prediction
        st.markdown("---")
        st.markdown("### Prediction Results")
        
        # Start timing
        start_time = time.time()
        
        # Initialize gradcam_heatmap variable
        gradcam_heatmap = None
        
        with st.spinner("Loading model and making prediction..."):
            if "Fine-Tuned" in model_type:
                model = load_fine_tuned_model()
                if model is not None:
                    # Generate Grad-CAM for fine-tuned model
                    try:
                        probabilities, processed_img, gradcam_heatmap = predict_fine_tuned(model, image_array, return_gradcam=True)
                        # Debug: Check if gradcam_heatmap was generated
                        if gradcam_heatmap is None:
                            print("Warning: Grad-CAM heatmap is None after prediction")
                    except Exception as e:
                        print(f"Error during prediction with Grad-CAM: {str(e)}")
                        import traceback
                        print(traceback.format_exc())
                        # Fallback: try without Grad-CAM
                        probabilities, processed_img, gradcam_heatmap = predict_fine_tuned(model, image_array, return_gradcam=False)
                else:
                    st.error("**Fine-Tuned Model Not Available**")
                    st.warning("""
                    The fine-tuned EfficientNet-B3 model file (`models/efficientnet_b3_finetuned.pth`) is not found.
                    
                    **To use the fine-tuned model:**
                    1. Train the model using the notebook: `notebooks/brain_tumor_classification.ipynb`
                    2. Or copy the model file from your Colab/Google Drive to `models/efficientnet_b3_finetuned.pth`
                    
                    **Note:** On Streamlit Cloud, the fine-tuned model file (~123MB) is tracked by Git LFS and may need to be downloaded separately.
                    
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
                    - `models/brain_tumor_classifier_efficientnet_b3.pkl`
                    - `models/scaler_efficientnet_b3.pkl`
                    
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
                    - `models/brain_tumor_classifier_resnet50.pkl`
                    - `models/scaler_resnet50.pkl`
                    
                    **To generate these files:**
                    Run the training cells in `notebooks/brain_tumor_classification.ipynb`
                    """)
                    st.stop()
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # gradcam_heatmap is already initialized above (line 722) and set by predict_fine_tuned if Fine-Tuned model
        # For non-Fine-Tuned models, ensure it's None
        if "Fine-Tuned" not in model_type:
            gradcam_heatmap = None
        
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
        
        # Visualization - Probability Distribution Chart (in expander for space efficiency)
        with st.expander("View Probability Distribution Chart", expanded=False):
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(6, 3))  # Reduced to smaller size
            colors = ['#2ecc71' if i == predicted_idx else '#95a5a6' for i in range(len(CLASS_NAMES))]
            bars = ax.barh(CLASS_NAMES, probabilities, color=colors)
            ax.set_xlabel("Probability", fontsize=10)
            ax.set_title("Class Probability Distribution", fontsize=11, fontweight='bold')
            ax.set_xlim(0, 1.15)  # Increased xlim to make room for labels
            
            # Add value labels on bars - positioned inside bars if space allows, otherwise outside
            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                width = bar.get_width()
                # Position label inside bar if probability > 0.15, otherwise outside
                if prob > 0.15:
                    # Inside the bar, white text for visibility
                    ax.text(width / 2, bar.get_y() + bar.get_height()/2, 
                           f'{prob*100:.2f}%', ha='center', va='center', 
                           fontsize=9, fontweight='bold', color='white')
                else:
                    # Outside the bar, to the right
                    ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                           f'{prob*100:.2f}%', ha='left', va='center', 
                           fontsize=9, fontweight='bold', color='#333333')
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        
        # Tumor Localization Visualization (Grad-CAM) - Only for Fine-Tuned model and tumor classes
        if "Fine-Tuned" in model_type and gradcam_heatmap is not None and not is_potentially_ood:
            # Only show Grad-CAM for tumor classes (not for "No Tumor")
            if predicted_class != "No Tumor":
                st.markdown("---")
                st.markdown("### Tumor Localization")
                st.markdown("""
                <div style="background-color: #e6f2ff; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;">
                    <strong>Grad-CAM Heatmap:</strong> This visualization shows which regions of the MRI image 
                    the model focuses on when making its prediction. <strong>Red/yellow areas</strong> indicate high attention 
                    (likely tumor regions), while <strong>blue areas</strong> indicate low attention (normal tissue).
                </div>
                """, unsafe_allow_html=True)
                
                col_orig, col_heatmap = st.columns(2)
                
                with col_orig:
                    st.markdown("#### Original Preprocessed Image")
                    st.image(processed_img, use_container_width=True, caption="Preprocessed MRI scan")
                
                with col_heatmap:
                    st.markdown("#### Tumor Localization Heatmap")
                    # Overlay heatmap on processed image with improved contrast for tumor classes
                    # Using 75th percentile threshold to highlight only high-attention regions
                    overlay = overlay_heatmap(processed_img, gradcam_heatmap, alpha=0.6, threshold_percentile=75)
                    st.image(overlay, use_container_width=True, caption="Red/yellow = High attention (tumor region)")
                
                st.info("""
                **Note:** This visualization is an approximation based on the model's attention. 
                It shows where the model focuses but may not perfectly align with exact tumor boundaries. 
                For precise tumor segmentation, a dedicated segmentation model would be required.
                """)
            else:
                # For "No Tumor" predictions, show a different message
                st.markdown("---")
                st.markdown("### Tumor Localization")
                st.markdown("""
                <div style="background-color: #e6f2ff; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;">
                    <strong>No Tumor Detected:</strong> The model has classified this image as "No Tumor". 
                    Tumor localization visualization is only shown when a tumor is detected.
                </div>
                """, unsafe_allow_html=True)
        elif "Fine-Tuned" in model_type and gradcam_heatmap is None:
            st.markdown("---")
            st.markdown("### Tumor Localization")
            error_info = ""
            if 'gradcam_error' in st.session_state:
                error_info = f"\n\n**Debug Info:** {st.session_state.gradcam_error[:200]}"
            st.warning(f"""
            **Tumor localization visualization could not be generated.**
            
            Possible reasons:
            - Grad-CAM generation encountered an error
            - Model architecture compatibility issue
            - The fine-tuned model file may not be fully loaded on Streamlit Cloud
            {error_info}
            
            **Note:** Grad-CAM visualization requires the fine-tuned model to be available and properly loaded.
            """)
        
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
        - Accuracy: 97.64%
        - F1-Score: 0.98
        
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

