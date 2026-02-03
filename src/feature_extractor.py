"""
Feature Extraction Module

Provides classes for extracting features using pretrained models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image
from .preprocessing import ImagePreprocessor


class MedicalFeatureExtractor:
    """
    Feature extractor using pretrained ResNet50 or EfficientNet-B3.
    Optimized for 2D medical image feature extraction.
    """
    
    def __init__(self, model_name='resnet50', device='cpu'):
        """
        Initialize feature extractor
        
        Args:
            model_name: 'resnet50' or 'efficientnet'
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        self.model_name = model_name
        
        # Load pretrained model
        self.encoder = self._load_pretrained_model(model_name)
        self.encoder.eval()
        
        # Freeze all parameters for feature extraction
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def _load_pretrained_model(self, model_name):
        """Load pretrained model and extract feature encoder"""
        if model_name == 'resnet50':
            model = models.resnet50(weights='DEFAULT')
            # Remove final layers (avgpool and fc)
            encoder = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 2048
            
        elif model_name == 'efficientnet':
            model = models.efficientnet_b3(weights='DEFAULT')
            # Remove classifier, keep features
            encoder = model.features
            # Add adaptive pooling
            encoder = nn.Sequential(
                encoder,
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.feature_dim = 1536
            
        else:
            raise ValueError(f"Unknown model: {model_name}. Use 'resnet50' or 'efficientnet'")
        
        encoder.to(self.device)
        return encoder
    
    def preprocess_image(self, image_array):
        """
        Preprocess 2D image for feature extraction with medical preprocessing
        
        Args:
            image_array: numpy array (H, W, 3) or (H, W) RGB or grayscale image
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Step 1: Medical preprocessing
        cropped = ImagePreprocessor.crop_brain_contour(image_array)
        enhanced = ImagePreprocessor.apply_clahe(cropped)
        
        # Step 2: Handle grayscale images
        if len(enhanced.shape) == 2:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        elif len(enhanced.shape) == 3 and enhanced.shape[2] == 1:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        # Step 3: Resize to 224x224 (ImageNet standard)
        resized = cv2.resize(enhanced, (224, 224))
        
        # Step 4: Convert to PIL Image for transforms
        pil_image = Image.fromarray(resized)
        
        # Step 5: Convert to tensor and normalize
        to_tensor = transforms.ToTensor()
        tensor = to_tensor(pil_image)
        tensor = self.normalize(tensor)
        
        # Step 6: Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def extract_features(self, image_array):
        """
        Extract features from image
        
        Args:
            image_array: numpy array image (H, W, 3) or (H, W)
            
        Returns:
            Feature vector as numpy array
        """
        if self.encoder is None:
            raise ValueError("Model not loaded")
        
        # Preprocess image
        tensor = self.preprocess_image(image_array)
        
        # Extract features
        with torch.no_grad():
            features = self.encoder(tensor)
            
            # Global average pooling if needed
            if len(features.shape) == 4:
                features = F.adaptive_avg_pool2d(features, (1, 1))
            
            # Flatten
            features = features.view(features.size(0), -1)
        
        return features.cpu().numpy().flatten()
    
    def get_feature_dimension(self):
        """Get the dimension of extracted features"""
        return self.feature_dim

