"""
Medical Image Preprocessing Module

Provides functions for brain contour cropping and CLAHE enhancement.
"""

import cv2
import numpy as np


class ImagePreprocessor:
    """Medical image preprocessing utilities"""
    
    @staticmethod
    def crop_brain_contour(image_array):
        """
        Remove black background and focus on brain region only.
        
        Args:
            image_array: numpy array (H, W, 3) RGB image or (H, W) grayscale image
            
        Returns:
            Cropped image array (same format as input)
        """
        # Handle grayscale vs RGB images
        if len(image_array.shape) == 2:
            # Already grayscale
            gray = image_array.copy()
            is_grayscale = True
        elif len(image_array.shape) == 3:
            # RGB image - convert to grayscale
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            is_grayscale = False
        else:
            # Unexpected format, return original
            return image_array
        
        # Blur slightly to remove noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold the image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove any small regions of noise
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return image_array
        
        # Find the largest contour (assumed to be the brain)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop the image
        cropped = image_array[y:y+h, x:x+w]
        
        return cropped
    
    @staticmethod
    def apply_clahe(image_array):
        """
        Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE).
        
        Args:
            image_array: numpy array (H, W, 3) RGB image or (H, W) grayscale image
            
        Returns:
            Enhanced image array (RGB format)
        """
        # Handle grayscale vs RGB images
        if len(image_array.shape) == 2:
            # Grayscale image - apply CLAHE directly
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image_array)
            # Convert to RGB for consistency
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        elif len(image_array.shape) == 3:
            # RGB image - apply CLAHE to L channel in LAB space
            lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # Merge channels
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        else:
            # Unexpected format, return original (converted to RGB if needed)
            if len(image_array.shape) == 2:
                enhanced = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            else:
                enhanced = image_array
        
        return enhanced

