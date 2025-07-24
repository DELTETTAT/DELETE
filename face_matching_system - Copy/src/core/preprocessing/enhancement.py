import cv2
import numpy as np
import logging
from PIL import Image, ImageEnhance
from typing import Optional

class ImageEnhancer:
    """
    Image enhancement module for brightness, contrast, and sharpness adjustments.
    Provides adaptive enhancement based on image characteristics.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality (brightness, contrast, sharpness)
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Enhanced image as numpy array
        """
        try:
            pil_image = Image.fromarray(image)
            
            # Calculate image statistics for adaptive enhancement
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # Adaptive brightness enhancement
            if mean_brightness < 100:  # Dark image
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(1.2)
            elif mean_brightness > 180:  # Bright image
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(0.9)
            
            # Adaptive contrast enhancement
            if std_brightness < 30:  # Low contrast
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.3)
            
            # Slight sharpness enhancement
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(1.1)
            
            # Histogram equalization for better exposure
            enhanced_image = np.array(pil_image)
            lab = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
            enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return enhanced_image
            
        except Exception as e:
            self.logger.warning(f"⚠️ Image enhancement failed: {e}")
            return image
    
    def calculate_brightness_stats(self, image: np.ndarray) -> dict:
        """Calculate brightness statistics for the image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return {
                'mean_brightness': float(np.mean(gray)),
                'std_brightness': float(np.std(gray)),
                'min_brightness': float(np.min(gray)),
                'max_brightness': float(np.max(gray))
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate brightness stats: {e}")
            return {}
