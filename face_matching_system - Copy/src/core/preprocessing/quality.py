import cv2
import numpy as np
import logging
from typing import Dict, Optional
import sys
import os
# Add the src directory to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, src_path)

from utils.image.processing import ImageProcessor

class QualityAssessor:
    """
    Image quality assessment module.
    Evaluates various quality metrics for face images.
    """
    
    def __init__(self, quality_threshold: float = 0.7):
        self.quality_threshold = quality_threshold
        self.logger = logging.getLogger(__name__)
        self.image_processor = ImageProcessor()
    
    def assess_image_quality(self, image: np.ndarray) -> Dict:
        """
        Assess comprehensive image quality metrics
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with quality scores and metrics
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # Calculate various quality metrics
            quality_metrics = {
                'sharpness': self._calculate_sharpness(gray),
                'brightness': self._calculate_brightness(gray),
                'contrast': self._calculate_contrast(gray),
                'noise_level': self._calculate_noise_level(gray),
                'blur_score': self._calculate_blur_score(gray),
                'exposure': self._calculate_exposure(gray)
            }
            
            # Calculate overall quality score
            quality_metrics['overall_score'] = self._calculate_overall_score(quality_metrics)
            quality_metrics['is_acceptable'] = quality_metrics['overall_score'] >= self.quality_threshold
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Quality assessment failed: {e}")
            return {"error": str(e)}
    
    def _calculate_sharpness(self, gray_image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        try:
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            sharpness = np.var(laplacian)
            # Normalize to 0-1 scale
            return float(min(sharpness / 1000.0, 1.0))
        except Exception:
            return 0.0
    
    def _calculate_brightness(self, gray_image: np.ndarray) -> float:
        """Calculate normalized brightness score"""
        try:
            mean_brightness = np.mean(gray_image)
            # Optimal brightness is around 127 (middle gray)
            brightness_score = 1.0 - abs(mean_brightness - 127.0) / 127.0
            return float(max(brightness_score, 0.0))
        except Exception:
            return 0.0
    
    def _calculate_contrast(self, gray_image: np.ndarray) -> float:
        """Calculate contrast using standard deviation"""
        try:
            contrast = np.std(gray_image)
            # Normalize to 0-1 scale (good contrast is typically > 50)
            return float(min(contrast / 100.0, 1.0))
        except Exception:
            return 0.0
    
    def _calculate_noise_level(self, gray_image: np.ndarray) -> float:
        """Estimate noise level in the image"""
        try:
            # Apply Gaussian blur and calculate difference
            blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
            noise = np.std(gray_image.astype(np.float32) - blurred.astype(np.float32))
            # Lower noise is better, so invert the score
            noise_score = max(1.0 - noise / 50.0, 0.0)
            return noise_score
        except Exception:
            return 0.5
    
    def _calculate_blur_score(self, gray_image: np.ndarray) -> float:
        """Calculate blur score using gradient magnitude"""
        try:
            # Calculate gradients
            grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            blur_score = np.mean(gradient_magnitude)
            
            # Normalize to 0-1 scale
            return float(min(blur_score / 100.0, 1.0))
        except Exception:
            return 0.0
    
    def _calculate_exposure(self, gray_image: np.ndarray) -> float:
        """Calculate exposure quality"""
        try:
            hist, _ = np.histogram(gray_image, bins=256, range=(0, 256))
            
            # Check for clipping (overexposure/underexposure)
            total_pixels = gray_image.size
            underexposed = hist[0] / total_pixels
            overexposed = hist[255] / total_pixels
            
            # Good exposure has minimal clipping
            clipping_penalty = underexposed + overexposed
            exposure_score = max(1.0 - clipping_penalty * 10, 0.0)
            
            return exposure_score
        except Exception:
            return 0.5
    
    def _calculate_overall_score(self, metrics: Dict) -> float:
        """Calculate weighted overall quality score"""
        try:
            weights = {
                'sharpness': 0.25,
                'brightness': 0.15,
                'contrast': 0.20,
                'noise_level': 0.15,
                'blur_score': 0.15,
                'exposure': 0.10
            }
            
            overall_score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in metrics and isinstance(metrics[metric], (int, float)):
                    overall_score += metrics[metric] * weight
                    total_weight += weight
            
            return overall_score / total_weight if total_weight > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def is_quality_acceptable(self, image: np.ndarray) -> bool:
        """Quick quality check"""
        try:
            quality_metrics = self.assess_image_quality(image)
            return quality_metrics.get('is_acceptable', False)
        except Exception:
            return False
