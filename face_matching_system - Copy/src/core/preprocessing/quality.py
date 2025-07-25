
import cv2
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List, Any
import sys
import os
import math

# Add the src directory to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, src_path)

from utils.image.processing import ImageProcessor

class AdvancedQualityAssessor:
    """
    Advanced image quality assessment with GPU acceleration and comprehensive metrics.
    Provides detailed quality analysis for optimal preprocessing decisions.
    """
    
    def __init__(self, quality_threshold: float = 0.7, use_gpu: bool = False):
        self.quality_threshold = quality_threshold
        self.use_gpu = use_gpu
        self.logger = logging.getLogger(__name__)
        self.image_processor = ImageProcessor()
        
        # Initialize GPU resources if available
        if self.use_gpu:
            self._init_gpu_resources()
        
        # Quality assessment parameters
        self.quality_params = {
            'sharpness_threshold': 100,
            'brightness_optimal_range': (80, 180),
            'contrast_minimum': 30,
            'noise_threshold': 20,
            'blur_threshold': 50,
            'face_size_minimum': 64
        }
        
        # Weights for overall quality score
        self.quality_weights = {
            'sharpness': 0.25,
            'brightness': 0.15,
            'contrast': 0.20,
            'noise_level': 0.15,
            'blur_score': 0.15,
            'exposure': 0.10
        }
    
    def _init_gpu_resources(self):
        """Initialize GPU resources for quality assessment"""
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.gpu_available = True
                self.logger.info("🚀 GPU resources initialized for quality assessment")
            else:
                self.gpu_available = False
                self.use_gpu = False
                self.logger.warning("⚠️ GPU requested but not available for quality assessment")
        except:
            self.gpu_available = False
            self.use_gpu = False
    
    def assess_image_quality(self, image: np.ndarray) -> Dict:
        """
        Comprehensive image quality assessment with advanced metrics
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with detailed quality scores and recommendations
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # Core quality metrics
            quality_metrics = {
                'sharpness': self._calculate_advanced_sharpness(gray),
                'brightness': self._calculate_brightness_quality(gray),
                'contrast': self._calculate_contrast_quality(gray),
                'noise_level': self._calculate_noise_quality(gray),
                'blur_score': self._calculate_blur_quality(gray),
                'exposure': self._calculate_exposure_quality(gray)
            }
            
            # Advanced metrics
            advanced_metrics = {
                'texture_quality': self._calculate_texture_quality(gray),
                'edge_density': self._calculate_edge_density(gray),
                'color_richness': self._calculate_color_richness(image) if len(image.shape) == 3 else 0.5,
                'illumination_uniformity': self._calculate_illumination_uniformity(gray),
                'face_quality': self._assess_face_quality(image) if len(image.shape) == 3 else None
            }
            
            # Combine all metrics
            all_metrics = {**quality_metrics, **advanced_metrics}
            
            # Calculate overall quality score
            overall_score = self._calculate_weighted_quality_score(quality_metrics)
            all_metrics['overall_score'] = overall_score
            all_metrics['is_acceptable'] = overall_score >= self.quality_threshold
            
            # Add quality recommendations
            all_metrics['recommendations'] = self._generate_quality_recommendations(all_metrics)
            
            # Add processing suggestions
            all_metrics['processing_suggestions'] = self._generate_processing_suggestions(all_metrics)
            
            return all_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Advanced quality assessment failed: {e}")
            return {"error": str(e)}
    
    def _calculate_advanced_sharpness(self, gray_image: np.ndarray) -> float:
        """Calculate advanced sharpness using multiple methods"""
        try:
            # Method 1: Laplacian variance
            laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
            
            # Method 2: Sobel gradient magnitude
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_mean = np.mean(sobel_magnitude)
            
            # Method 3: Tenengrad variance
            tenengrad = np.sum(sobel_x**2 + sobel_y**2)
            tenengrad_normalized = tenengrad / gray_image.size
            
            # Combine methods with weights
            combined_sharpness = (
                0.4 * min(laplacian_var / 1000.0, 1.0) +
                0.4 * min(sobel_mean / 100.0, 1.0) +
                0.2 * min(tenengrad_normalized / 10000.0, 1.0)
            )
            
            return float(combined_sharpness)
            
        except Exception:
            return 0.0
    
    def _calculate_brightness_quality(self, gray_image: np.ndarray) -> float:
        """Calculate brightness quality with optimal range consideration"""
        try:
            mean_brightness = np.mean(gray_image)
            std_brightness = np.std(gray_image)
            
            # Optimal brightness score (peak at 127)
            brightness_distance = abs(mean_brightness - 127.0)
            brightness_score = max(1.0 - brightness_distance / 127.0, 0.0)
            
            # Brightness distribution score
            distribution_score = min(std_brightness / 60.0, 1.0)
            
            # Combined brightness quality
            combined_brightness = 0.7 * brightness_score + 0.3 * distribution_score
            
            return float(combined_brightness)
            
        except Exception:
            return 0.0
    
    def _calculate_contrast_quality(self, gray_image: np.ndarray) -> float:
        """Calculate contrast quality with dynamic range analysis"""
        try:
            # Standard deviation contrast
            std_contrast = np.std(gray_image)
            
            # RMS contrast
            mean_val = np.mean(gray_image)
            rms_contrast = np.sqrt(np.mean((gray_image - mean_val)**2))
            
            # Michelson contrast
            min_val = np.min(gray_image)
            max_val = np.max(gray_image)
            if max_val + min_val > 0:
                michelson_contrast = (max_val - min_val) / (max_val + min_val)
            else:
                michelson_contrast = 0
            
            # Combine contrast measures
            combined_contrast = (
                0.4 * min(std_contrast / 80.0, 1.0) +
                0.4 * min(rms_contrast / 80.0, 1.0) +
                0.2 * michelson_contrast
            )
            
            return float(combined_contrast)
            
        except Exception:
            return 0.0
    
    def _calculate_noise_quality(self, gray_image: np.ndarray) -> float:
        """Calculate noise quality using multiple noise estimation methods"""
        try:
            # Method 1: High-frequency noise estimation
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            noise_image = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
            high_freq_noise = np.std(noise_image)
            
            # Method 2: Gaussian blur difference
            blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
            blur_diff_noise = np.std(gray_image.astype(np.float32) - blurred.astype(np.float32))
            
            # Method 3: Median filter difference
            median_filtered = cv2.medianBlur(gray_image, 5)
            median_diff_noise = np.std(gray_image.astype(np.float32) - median_filtered.astype(np.float32))
            
            # Combine noise estimates
            combined_noise = np.mean([high_freq_noise, blur_diff_noise, median_diff_noise])
            
            # Convert to quality score (lower noise = higher quality)
            noise_quality = max(1.0 - combined_noise / 50.0, 0.0)
            
            return float(noise_quality)
            
        except Exception:
            return 0.5
    
    def _calculate_blur_quality(self, gray_image: np.ndarray) -> float:
        """Calculate blur quality using multiple blur detection methods"""
        try:
            # Method 1: Laplacian blur detection
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            laplacian_blur = np.var(laplacian)
            
            # Method 2: FFT-based blur detection
            fft = np.fft.fft2(gray_image)
            fft_shift = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shift)
            
            # Calculate high-frequency content
            h, w = gray_image.shape
            center_h, center_w = h // 2, w // 2
            high_freq_region = magnitude_spectrum[
                center_h - h//4:center_h + h//4,
                center_w - w//4:center_w + w//4
            ]
            high_freq_content = np.mean(high_freq_region)
            
            # Method 3: Sobel variance
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_variance = np.var(np.sqrt(sobel_x**2 + sobel_y**2))
            
            # Combine blur measures
            blur_score = (
                0.4 * min(laplacian_blur / 200.0, 1.0) +
                0.3 * min(high_freq_content / 1000.0, 1.0) +
                0.3 * min(sobel_variance / 500.0, 1.0)
            )
            
            return float(blur_score)
            
        except Exception:
            return 0.0
    
    def _calculate_exposure_quality(self, gray_image: np.ndarray) -> float:
        """Calculate exposure quality with histogram analysis"""
        try:
            hist, bins = np.histogram(gray_image, bins=256, range=(0, 256))
            total_pixels = gray_image.size
            
            # Check for clipping
            underexposed = hist[0:10].sum() / total_pixels
            overexposed = hist[246:256].sum() / total_pixels
            
            # Check for good tonal distribution
            shadows = hist[0:85].sum() / total_pixels
            midtones = hist[85:170].sum() / total_pixels
            highlights = hist[170:256].sum() / total_pixels
            
            # Ideal distribution: some shadows, lots of midtones, some highlights
            distribution_score = 1.0 - abs(midtones - 0.5) - abs(shadows - 0.25) - abs(highlights - 0.25)
            distribution_score = max(distribution_score, 0.0)
            
            # Penalize severe clipping
            clipping_penalty = (underexposed + overexposed) * 5
            exposure_score = max(distribution_score - clipping_penalty, 0.0)
            
            return float(exposure_score)
            
        except Exception:
            return 0.5
    
    def _calculate_texture_quality(self, gray_image: np.ndarray) -> float:
        """Calculate texture quality using local binary patterns"""
        try:
            # Simple texture analysis using standard deviation in local windows
            kernel = np.ones((8, 8), np.float32) / 64
            local_mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((gray_image.astype(np.float32) - local_mean)**2, -1, kernel)
            
            texture_score = np.mean(np.sqrt(local_variance))
            normalized_texture = min(texture_score / 30.0, 1.0)
            
            return float(normalized_texture)
            
        except Exception:
            return 0.5
    
    def _calculate_edge_density(self, gray_image: np.ndarray) -> float:
        """Calculate edge density for structural quality assessment"""
        try:
            # Canny edge detection
            edges = cv2.Canny(gray_image, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Normalize edge density
            normalized_density = min(edge_density * 10, 1.0)
            
            return float(normalized_density)
            
        except Exception:
            return 0.5
    
    def _calculate_color_richness(self, color_image: np.ndarray) -> float:
        """Calculate color richness and saturation"""
        try:
            # Convert to HSV for saturation analysis
            hsv = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            
            # Calculate mean saturation
            mean_saturation = np.mean(saturation) / 255.0
            
            # Calculate color diversity using histogram
            hist_r = cv2.calcHist([color_image], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([color_image], [1], None, [32], [0, 256])
            hist_b = cv2.calcHist([color_image], [2], None, [32], [0, 256])
            
            # Calculate histogram entropy as diversity measure
            def entropy(hist):
                hist_norm = hist / (hist.sum() + 1e-7)
                entropy_val = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
                return entropy_val / np.log2(32)  # Normalize
            
            color_diversity = (entropy(hist_r) + entropy(hist_g) + entropy(hist_b)) / 3
            
            # Combine saturation and diversity
            color_richness = 0.6 * mean_saturation + 0.4 * color_diversity
            
            return float(color_richness)
            
        except Exception:
            return 0.5
    
    def _calculate_illumination_uniformity(self, gray_image: np.ndarray) -> float:
        """Calculate illumination uniformity"""
        try:
            # Divide image into blocks and analyze brightness variation
            h, w = gray_image.shape
            block_size = min(h, w) // 8
            
            if block_size < 8:
                return 0.5
            
            block_means = []
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray_image[i:i+block_size, j:j+block_size]
                    block_means.append(np.mean(block))
            
            if not block_means:
                return 0.5
            
            # Calculate coefficient of variation
            mean_brightness = np.mean(block_means)
            std_brightness = np.std(block_means)
            
            if mean_brightness > 0:
                cv = std_brightness / mean_brightness
                uniformity = max(1.0 - cv, 0.0)
            else:
                uniformity = 0.0
            
            return float(uniformity)
            
        except Exception:
            return 0.5
    
    def _assess_face_quality(self, image: np.ndarray) -> Optional[Dict]:
        """Assess face-specific quality metrics"""
        try:
            # This would integrate with face detection for face-specific quality
            # For now, return basic face quality assessment
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Simple face quality based on central region
            h, w = gray.shape
            center_region = gray[h//4:3*h//4, w//4:3*w//4]
            
            face_quality = {
                'central_sharpness': float(cv2.Laplacian(center_region, cv2.CV_64F).var() / 1000.0),
                'central_contrast': float(np.std(center_region) / 80.0),
                'size_adequacy': min(min(h, w) / 160.0, 1.0)
            }
            
            face_quality['overall_face_quality'] = np.mean(list(face_quality.values()))
            
            return face_quality
            
        except Exception:
            return None
    
    def _calculate_weighted_quality_score(self, metrics: Dict) -> float:
        """Calculate weighted overall quality score"""
        try:
            overall_score = 0.0
            total_weight = 0.0
            
            for metric, weight in self.quality_weights.items():
                if metric in metrics and isinstance(metrics[metric], (int, float)):
                    overall_score += metrics[metric] * weight
                    total_weight += weight
            
            return overall_score / total_weight if total_weight > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _generate_quality_recommendations(self, metrics: Dict) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        try:
            if metrics.get('sharpness', 0) < 0.5:
                recommendations.append("Apply sharpening filter to improve image sharpness")
            
            if metrics.get('brightness', 0) < 0.5:
                brightness_val = metrics.get('brightness', 0.5)
                if brightness_val < 0.3:
                    recommendations.append("Increase brightness - image appears too dark")
                else:
                    recommendations.append("Adjust brightness for optimal exposure")
            
            if metrics.get('contrast', 0) < 0.4:
                recommendations.append("Enhance contrast to improve image definition")
            
            if metrics.get('noise_level', 1.0) < 0.6:
                recommendations.append("Apply noise reduction filter")
            
            if metrics.get('blur_score', 0) < 0.4:
                recommendations.append("Image appears blurry - consider deblurring techniques")
            
            if metrics.get('color_richness', 0.5) < 0.4:
                recommendations.append("Enhance color saturation for better color quality")
            
            if not recommendations:
                recommendations.append("Image quality is acceptable - minimal enhancement needed")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate recommendations: {e}")
            recommendations = ["Unable to generate specific recommendations"]
        
        return recommendations
    
    def _generate_processing_suggestions(self, metrics: Dict) -> Dict[str, Any]:
        """Generate processing pipeline suggestions"""
        suggestions = {
            'apply_enhancement': metrics.get('overall_score', 0) < 0.7,
            'noise_reduction_strength': max(1.0 - metrics.get('noise_level', 1.0), 0.0),
            'sharpening_strength': max(0.6 - metrics.get('sharpness', 0.6), 0.0),
            'contrast_boost': max(0.5 - metrics.get('contrast', 0.5), 0.0),
            'brightness_adjustment': 0.5 - abs(metrics.get('brightness', 0.5) - 0.5),
            'skip_processing': metrics.get('overall_score', 0) > 0.9
        }
        
        return suggestions
    
    def is_quality_acceptable(self, image: np.ndarray) -> bool:
        """Quick quality check"""
        try:
            quality_metrics = self.assess_image_quality(image)
            return quality_metrics.get('is_acceptable', False)
        except Exception:
            return False
    
    def get_quality_capabilities(self) -> Dict[str, Any]:
        """Get current quality assessment capabilities"""
        return {
            'gpu_acceleration': self.use_gpu and getattr(self, 'gpu_available', False),
            'advanced_metrics': [
                'sharpness', 'brightness', 'contrast', 'noise_level',
                'blur_score', 'exposure', 'texture_quality', 'edge_density',
                'color_richness', 'illumination_uniformity', 'face_quality'
            ],
            'quality_threshold': self.quality_threshold,
            'recommendations_enabled': True,
            'processing_suggestions_enabled': True
        }
