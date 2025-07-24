
import cv2
import numpy as np
import logging
from PIL import Image, ImageEnhance, ImageFilter
from typing import Optional, Dict, Any, Tuple, List
import sys

class AdvancedImageEnhancer:
    """
    Advanced image enhancement with GPU acceleration and automatic optimization.
    Uses multiple enhancement techniques based on image characteristics.
    """
    
    def __init__(self, use_gpu: bool = False, processing_config: Dict = None):
        self.use_gpu = use_gpu
        self.processing_config = processing_config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize GPU resources if available
        if self.use_gpu:
            self._init_gpu_resources()
        
        # Enhancement parameters
        self.enhancement_params = {
            'clahe_clip_limit': 2.0,
            'clahe_tile_size': (8, 8),
            'bilateral_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75,
            'unsharp_radius': 2,
            'unsharp_percent': 150,
            'unsharp_threshold': 3
        }
    
    def _init_gpu_resources(self):
        """Initialize GPU resources for OpenCV operations"""
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.gpu_available = True
                self.logger.info("🚀 GPU resources initialized for image enhancement")
            else:
                self.gpu_available = False
                self.use_gpu = False
                self.logger.warning("⚠️ GPU requested but not available, falling back to CPU")
        except:
            self.gpu_available = False
            self.use_gpu = False
            self.logger.warning("⚠️ OpenCV GPU support not available, using CPU")
    
    def auto_enhance_image(self, image: np.ndarray, quality_metrics: Dict = None) -> np.ndarray:
        """
        Automatically enhance image based on quality metrics and characteristics
        
        Args:
            image: Input image as numpy array
            quality_metrics: Quality assessment results to guide enhancement
            
        Returns:
            Enhanced image as numpy array
        """
        try:
            if quality_metrics is None:
                quality_metrics = self._quick_quality_assessment(image)
            
            # Create enhancement pipeline based on image characteristics
            enhancement_pipeline = self._create_enhancement_pipeline(image, quality_metrics)
            
            enhanced_image = image.copy()
            
            for enhancement_step in enhancement_pipeline:
                enhanced_image = self._apply_enhancement_step(enhanced_image, enhancement_step)
            
            return enhanced_image
            
        except Exception as e:
            self.logger.warning(f"⚠️ Auto enhancement failed: {e}")
            return self.enhance_image_basic(image)
    
    def _quick_quality_assessment(self, image: np.ndarray) -> Dict:
        """Quick quality assessment for enhancement guidance"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        return {
            'brightness': float(np.mean(gray)),
            'contrast': float(np.std(gray)),
            'sharpness': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            'noise_level': self._estimate_noise_level(gray)
        }
    
    def _estimate_noise_level(self, gray_image: np.ndarray) -> float:
        """Estimate noise level in grayscale image"""
        try:
            # Use difference between original and gaussian blur
            blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
            noise = np.std(gray_image.astype(np.float32) - blurred.astype(np.float32))
            return float(noise)
        except Exception:
            return 10.0  # Default moderate noise level
    
    def _create_enhancement_pipeline(self, image: np.ndarray, quality_metrics: Dict) -> List[Dict]:
        """Create optimal enhancement pipeline based on image characteristics"""
        pipeline = []
        
        brightness = quality_metrics.get('brightness', 127)
        contrast = quality_metrics.get('contrast', 50)
        sharpness = quality_metrics.get('sharpness', 100)
        noise_level = quality_metrics.get('noise_level', 10)
        
        # Noise reduction (if needed)
        if noise_level > 15:
            if self.use_gpu and self.gpu_available:
                pipeline.append({'type': 'gpu_bilateral_filter', 'strength': min(noise_level / 20, 1.0)})
            else:
                pipeline.append({'type': 'bilateral_filter', 'strength': min(noise_level / 20, 1.0)})
        
        # Brightness correction
        if brightness < 80:  # Dark image
            pipeline.append({'type': 'brightness_correction', 'factor': 1.0 + (80 - brightness) / 160})
        elif brightness > 180:  # Bright image
            pipeline.append({'type': 'brightness_correction', 'factor': 1.0 - (brightness - 180) / 160})
        
        # Contrast enhancement
        if contrast < 40:  # Low contrast
            if self.use_gpu and self.gpu_available:
                pipeline.append({'type': 'gpu_clahe', 'clip_limit': 3.0})
            else:
                pipeline.append({'type': 'clahe', 'clip_limit': 3.0})
            pipeline.append({'type': 'contrast_enhancement', 'factor': 1.0 + (40 - contrast) / 80})
        
        # Sharpness enhancement
        if sharpness < 50:  # Blurry image
            pipeline.append({'type': 'unsharp_mask', 'strength': min((50 - sharpness) / 25, 2.0)})
        
        # Color enhancement
        pipeline.append({'type': 'color_enhancement', 'saturation_factor': 1.1})
        
        # Final histogram equalization
        pipeline.append({'type': 'histogram_equalization'})
        
        return pipeline
    
    def _apply_enhancement_step(self, image: np.ndarray, step: Dict) -> np.ndarray:
        """Apply individual enhancement step"""
        try:
            step_type = step['type']
            
            if step_type == 'bilateral_filter':
                return self._apply_bilateral_filter(image, step.get('strength', 1.0))
            elif step_type == 'gpu_bilateral_filter' and self.use_gpu:
                return self._apply_gpu_bilateral_filter(image, step.get('strength', 1.0))
            elif step_type == 'brightness_correction':
                return self._apply_brightness_correction(image, step.get('factor', 1.0))
            elif step_type == 'clahe':
                return self._apply_clahe(image, step.get('clip_limit', 2.0))
            elif step_type == 'gpu_clahe' and self.use_gpu:
                return self._apply_gpu_clahe(image, step.get('clip_limit', 2.0))
            elif step_type == 'contrast_enhancement':
                return self._apply_contrast_enhancement(image, step.get('factor', 1.0))
            elif step_type == 'unsharp_mask':
                return self._apply_unsharp_mask(image, step.get('strength', 1.0))
            elif step_type == 'color_enhancement':
                return self._apply_color_enhancement(image, step.get('saturation_factor', 1.1))
            elif step_type == 'histogram_equalization':
                return self._apply_histogram_equalization(image)
            else:
                return image
                
        except Exception as e:
            self.logger.warning(f"⚠️ Enhancement step {step_type} failed: {e}")
            return image
    
    def _apply_bilateral_filter(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply CPU bilateral filter for noise reduction"""
        d = int(self.enhancement_params['bilateral_d'] * strength)
        sigma_color = self.enhancement_params['bilateral_sigma_color'] * strength
        sigma_space = self.enhancement_params['bilateral_sigma_space'] * strength
        
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def _apply_gpu_bilateral_filter(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply GPU bilateral filter for noise reduction"""
        try:
            # Upload to GPU
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(image)
            
            d = int(self.enhancement_params['bilateral_d'] * strength)
            sigma_color = self.enhancement_params['bilateral_sigma_color'] * strength
            sigma_space = self.enhancement_params['bilateral_sigma_space'] * strength
            
            # Apply bilateral filter on GPU
            gpu_result = cv2.cuda.bilateralFilter(gpu_image, d, sigma_color, sigma_space)
            
            # Download from GPU
            result = gpu_result.download()
            return result
            
        except Exception as e:
            self.logger.warning(f"GPU bilateral filter failed, using CPU: {e}")
            return self._apply_bilateral_filter(image, strength)
    
    def _apply_brightness_correction(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Apply brightness correction"""
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(factor)
        return np.array(enhanced)
    
    def _apply_clahe(self, image: np.ndarray, clip_limit: float) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=self.enhancement_params['clahe_tile_size'])
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def _apply_gpu_clahe(self, image: np.ndarray, clip_limit: float) -> np.ndarray:
        """Apply GPU-accelerated CLAHE"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Upload L channel to GPU
            gpu_l = cv2.cuda_GpuMat()
            gpu_l.upload(lab[:, :, 0])
            
            # Create CLAHE object for GPU
            clahe = cv2.cuda.createCLAHE(clipLimit=clip_limit, tileGridSize=self.enhancement_params['clahe_tile_size'])
            gpu_result = clahe.apply(gpu_l)
            
            # Download result
            lab[:, :, 0] = gpu_result.download()
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
        except Exception as e:
            self.logger.warning(f"GPU CLAHE failed, using CPU: {e}")
            return self._apply_clahe(image, clip_limit)
    
    def _apply_contrast_enhancement(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Apply contrast enhancement"""
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(factor)
        return np.array(enhanced)
    
    def _apply_unsharp_mask(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply unsharp masking for sharpness enhancement"""
        try:
            pil_image = Image.fromarray(image)
            
            # Create unsharp mask filter
            radius = self.enhancement_params['unsharp_radius'] * strength
            percent = int(self.enhancement_params['unsharp_percent'] * strength)
            threshold = self.enhancement_params['unsharp_threshold']
            
            # Apply unsharp mask
            enhanced = pil_image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
            return np.array(enhanced)
            
        except Exception as e:
            self.logger.warning(f"Unsharp mask failed: {e}")
            return image
    
    def _apply_color_enhancement(self, image: np.ndarray, saturation_factor: float) -> np.ndarray:
        """Apply color saturation enhancement"""
        try:
            pil_image = Image.fromarray(image)
            enhancer = ImageEnhance.Color(pil_image)
            enhanced = enhancer.enhance(saturation_factor)
            return np.array(enhanced)
        except Exception as e:
            self.logger.warning(f"Color enhancement failed: {e}")
            return image
    
    def _apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive histogram equalization"""
        try:
            # Convert to YUV color space
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            
            # Apply histogram equalization to Y channel
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            
            # Convert back to RGB
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            
        except Exception as e:
            self.logger.warning(f"Histogram equalization failed: {e}")
            return image
    
    def enhance_image_basic(self, image: np.ndarray) -> np.ndarray:
        """
        Basic enhancement for fallback scenarios
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
            self.logger.warning(f"⚠️ Basic image enhancement failed: {e}")
            return image
    
    def get_enhancement_capabilities(self) -> Dict[str, Any]:
        """Get current enhancement capabilities"""
        return {
            'gpu_acceleration': self.use_gpu and self.gpu_available,
            'available_enhancements': [
                'bilateral_filter',
                'brightness_correction',
                'clahe',
                'contrast_enhancement',
                'unsharp_mask',
                'color_enhancement',
                'histogram_equalization'
            ],
            'gpu_enhancements': [
                'gpu_bilateral_filter',
                'gpu_clahe'
            ] if self.use_gpu and self.gpu_available else [],
            'processing_config': self.processing_config
        }
