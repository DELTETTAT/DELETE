# Added ImagePreprocessor alias for backward compatibility

import os
import numpy as np
import logging
import cv2
from PIL import Image, ExifTags
from typing import Tuple, Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import psutil
import platform
import multiprocessing as mp

# Add the src directory to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, src_path)

from utils.image.processing import ImageProcessor
from .enhancement import AdvancedImageEnhancer
from .alignment import AdvancedFaceAligner
from .quality import AdvancedQualityAssessor

class AdvancedImagePreprocessor:
    """
    Advanced automatic image preprocessing system.
    Automatically detects hardware capabilities and optimizes processing.
    """

    def __init__(self, 
                 target_size: Tuple[int, int] = (160, 160),
                 enable_face_alignment: bool = True,
                 enable_enhancement: bool = True,
                 quality_threshold: float = 0.7,
                 auto_optimize: bool = True):

        self.target_size = target_size
        self.enable_face_alignment = enable_face_alignment
        self.enable_enhancement = enable_enhancement
        self.quality_threshold = quality_threshold
        self.auto_optimize = auto_optimize

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Hardware detection and optimization
        self.hardware_info = self._detect_hardware_capabilities()
        self.processing_config = self._optimize_processing_config()

        # Initialize components with hardware-optimized settings
        self.image_processor = ImageProcessor()
        self.enhancer = AdvancedImageEnhancer(
            use_gpu=self.hardware_info['gpu_available'],
            processing_config=self.processing_config
        ) if enable_enhancement else None

        self.aligner = AdvancedFaceAligner(
            target_size,
            use_gpu=self.hardware_info['gpu_available'],
            processing_config=self.processing_config
        ) if enable_face_alignment else None

        self.quality_assessor = AdvancedQualityAssessor(
            quality_threshold,
            use_gpu=self.hardware_info['gpu_available']
        )

        self._log_system_info()

    def _detect_hardware_capabilities(self) -> Dict[str, Any]:
        """Detect available hardware capabilities"""
        hardware_info = {
            'cpu_count': mp.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_available': False,
            'gpu_memory_gb': 0,
            'gpu_name': None,
            'opencv_gpu': False,
            'cuda_available': False
        }

        try:
            # Check for CUDA availability
            import torch
            if torch.cuda.is_available():
                hardware_info['cuda_available'] = True
                hardware_info['gpu_available'] = True
                hardware_info['gpu_name'] = torch.cuda.get_device_name(0)
                hardware_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.logger.info(f"🚀 CUDA GPU detected: {hardware_info['gpu_name']} ({hardware_info['gpu_memory_gb']:.1f}GB)")
        except ImportError:
            self.logger.info("PyTorch not available, checking OpenCV GPU support")

        # Check OpenCV GPU support
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                hardware_info['opencv_gpu'] = True
                if not hardware_info['gpu_available']:
                    hardware_info['gpu_available'] = True
                    self.logger.info("🚀 OpenCV CUDA support detected")
        except:
            pass

        if not hardware_info['gpu_available']:
            self.logger.info("💻 Using CPU-only processing")

        return hardware_info

    def _optimize_processing_config(self) -> Dict[str, Any]:
        """Optimize processing configuration based on hardware"""
        config = {
            'batch_size': 32,
            'num_workers': min(self.hardware_info['cpu_count'], 8),
            'use_threading': True,
            'memory_optimization': True,
            'precision': 'float32'
        }

        # GPU optimizations
        if self.hardware_info['gpu_available']:
            if self.hardware_info['gpu_memory_gb'] > 6:
                config['batch_size'] = 64
                config['precision'] = 'float32'
            elif self.hardware_info['gpu_memory_gb'] > 4:
                config['batch_size'] = 48
                config['precision'] = 'float32'
            else:
                config['batch_size'] = 24
                config['precision'] = 'float16'

        # CPU optimizations
        else:
            if self.hardware_info['memory_gb'] > 16:
                config['batch_size'] = 24
                config['num_workers'] = min(self.hardware_info['cpu_count'], 6)
            elif self.hardware_info['memory_gb'] > 8:
                config['batch_size'] = 16
                config['num_workers'] = min(self.hardware_info['cpu_count'], 4)
            else:
                config['batch_size'] = 8
                config['num_workers'] = min(self.hardware_info['cpu_count'], 2)
                config['memory_optimization'] = True

        return config

    def _log_system_info(self):
        """Log system information and optimization settings"""
        self.logger.info("🔧 Advanced Preprocessing System Initialized")
        self.logger.info(f"   CPU Cores: {self.hardware_info['cpu_count']}")
        self.logger.info(f"   Memory: {self.hardware_info['memory_gb']:.1f}GB")
        self.logger.info(f"   GPU Available: {self.hardware_info['gpu_available']}")
        if self.hardware_info['gpu_available']:
            self.logger.info(f"   GPU: {self.hardware_info.get('gpu_name', 'Unknown')}")
        self.logger.info(f"   Batch Size: {self.processing_config['batch_size']}")
        self.logger.info(f"   Workers: {self.processing_config['num_workers']}")

    def preprocess_image(self, input_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Conservative preprocessing pipeline focused on face alignment and minimal enhancement

        Args:
            input_path: Path to input image
            output_path: Path to save processed image

        Returns:
            Dictionary with success status and detailed metrics
        """
        try:
            # Validate output path
            if output_path is None:
                filename = os.path.basename(input_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(os.path.dirname(input_path), f"processed_{name}{ext}")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Load and validate image
            image = self.image_processor.load_image(input_path)
            if image is None:
                return {'success': False, 'error': 'Failed to load image'}

            # Store original dimensions
            original_shape = image.shape

            # Correct orientation based on EXIF (important for proper face detection)
            image = self._correct_orientation(image, input_path)

            # Conservative face alignment - only align, don't modify features
            if self.enable_face_alignment and self.aligner:
                aligned_image = self.aligner.align_face_conservative(image)
                if aligned_image is not None:
                    image = aligned_image
                else:
                    self.logger.warning(f"⚠️ Face alignment failed for {os.path.basename(input_path)}")

            # Resize to target size with proper interpolation
            image = self._conservative_resize(image, self.target_size)

            # Conservative lighting normalization only
            image = self._conservative_lighting_correction(image)

            # Minimal noise reduction without losing detail
            image = self._minimal_noise_reduction(image)

            # Save processed image
            save_success = self.image_processor.save_image(image, output_path)

            return {
                'success': save_success,
                'original_shape': original_shape,
                'final_shape': image.shape,
                'processing_applied': ['orientation_correction', 'face_alignment', 'lighting_correction', 'minimal_noise_reduction'],
                'processing_config': self.processing_config
            }

        except Exception as e:
            self.logger.error(f"❌ Conservative preprocessing failed for {input_path}: {e}")
            return {'success': False, 'error': str(e)}

    def _correct_orientation(self, image: np.ndarray, image_path: str) -> np.ndarray:
        """Advanced orientation correction with multiple methods"""
        try:
            # Method 1: EXIF-based correction
            pil_image = Image.open(image_path)

            if hasattr(pil_image, '_getexif'):
                exif = pil_image._getexif()
                if exif is not None:
                    for tag, value in exif.items():
                        if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == 'Orientation':
                            if value == 3:
                                pil_image = pil_image.rotate(180, expand=True)
                            elif value == 6:
                                pil_image = pil_image.rotate(270, expand=True)
                            elif value == 8:
                                pil_image = pil_image.rotate(90, expand=True)
                            break

            corrected_image = np.array(pil_image.convert('RGB'))

            # Method 2: Automatic face-based orientation detection
            if self.aligner and hasattr(self.aligner, 'detect_orientation'):
                orientation_angle = self.aligner.detect_orientation(corrected_image)
                if orientation_angle != 0:
                    center = (corrected_image.shape[1] // 2, corrected_image.shape[0] // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, orientation_angle, 1.0)
                    corrected_image = cv2.warpAffine(corrected_image, rotation_matrix, 
                                                   (corrected_image.shape[1], corrected_image.shape[0]))

            return corrected_image

        except Exception as e:
            self.logger.warning(f"⚠️ Advanced orientation correction failed: {e}")
            return image

    def _intelligent_resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Intelligent resizing with content-aware scaling"""
        try:
            h, w = image.shape[:2]
            target_w, target_h = target_size

            # Calculate aspect ratios
            original_aspect = w / h
            target_aspect = target_w / target_h

            if abs(original_aspect - target_aspect) < 0.1:
                # Aspect ratios are similar, direct resize
                return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)

            # Use smart cropping to maintain important content
            if original_aspect > target_aspect:
                # Image is wider, crop width
                new_w = int(h * target_aspect)
                start_x = (w - new_w) // 2
                cropped = image[:, start_x:start_x + new_w]
            else:
                # Image is taller, crop height
                new_h = int(w / target_aspect)
                start_y = (h - new_h) // 2
                cropped = image[start_y:start_y + new_h, :]

            return cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)

        except Exception as e:
            self.logger.warning(f"⚠️ Intelligent resize failed, using standard resize: {e}")
            return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    def _advanced_normalize(self, image: np.ndarray) -> np.ndarray:
        """Advanced normalization with histogram equalization"""
        try:
            # Convert to float32 for processing
            image_float = image.astype(np.float32) / 255.0

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

            # Blend original and enhanced based on image characteristics
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            contrast = np.std(gray)

            if contrast < 30:  # Low contrast image
                blend_factor = 0.7
            else:
                blend_factor = 0.3

            final_image = cv2.addWeighted(image, 1 - blend_factor, enhanced, blend_factor, 0)

            # Final normalization
            return final_image.astype(np.uint8)

        except Exception as e:
            self.logger.warning(f"⚠️ Advanced normalization failed: {e}")
            return self.image_processor.normalize_image(image)

    def preprocess_batch_optimized(self, image_paths: List[str], 
                                  output_dir: Optional[str] = None,
                                  progress_callback: Optional[callable] = None) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        Hardware-optimized batch preprocessing with dynamic load balancing

        Args:
            image_paths: List of image file paths
            output_dir: Directory to save preprocessed images
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (successful paths, failed paths, processing stats)
        """
        if not image_paths:
            return [], [], {}

        successful_paths = []
        failed_paths = []
        processing_stats = {
            'total_images': len(image_paths),
            'successful': 0,
            'failed': 0,
            'average_quality_improvement': 0,
            'processing_time': 0,
            'hardware_utilized': self.hardware_info
        }

        import time
        start_time = time.time()

        # Dynamic batch processing based on system load
        current_batch_size = self.processing_config['batch_size']
        quality_improvements = []

        # Process in optimized batches
        for i in range(0, len(image_paths), current_batch_size):
            batch_paths = image_paths[i:i + current_batch_size]

            # Monitor system resources and adjust
            memory_usage = psutil.virtual_memory().percent
            if memory_usage > 85:
                current_batch_size = max(4, current_batch_size // 2)
                self.logger.info(f"High memory usage ({memory_usage}%), reducing batch size to {current_batch_size}")
            elif memory_usage < 50 and current_batch_size < self.processing_config['batch_size']:
                current_batch_size = min(self.processing_config['batch_size'], current_batch_size * 2)

            # Process batch
            batch_successful, batch_failed, batch_stats = self._process_batch(
                batch_paths, output_dir, progress_callback
            )

            successful_paths.extend(batch_successful)
            failed_paths.extend(batch_failed)

            if batch_stats.get('quality_improvements'):
                quality_improvements.extend(batch_stats['quality_improvements'])

            # Update progress
            if progress_callback:
                progress = (i + len(batch_paths)) / len(image_paths)
                progress_callback(progress, f"Processed {i + len(batch_paths)}/{len(image_paths)} images")

        # Calculate final stats
        processing_stats['successful'] = len(successful_paths)
        processing_stats['failed'] = len(failed_paths)
        processing_stats['processing_time'] = time.time() - start_time

        if quality_improvements:
            processing_stats['average_quality_improvement'] = np.mean(quality_improvements)

        self.logger.info(f"✅ Batch processing completed: {processing_stats['successful']}/{processing_stats['total_images']} successful")
        self.logger.info(f"   Processing time: {processing_stats['processing_time']:.2f}s")
        if quality_improvements:
            self.logger.info(f"   Average quality improvement: {processing_stats['average_quality_improvement']:.3f}")

        return successful_paths, failed_paths, processing_stats

    def _process_batch(self, batch_paths: List[str], output_dir: Optional[str], 
                      progress_callback: Optional[callable]) -> Tuple[List[str], List[str], Dict]:
        """Process a single batch of images"""
        successful = []
        failed = []
        quality_improvements = []

        def process_single_image(path):
            try:
                if output_dir:
                    filename = os.path.basename(path)
                    output_path = os.path.join(output_dir, filename)
                    os.makedirs(output_dir, exist_ok=True)
                else:
                    output_path = None

                result = self.preprocess_image(path, output_path)

                if result['success']:
                    quality_improvement = result.get('quality_improvement', 0)
                    return 'success', output_path if output_path else path, quality_improvement
                else:
                    return 'failed', path, 0

            except Exception as e:
                self.logger.error(f"Error processing {path}: {e}")
                return 'failed', path, 0

        # Use optimal number of workers
        max_workers = self.processing_config['num_workers']

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(process_single_image, path): path 
                             for path in batch_paths}

            for future in as_completed(future_to_path):
                status, result_path, quality_improvement = future.result()

                if status == 'success':
                    successful.append(result_path)
                    quality_improvements.append(quality_improvement)
                else:
                    failed.append(result_path)

        return successful, failed, {'quality_improvements': quality_improvements}

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get detailed optimization and performance report"""
        return {
            'hardware_info': self.hardware_info,
            'processing_config': self.processing_config,
            'optimization_features': {
                'gpu_acceleration': self.hardware_info['gpu_available'],
                'multi_threading': self.processing_config['use_threading'],
                'dynamic_batching': True,
                'memory_optimization': self.processing_config['memory_optimization'],
                'quality_based_enhancement': True,
                'advanced_alignment': self.enable_face_alignment,
                'intelligent_resizing': True
            },
            'recommended_settings': self._get_recommended_settings()
        }

    def _get_recommended_settings(self) -> Dict[str, Any]:
        """Get recommended settings based on hardware"""
        recommendations = {
            'batch_size': self.processing_config['batch_size'],
            'num_workers': self.processing_config['num_workers'],
            'enable_gpu_acceleration': self.hardware_info['gpu_available'],
            'quality_threshold': self.quality_threshold
        }

        if self.hardware_info['memory_gb'] < 4:
            recommendations['memory_conservation_mode'] = True
            recommendations['suggested_batch_size'] = min(8, self.processing_config['batch_size'])

        return recommendations

    def _conservative_resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Conservative resizing that preserves facial features"""
        try:
            # Use high-quality interpolation that preserves edges and features
            return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
        except Exception as e:
            self.logger.warning(f"⚠️ Conservative resize failed: {e}")
            return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    def _conservative_lighting_correction(self, image: np.ndarray) -> np.ndarray:
        """Conservative lighting correction that preserves natural appearance"""
        try:
            # Convert to LAB color space for better lighting control
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]

            # Check if lighting correction is needed
            mean_brightness = np.mean(l_channel)

            # Only apply gentle correction if severely under/over exposed
            if mean_brightness < 80:  # Very dark
                # Gentle brightening
                correction_factor = min(1.2, 100 / mean_brightness)
                l_channel = np.clip(l_channel * correction_factor, 0, 255)
            elif mean_brightness > 200:  # Very bright
                # Gentle darkening
                correction_factor = max(0.8, 180 / mean_brightness)
                l_channel = np.clip(l_channel * correction_factor, 0, 255)

            # Apply very mild CLAHE only if needed (low contrast)
            contrast = np.std(l_channel)
            if contrast < 25:  # Very low contrast
                clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
                l_channel = clahe.apply(l_channel.astype(np.uint8))

            lab[:, :, 0] = l_channel
            corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

            return corrected.astype(np.uint8)

        except Exception as e:
            self.logger.warning(f"⚠️ Conservative lighting correction failed: {e}")
            return image

    def _minimal_noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """Minimal noise reduction that preserves facial features and edges"""
        try:
            # Check if noise reduction is needed
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Estimate noise level using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Only apply noise reduction if image is actually noisy
            if laplacian_var < 100:  # Low sharpness indicates potential noise
                # Very gentle bilateral filter that preserves edges
                denoised = cv2.bilateralFilter(image, d=5, sigmaColor=20, sigmaSpace=20)

                # Blend with original to preserve details (70% original, 30% denoised)
                result = cv2.addWeighted(image, 0.7, denoised, 0.3, 0)
                return result

            return image

        except Exception as e:
            self.logger.warning(f"⚠️ Minimal noise reduction failed: {e}")
            return image

# Backward compatibility alias
ImagePreprocessor = AdvancedImagePreprocessor