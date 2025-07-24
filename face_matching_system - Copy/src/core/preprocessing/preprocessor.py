import os
import numpy as np
import logging
from PIL import Image, ExifTags
from typing import Tuple, Optional
from .enhancement import ImageEnhancer
from .alignment import FaceAligner
from .quality import QualityAssessor
import sys
import os
# Add the src directory to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, src_path)

from utils.image.processing import ImageProcessor
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

class ImagePreprocessor:
    """
    Main image preprocessing coordinator.
    Orchestrates enhancement, alignment, and quality assessment.
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (160, 160),
                 enable_face_alignment: bool = True,
                 enable_enhancement: bool = True,
                 quality_threshold: float = 0.7):
        
        self.target_size = target_size
        self.enable_face_alignment = enable_face_alignment
        self.enable_enhancement = enable_enhancement
        self.quality_threshold = quality_threshold
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.image_processor = ImageProcessor()
        self.enhancer = ImageEnhancer() if enable_enhancement else None
        self.aligner = FaceAligner(target_size) if enable_face_alignment else None
        self.quality_assessor = QualityAssessor(quality_threshold)
    
    def preprocess_image(self, input_path: str, output_path: Optional[str] = None) -> dict:
        """
        Main preprocessing pipeline

        Args:
            input_path: Path to input image
            output_path: Path to save processed image

        Returns:
            Dictionary with success status and details
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
            
            # Correct orientation based on EXIF
            image = self._correct_orientation(image, input_path)
            
            # Enhance image quality if enabled
            if self.enable_enhancement and self.enhancer:
                image = self.enhancer.enhance_image(image)
            
            # Face alignment if enabled
            if self.enable_face_alignment and self.aligner:
                aligned_image = self.aligner.align_face(image)
                if aligned_image is not None:
                    image = aligned_image
                else:
                    self.logger.warning(f"⚠️ Face alignment failed for {os.path.basename(input_path)}")
            
            # Resize to target size
            image = self.image_processor.resize_image(image, self.target_size)
            
            # Normalize
            image = self.image_processor.normalize_image(image)
            
            # Save processed image
            save_success = self.image_processor.save_image(image, output_path)
            return {'success': save_success}
            
        except Exception as e:
            self.logger.error(f"❌ Preprocessing failed for {input_path}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _correct_orientation(self, image: np.ndarray, image_path: str) -> np.ndarray:
        """Correct image orientation based on EXIF data"""
        try:
            # Convert back to PIL to read EXIF
            pil_image = Image.open(image_path)
            
            # Get EXIF data
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
            
            return np.array(pil_image.convert('RGB'))
            
        except Exception as e:
            self.logger.warning(f"⚠️ EXIF orientation correction failed: {e}")
            return image
    
    def assess_quality(self, image_path: str) -> dict:
        """Assess image quality using quality assessor"""
        try:
            image = self.image_processor.load_image(image_path)
            if image is None:
                return {"error": "Failed to load image"}
            
            return self.quality_assessor.assess_image_quality(image)
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            return {"error": str(e)}
    
    def is_quality_acceptable(self, image_path: str) -> bool:
        """Check if image quality meets threshold"""
        quality_metrics = self.assess_quality(image_path)
        return quality_metrics.get('is_acceptable', False)

    def preprocess_single_image(self, image_path: str, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Preprocess a single image

        Args:
            image_path: Path to the image file
            output_dir: Directory to save preprocessed image (optional)

        Returns:
            Path to preprocessed image or None if failed
        """
        if output_dir is None:
            # Create a default output path if not provided
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_dir = os.path.dirname(image_path)  # Use same directory as input
            output_path = os.path.join(output_dir, f"processed_{name}{ext}")
        else:
            # Create output path in the specified directory
            filename = os.path.basename(image_path)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
        
        result = self.preprocess_image(image_path, output_path)
        if result.get('success', False):
            return output_path
        else:
            return None

    def preprocess_batch(self, image_paths: List[str], 
                        output_dir: Optional[str] = None,
                        max_workers: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """
        Preprocess multiple images in parallel

        Args:
            image_paths: List of image file paths
            output_dir: Directory to save preprocessed images (optional)
            max_workers: Maximum number of worker threads

        Returns:
            Tuple of (successful paths, failed paths)
        """
        if max_workers is None:
            max_workers = min(4, len(image_paths))

        successful_paths = []
        failed_paths = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self.preprocess_image, path, output_dir): path 
                for path in image_paths
            }

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result_path = future.result()
                    if result_path:
                        successful_paths.append(result_path)
                    else:
                        failed_paths.append(path)
                except Exception as e:
                    self.logger.error(f"Failed to preprocess {path}: {e}")
                    failed_paths.append(path)

        return successful_paths, failed_paths