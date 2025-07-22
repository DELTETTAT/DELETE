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
    
    def preprocess_image(self, input_path: str, output_path: str) -> bool:
        """
        Main preprocessing pipeline
        
        Args:
            input_path: Path to input image
            output_path: Path to save processed image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load and validate image
            image = self.image_processor.load_image(input_path)
            if image is None:
                return False
            
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
            return self.image_processor.save_image(image, output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Preprocessing failed for {input_path}: {e}")
            return False
    
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
