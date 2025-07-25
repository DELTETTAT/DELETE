"""
Advanced image preprocessing modules for face matching system.

Provides comprehensive image preprocessing capabilities with GPU optimization:
- enhancement: Advanced brightness, contrast, and sharpness adjustments with GPU acceleration
- alignment: Multi-method face alignment using MediaPipe, OpenCV DNN, and Haar cascades
- quality: Comprehensive image quality assessment with advanced metrics
- preprocessor: Advanced coordination class with automatic hardware optimization
"""

from .preprocessor import AdvancedImagePreprocessor, ImagePreprocessor
from .enhancement import AdvancedImageEnhancer, ImageEnhancer
from .alignment import AdvancedFaceAligner, FaceAligner
from .quality import AdvancedQualityAssessor, QualityAssessor

# For backward compatibility, ensure aliases are available
ImagePreprocessor = AdvancedImagePreprocessor
ImageEnhancer = AdvancedImageEnhancer
FaceAligner = AdvancedFaceAligner
QualityAssessor = AdvancedQualityAssessor

__all__ = [
    'AdvancedImagePreprocessor',
    'AdvancedImageEnhancer', 
    'AdvancedFaceAligner',
    'AdvancedQualityAssessor',
    'ImagePreprocessor',
    'ImageEnhancer',
    'FaceAligner',
    'QualityAssessor'
]