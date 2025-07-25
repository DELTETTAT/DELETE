# Applying the changes to update paths and create data directories within the src folder.
import os
import logging
from typing import Optional
from pathlib import Path

class Settings:
    """
    Centralized configuration settings for the face matching system.
    """
    
    def __init__(self):
        # Base directories
        self.BASE_DIR = Path(__file__).parent.parent.parent
        self.SRC_DIR = self.BASE_DIR / "src"

        # Organized data directories inside src
        self.DATA_DIR = self.SRC_DIR / "data"
        self.TEMP_DIR = self.DATA_DIR / "temp"
        self.PREPROCESSED_DIR = self.DATA_DIR / "preprocessed"
        self.EMBEDDINGS_DIR = self.DATA_DIR / "embeddings"
        self.CACHE_DIR = self.DATA_DIR / "cache"
        self.INDEXES_DIR = self.DATA_DIR / "indexes"
        
        # Backward compatibility aliases
        self.EMBED_DIR = self.EMBEDDINGS_DIR  # Legacy alias
        self.UPLOAD_DIR = self.DATA_DIR / "uploads"  # Create uploads directory
        
        # Model configuration
        self.DEFAULT_MODEL = "Facenet512"
        self.DEFAULT_DETECTOR = "retinaface"
        self.AVAILABLE_MODELS = [
            "VGG-Face", "Facenet", "Facenet512", "OpenFace", 
            "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"
        ]
        self.AVAILABLE_DETECTORS = [
            "opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"
        ]
        
        # Image processing configuration
        self.DEFAULT_TARGET_SIZE = (160, 160)
        self.SUPPORTED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp')
        self.DEFAULT_QUALITY_THRESHOLD = 0.7
        self.DEFAULT_IMAGE_QUALITY = 95
        
        # Search configuration
        self.DEFAULT_SEARCH_K = 5
        self.DEFAULT_DISTANCE_THRESHOLD = 450.0
        self.MAX_SEARCH_RESULTS = 50

        # Index settings
        self.INDEX_PATH = self.INDEXES_DIR / "face_index.faiss"
        self.METADATA_PATH = self.INDEXES_DIR / "face_metadata.pkl"
        
        # Processing configuration
        self.DEFAULT_MAX_WORKERS = None  # Will use system default
        self.BATCH_SIZE = 32
        self.ENFORCE_DETECTION = True
        
        # Logging configuration
        self.LOG_LEVEL = logging.INFO
        self.LOG_FORMAT = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
        self.LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
        
        # Create directories on initialization
        self._create_directories()
        
        # Setup basic logging
        self._setup_logging()
    
    def _create_directories(self):
        """Create required directories if they don't exist"""
        directories = [
            self.DATA_DIR,
            self.TEMP_DIR,
            self.PREPROCESSED_DIR,
            self.EMBEDDINGS_DIR,
            self.CACHE_DIR,
            self.INDEXES_DIR,
            self.UPLOAD_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _setup_logging(self):
        """Setup basic logging configuration"""
        logging.basicConfig(
            level=self.LOG_LEVEL,
            format=self.LOG_FORMAT,
            datefmt=self.LOG_DATE_FORMAT
        )
        self.logger = logging.getLogger(__name__)
    
    def get_model_config(self, model_name: Optional[str] = None, 
                        detector_backend: Optional[str] = None) -> dict:
        """Get model configuration"""
        return {
            'model_name': model_name or self.DEFAULT_MODEL,
            'detector_backend': detector_backend or self.DEFAULT_DETECTOR,
            'enforce_detection': self.ENFORCE_DETECTION
        }
    
    def get_preprocessing_config(self, target_size: Optional[tuple] = None,
                               enable_alignment: bool = True,
                               enable_enhancement: bool = True,
                               quality_threshold: Optional[float] = None) -> dict:
        """Get preprocessing configuration"""
        return {
            'target_size': target_size or self.DEFAULT_TARGET_SIZE,
            'enable_face_alignment': enable_alignment,
            'enable_enhancement': enable_enhancement,
            'quality_threshold': quality_threshold or self.DEFAULT_QUALITY_THRESHOLD
        }
    
    def get_search_config(self, k: Optional[int] = None,
                         threshold: Optional[float] = None) -> dict:
        """Get search configuration"""
        return {
            'k': min(k or self.DEFAULT_SEARCH_K, self.MAX_SEARCH_RESULTS),
            'threshold': threshold or self.DEFAULT_DISTANCE_THRESHOLD,
            'enforce_detection': self.ENFORCE_DETECTION
        }
    
    def get_processing_config(self, max_workers: Optional[int] = None) -> dict:
        """Get processing configuration"""
        return {
            'max_workers': max_workers or self.DEFAULT_MAX_WORKERS,
            'batch_size': self.BATCH_SIZE,
            'enforce_detection': self.ENFORCE_DETECTION
        }
    
    def validate_model(self, model_name: str) -> bool:
        """Validate if model is supported"""
        return model_name in self.AVAILABLE_MODELS
    
    def validate_detector(self, detector_name: str) -> bool:
        """Validate if detector is supported"""
        return detector_name in self.AVAILABLE_DETECTORS
    
    def is_image_file(self, filename: str) -> bool:
        """Check if filename has supported image extension"""
        ext = os.path.splitext(filename)[1].lower()
        return ext in self.SUPPORTED_IMAGE_EXTENSIONS
    
    def get_environment_config(self) -> dict:
        """Get environment-specific configuration"""
        return {
            'base_dir': self.BASE_DIR,
            'upload_dir': self.UPLOAD_DIR,
            'embed_dir': self.EMBED_DIR,
            'temp_dir': self.TEMP_DIR,
            'preprocessed_dir': self.PREPROCESSED_DIR
        }
    
    def update_directories(self, **kwargs):
        """Update directory configurations"""
        for key, value in kwargs.items():
            if hasattr(self, key.upper()):
                setattr(self, key.upper(), value)
                # Create directory if it doesn't exist
                os.makedirs(value, exist_ok=True)
    
    def get_logging_config(self) -> dict:
        """Get logging configuration"""
        return {
            'level': self.LOG_LEVEL,
            'format': self.LOG_FORMAT,
            'datefmt': self.LOG_DATE_FORMAT
        }
    
    def __str__(self) -> str:
        """String representation of settings"""
        config_info = [
            f"Base Directory: {self.BASE_DIR}",
            f"Upload Directory: {self.UPLOAD_DIR}",
            f"Embeddings Directory: {self.EMBEDDINGS_DIR}",
            f"Default Model: {self.DEFAULT_MODEL}",
            f"Default Detector: {self.DEFAULT_DETECTOR}",
            f"Target Size: {self.DEFAULT_TARGET_SIZE}",
            f"Distance Threshold: {self.DEFAULT_DISTANCE_THRESHOLD}"
        ]
        return "\n".join(config_info)

# Global settings instance
settings = Settings()
