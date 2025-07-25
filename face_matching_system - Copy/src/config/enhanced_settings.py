import os
import logging
import importlib
from typing import Optional, Dict, Any, List
from pathlib import Path

class EnhancedSettings:
    """
    Enhanced configuration settings with validation and error handling.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Directory configuration
        self.BASE_DIR = os.getcwd()
        self.UPLOAD_DIR = os.path.join(self.BASE_DIR, "uploads")
        self.EMBED_DIR = os.path.join(self.BASE_DIR, "embeddings")
        self.TEMP_DIR = os.path.join(self.BASE_DIR, "temp_processing")
        self.PREPROCESSED_DIR = os.path.join(self.BASE_DIR, "preprocessed_images")
        
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
        
        # Processing configuration
        self.DEFAULT_MAX_WORKERS = None
        self.BATCH_SIZE = 32
        self.ENFORCE_DETECTION = True
        
        # Memory management
        self.MAX_MEMORY_USAGE_MB = 2048
        self.CLEANUP_INTERVAL_SECONDS = 300
        
        # Logging configuration
        self.LOG_LEVEL = logging.INFO
        self.LOG_FORMAT = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
        self.LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
        
        # Initialize and validate
        self._create_directories()
        self._setup_logging()
        self._validate_dependencies()
    
    def _create_directories(self):
        """Create required directories with proper error handling"""
        directories = [
            self.UPLOAD_DIR,
            self.EMBED_DIR,
            self.TEMP_DIR,
            self.PREPROCESSED_DIR
        ]
        
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                
                # Test write permissions
                test_file = os.path.join(directory, '.write_test')
                try:
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                except Exception as e:
                    self.logger.warning(f"Directory {directory} may not be writable: {e}")
                    
            except Exception as e:
                self.logger.error(f"Failed to create directory {directory}: {e}")
    
    def _setup_logging(self):
        """Setup logging with enhanced configuration"""
        try:
            logging.basicConfig(
                level=self.LOG_LEVEL,
                format=self.LOG_FORMAT,
                datefmt=self.LOG_DATE_FORMAT,
                force=True
            )
            self.logger = logging.getLogger(__name__)
            self.logger.info("Enhanced settings initialized")
        except Exception as e:
            print(f"Failed to setup logging: {e}")
    
    def _validate_dependencies(self) -> Dict[str, bool]:
        """Validate all required dependencies"""
        dependencies = {
            'deepface': False,
            'faiss': False,
            'opencv': False,
            'tensorflow': False,
            'mediapipe': False,
            'numpy': False,
            'PIL': False,
            'plotly': False
        }
        
        for dep in dependencies:
            try:
                if dep == 'opencv':
                    import cv2
                    # Verify OpenCV is working properly
                    _ = cv2.__version__
                    dependencies[dep] = True
                elif dep == 'PIL':
                    importlib.import_module('PIL')
                    dependencies[dep] = True
                elif dep == 'faiss':
                    try:
                        importlib.import_module('faiss')
                        dependencies[dep] = True
                    except ImportError:
                        importlib.import_module('faiss_cpu')
                        dependencies[dep] = True
                else:
                    importlib.import_module(dep)
                    dependencies[dep] = True
            except (ImportError, AttributeError, Exception) as e:
                self.logger.debug(f"Failed to validate {dep}: {e}")
                dependencies[dep] = False
        
        self.dependency_status = dependencies
        return dependencies
    
    def validate_model_availability(self, model_name: str) -> bool:
        """Validate if a specific model is actually available"""
        if not self.validate_model(model_name):
            return False
        
        try:
            # Try to import deepface and test model availability
            import deepface
            # This is a lightweight check - you might want to enhance this
            return True
        except Exception as e:
            self.logger.error(f"Model {model_name} validation failed: {e}")
            return False
    
    def validate_detector_availability(self, detector_name: str) -> bool:
        """Validate if a specific detector is actually available"""
        if not self.validate_detector(detector_name):
            return False
        
        try:
            if detector_name == 'opencv':
                import cv2
                return True
            elif detector_name == 'mediapipe':
                import mediapipe
                return True
            elif detector_name in ['mtcnn', 'retinaface']:
                import tensorflow
                return True
            else:
                return True  # Basic validation passed
        except ImportError:
            self.logger.error(f"Detector {detector_name} dependencies not available")
            return False
    
    def get_safe_model_config(self, model_name: Optional[str] = None, 
                             detector_backend: Optional[str] = None) -> dict:
        """Get model configuration with validation"""
        # Fallback to available models/detectors if specified ones aren't available
        safe_model = model_name or self.DEFAULT_MODEL
        safe_detector = detector_backend or self.DEFAULT_DETECTOR
        
        if not self.validate_model_availability(safe_model):
            # Try to find a working model
            for model in self.AVAILABLE_MODELS:
                if self.validate_model_availability(model):
                    safe_model = model
                    break
            else:
                self.logger.warning("No working models found, using default")
        
        if not self.validate_detector_availability(safe_detector):
            # Try to find a working detector
            for detector in self.AVAILABLE_DETECTORS:
                if self.validate_detector_availability(detector):
                    safe_detector = detector
                    break
            else:
                self.logger.warning("No working detectors found, using default")
        
        return {
            'model_name': safe_model,
            'detector_backend': safe_detector,
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
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        health = {
            'directories': {},
            'dependencies': self.dependency_status,
            'memory': self._check_memory_usage(),
            'permissions': {}
        }
        
        # Check directories
        for attr_name in ['UPLOAD_DIR', 'EMBED_DIR', 'TEMP_DIR', 'PREPROCESSED_DIR']:
            path = getattr(self, attr_name)
            health['directories'][attr_name] = {
                'exists': os.path.exists(path),
                'readable': os.access(path, os.R_OK) if os.path.exists(path) else False,
                'writable': os.access(path, os.W_OK) if os.path.exists(path) else False
            }
        
        return health
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / (1024 * 1024)
            }
        except ImportError:
            return {'error': 'psutil not available'}
        except Exception as e:
            return {'error': str(e)}
    
    def cleanup_resources(self):
        """Clean up system resources"""
        try:
            import gc
            gc.collect()
            
            # Clean temporary files older than 1 hour
            import time
            current_time = time.time()
            
            if os.path.exists(self.TEMP_DIR):
                for root, dirs, files in os.walk(self.TEMP_DIR):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.getmtime(file_path) < current_time - 3600:  # 1 hour
                            try:
                                os.remove(file_path)
                                self.logger.debug(f"Cleaned up temporary file: {file_path}")
                            except Exception:
                                pass
                                
        except Exception as e:
            self.logger.error(f"Resource cleanup failed: {e}")

# Create a global enhanced settings instance
enhanced_settings = EnhancedSettings()