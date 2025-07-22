import os
import sys

# Configure TensorFlow before any other imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
import tensorflow as tf

# Configure TensorFlow for compatibility
tf.config.run_functions_eagerly(True)
tf.compat.v1.disable_eager_execution()
tf.compat.v1.enable_eager_execution()

# Set memory growth to avoid GPU memory issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

import logging
import numpy as np
from typing import Optional, List, Tuple
from deepface import DeepFace
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
from PIL import Image

# Add the src directory to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, src_path)

from config.settings import Settings

class EmbeddingExtractor:
    """
    Face embedding extraction using DeepFace models.
    Supports batch processing and various face recognition models.
    """
    
    def __init__(self, 
                 model_name: str = None,
                 detector_backend: str = None):
        
        self.settings = Settings()
        self.model_name = model_name or self.settings.DEFAULT_MODEL
        self.detector_backend = detector_backend or self.settings.DEFAULT_DETECTOR
        self.logger = logging.getLogger(__name__)
        
        # Validate image before processing
        self._validate_model_setup()
        
        self.logger.info(f"Initialized EmbeddingExtractor with model: {self.model_name}, detector: {self.detector_backend}")
    
    def _validate_model_setup(self):
        """Validate model setup and warm up the model"""
        try:
            # Create a dummy image for model warmup
            dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Try to initialize the model with the dummy image
            DeepFace.represent(
                img_path=dummy_img,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            self.logger.info("Model warmup successful")
            
        except Exception as e:
            self.logger.warning(f"Model warmup failed (this may be normal): {e}")
    
    def _preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Preprocess image to ensure it's in the correct format"""
        try:
            # First try to read with OpenCV
            img = cv2.imread(image_path)
            if img is None:
                # Fallback to PIL
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            if img is None:
                self.logger.error(f"Could not load image: {image_path}")
                return None
                
            # Ensure image has 3 channels
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
            return img
            
        except Exception as e:
            self.logger.error(f"Failed to preprocess image {image_path}: {e}")
            return None
    
    def extract_single_embedding(self, image_path: str, enforce_detection: bool = True) -> Optional[np.ndarray]:
        """
        Extract embedding from a single image
        
        Args:
            image_path: Path to the image file
            enforce_detection: Whether to enforce face detection
            
        Returns:
            Face embedding as numpy array or None if extraction fails
        """
        try:
            # Validate image file exists
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return None
            
            # Preprocess image
            img_array = self._preprocess_image(image_path)
            if img_array is None:
                return None
            
            # Use the preprocessed image array instead of file path
            with tf.device('/CPU:0'):  # Force CPU execution for stability
                result = DeepFace.represent(
                    img_path=img_array,  # Use numpy array instead of path
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=enforce_detection,
                    normalization='base'  # Use base normalization
                )
            
            if result and len(result) > 0:
                embedding = np.array(result[0]["embedding"], dtype=np.float32)
                self.logger.debug(f"Successfully extracted embedding of shape {embedding.shape} from {image_path}")
                return embedding
            else:
                self.logger.warning(f"No face detected in {image_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to extract embedding from {image_path}: {e}")
            # Try alternative approach with file path directly
            try:
                self.logger.info(f"Trying direct file path approach for {image_path}")
                result = DeepFace.represent(
                    img_path=image_path,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=False,  # Relax detection for fallback
                    normalization='base'
                )
                
                if result and len(result) > 0:
                    embedding = np.array(result[0]["embedding"], dtype=np.float32)
                    self.logger.info(f"Fallback extraction successful for {image_path}")
                    return embedding
            except Exception as fallback_error:
                self.logger.error(f"Fallback extraction also failed for {image_path}: {fallback_error}")
            
            return None
    
    def extract_batch_embeddings(self, 
                                 image_paths: List[str],
                                 enforce_detection: bool = True,
                                 max_workers: Optional[int] = None) -> Tuple[List[np.ndarray], List[str]]:
        """
        Extract embeddings from multiple images in parallel
        
        Args:
            image_paths: List of image file paths
            enforce_detection: Whether to enforce face detection
            max_workers: Maximum number of worker processes (reduced for stability)
            
        Returns:
            Tuple of (embeddings list, successful paths list)
        """
        embeddings = []
        successful_paths = []
        
        # Limit max_workers for stability with TensorFlow
        if max_workers is None:
            max_workers = min(2, len(image_paths))  # Conservative default
        
        self.logger.info(f"Processing {len(image_paths)} images with {max_workers} workers")
        
        # Process in smaller batches to avoid memory issues
        batch_size = max(1, len(image_paths) // max_workers) if max_workers > 1 else len(image_paths)
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_paths)} images")
            
            # Process batch sequentially for now (more stable)
            for path in batch_paths:
                try:
                    embedding = self.extract_single_embedding(path, enforce_detection)
                    if embedding is not None:
                        embeddings.append(embedding)
                        successful_paths.append(path)
                        self.logger.debug(f"✅ Extracted embedding from {path}")
                    else:
                        self.logger.warning(f"❌ Failed to extract embedding from {path}")
                except Exception as e:
                    self.logger.error(f"❌ Error processing {path}: {e}")
        
        self.logger.info(f"Successfully extracted {len(embeddings)} embeddings from {len(image_paths)} images")
        return embeddings, successful_paths
    
    def _extract_embedding_worker(self, image_path: str, enforce_detection: bool) -> Optional[np.ndarray]:
        """Worker function for parallel embedding extraction"""
        return self.extract_single_embedding(image_path, enforce_detection)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for the current model"""
        dimensions = {
            'VGG-Face': 4096,
            'Facenet': 128,
            'Facenet512': 512,
            'OpenFace': 128,
            'DeepFace': 4096,
            'DeepID': 160,
            'ArcFace': 512,
            'Dlib': 128,
            'SFace': 128
        }
        
        return dimensions.get(self.model_name, 512)  # Default to 512 if unknown
    
    def validate_model_availability(self) -> bool:
        """Validate that the specified model and detector are available"""
        try:
            # Try to create a small test embedding with a simple test image
            test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            result = DeepFace.represent(
                img_path=test_img,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                normalization='base'
            )
            return True
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return False
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models"""
        return ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']
    
    def get_supported_detectors(self) -> List[str]:
        """Get list of supported detector backends"""
        return ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']