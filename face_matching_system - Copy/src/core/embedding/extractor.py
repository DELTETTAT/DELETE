import os
import sys

# Configure TensorFlow before any other imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Fix OpenMP runtime conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
# Disable TensorFlow's multi-threading to avoid conflicts with our parallelism
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import tensorflow as tf

# Configure TensorFlow for better compatibility with multiprocessing
try:
    # Disable GPU if not available to avoid warnings
    tf.config.set_visible_devices([], 'GPU')

    # Set threading configuration
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    # Ensure eager execution
    tf.config.run_functions_eagerly(True)

except Exception as e:
    print(f"TensorFlow configuration warning: {e}")

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

def _extract_embedding_worker_process(image_path: str, model_name: str, detector_backend: str, enforce_detection: bool) -> Optional[np.ndarray]:
    """
    Worker function for process-based parallel embedding extraction.
    This runs in a separate process to achieve true parallelism.
    """
    try:
        import os
        # Configure TensorFlow for this worker process
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_NUM_INTEROP_THREADS'] = '1' 
        os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

        import tensorflow as tf
        from deepface import DeepFace
        import cv2
        from PIL import Image

        # Configure TensorFlow for this process
        tf.config.set_visible_devices([], 'GPU')  # Force CPU usage
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        # Validate image file exists
        if not os.path.exists(image_path):
            return None

        # Preprocess image
        try:
            img = cv2.imread(image_path)
            if img is None:
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            if img is None:
                return None

            # Ensure image has 3 channels
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception:
            return None

        # Extract embedding with fallback strategies
        fallback_configs = [
            # Primary approach
            {'img_input': img, 'model': model_name, 'detector': detector_backend, 'enforce': enforce_detection},
            # Fallback 1: Use image path directly
            {'img_input': image_path, 'model': model_name, 'detector': detector_backend, 'enforce': False},
            # Fallback 2: Use opencv detector
            {'img_input': image_path, 'model': model_name, 'detector': 'opencv', 'enforce': False},
            # Fallback 3: Use different model
            {'img_input': image_path, 'model': 'Facenet', 'detector': 'opencv', 'enforce': False},
        ]

        for config in fallback_configs:
            try:
                result = DeepFace.represent(
                    img_path=config['img_input'],
                    model_name=config['model'],
                    detector_backend=config['detector'],
                    enforce_detection=config['enforce'],
                    normalization='base'
                )

                if result and len(result) > 0:
                    embedding = np.array(result[0]["embedding"], dtype=np.float32)
                    return embedding

            except Exception:
                continue

        return None

    except Exception:
        return None

# Define a simple cache class
class EmbeddingCache:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache = {}
        self.access_order = []

    def get_embedding(self, image_path: str) -> Optional[np.ndarray]:
        if image_path in self.cache:
            # Move to the end to indicate recent use
            self.access_order.remove(image_path)
            self.access_order.append(image_path)
            return self.cache[image_path]
        return None

    def store_embedding(self, image_path: str, embedding: np.ndarray):
        if image_path in self.cache:
            # Update existing entry
            self.cache[image_path] = embedding
            self.access_order.remove(image_path)
            self.access_order.append(image_path)
        else:
            # Add new entry
            if len(self.cache) >= self.capacity:
                # Evict least recently used entry
                lru_path = self.access_order.pop(0)
                del self.cache[lru_path]
            self.cache[image_path] = embedding
            self.access_order.append(image_path)

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

        # Initialize embedding cache
        self.cache = EmbeddingCache()

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

    def _validate_image_file(self, image_path: str) -> bool:
        """Validate image file before processing"""
        try:
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return False

            # Check file size
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                self.logger.error(f"Image file is empty: {image_path}")
                return False

            if file_size < 100:  # Very small files are likely corrupted
                self.logger.warning(f"Image file very small ({file_size} bytes): {image_path}")

            # Check file extension
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            file_ext = os.path.splitext(image_path)[1].lower()
            if file_ext not in valid_extensions:
                self.logger.warning(f"Unsupported image format {file_ext}: {image_path}")

            return True

        except Exception as e:
            self.logger.error(f"Error validating image {image_path}: {e}")
            return False

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
            # Validate image file
            if not self._validate_image_file(image_path):
                return None

            # Check cache first
            cached_embedding = self.cache.get_embedding(image_path)
            if cached_embedding is not None:
                self.logger.debug(f"Using cached embedding for {image_path}")
                return cached_embedding

            # Preprocess image
            img_array = self._preprocess_image(image_path)
            if img_array is None:
                return None

            # Use the preprocessed image array instead of file path
            # Remove tf.device context to avoid Keras tensor issues
            result = DeepFace.represent(
                img_path=img_array,  # Use numpy array instead of path
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=enforce_detection,
                normalization='base'  # Use base normalization
            )

            if result and len(result) > 0:
                embedding = np.array(result[0]["embedding"], dtype=np.float32)

                # Ensure embedding is 1D array
                if len(embedding.shape) > 1:
                    embedding = embedding.flatten()

                # Validate embedding values
                if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                    self.logger.warning(f"Invalid embedding values detected for {image_path}")
                    return None

                self.logger.debug(f"Successfully extracted embedding of shape {embedding.shape} from {image_path}")

                # Store in cache
                self.cache.store_embedding(image_path, embedding)

                return embedding
            else:
                self.logger.warning(f"No face detected in {image_path}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to extract embedding from {image_path}: {e}")

            # Log image details for debugging
            try:
                import os
                if os.path.exists(image_path):
                    file_size = os.path.getsize(image_path)
                    self.logger.debug(f"Image details - Size: {file_size} bytes, Path: {image_path}")
                else:
                    self.logger.error(f"Image file does not exist: {image_path}")
                    return None
            except Exception as debug_e:
                self.logger.debug(f"Could not get image details: {debug_e}")

            # Try multiple fallback approaches with more comprehensive options
            fallback_attempts = [
                # Attempt 1: Direct file path, no enforcement
                lambda: DeepFace.represent(
                    img_path=image_path,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=False,
                    normalization='base'
                ),
                # Attempt 2: OpenCV detector (most stable)
                lambda: DeepFace.represent(
                    img_path=image_path,
                    model_name=self.model_name,
                    detector_backend='opencv',
                    enforce_detection=False,
                    normalization='base'
                ),
                # Attempt 3: Different model with OpenCV
                lambda: DeepFace.represent(
                    img_path=image_path,
                    model_name='Facenet',
                    detector_backend='opencv',
                    enforce_detection=False,
                    normalization='base'
                ),
                # Attempt 4: Use preprocessed image array
                lambda: DeepFace.represent(
                    img_path=self._preprocess_image(image_path) if self._preprocess_image(image_path) is not None else image_path,
                    model_name='Facenet',
                    detector_backend='opencv',
                    enforce_detection=False,
                    normalization='base'
                ),
                # Attempt 5: Try with SSD detector
                lambda: DeepFace.represent(
                    img_path=image_path,
                    model_name='Facenet',
                    detector_backend='ssd',
                    enforce_detection=False,
                    normalization='base'
                )
            ]

            for i, attempt in enumerate(fallback_attempts):
                try:
                    self.logger.info(f"Trying fallback approach {i+1} for {image_path}")
                    result = attempt()

                    if result and len(result) > 0:
                        embedding = np.array(result[0]["embedding"], dtype=np.float32)
                        self.logger.info(f"Fallback {i+1} successful for {image_path}")
                        return embedding
                except Exception as fallback_error:
                    self.logger.debug(f"Fallback {i+1} failed for {image_path}: {fallback_error}")
                    continue

            self.logger.error(f"All {len(fallback_attempts)} fallback attempts failed for {image_path}")
            return None

    def extract_batch_embeddings_with_preprocessing(self, image_paths: List[str], 
                                              enforce_detection: bool = True,
                                              max_workers: Optional[int] = None,
                                              use_preprocessing: bool = False,
                                              progress_callback: Optional[callable] = None) -> Tuple[List[np.ndarray], List[str]]:
        """
        Extract embeddings with optional preprocessing, avoiding image duplication

        Args:
            image_paths: List of image paths to process
            enforce_detection: Whether to enforce face detection
            max_workers: Maximum number of workers for parallel processing
            use_preprocessing: Whether to apply preprocessing before embedding extraction
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (embeddings, successful_paths)
        """
        if not image_paths:
            return [], []

        self.logger.info(f"Extracting embeddings from {len(image_paths)} images (preprocessing: {use_preprocessing})")

        embeddings = []
        successful_paths = []
        processed_count = 0

        # Determine number of workers - fewer workers work better for DeepFace
        if max_workers is None:
            max_workers = min(3, len(image_paths))  # Limited workers for better performance

        def process_single_image_with_preprocessing(args):
            image_path, idx = args
            try:
                if not os.path.exists(image_path):
                    self.logger.warning(f"Image not found: {image_path}")
                    return None, image_path, idx

                processed_path = image_path
                temp_file_created = False

                # Apply preprocessing if requested
                if use_preprocessing:
                    try:
                        import tempfile
                        import shutil
                        from core.preprocessing.preprocessor import ImagePreprocessor

                        # Create temporary file for preprocessing
                        temp_dir = tempfile.mkdtemp(prefix="temp_preprocess_")
                        temp_filename = os.path.basename(image_path)
                        temp_path = os.path.join(temp_dir, temp_filename)

                        # Copy original to temp location
                        shutil.copy2(image_path, temp_path)

                        # Apply preprocessing
                        preprocessor = ImagePreprocessor()
                        result = preprocessor.preprocess_image(temp_path, temp_path)

                        if result.get('success', False):
                            processed_path = temp_path
                            temp_file_created = True
                        else:
                            self.logger.warning(f"Preprocessing failed for {image_path}, using original")
                            # Clean up failed temp file
                            shutil.rmtree(temp_dir)

                    except Exception as e:
                        self.logger.error(f"Preprocessing error for {image_path}: {e}")

                # Extract embedding from processed image
                embedding = self.extract_single_embedding(processed_path, enforce_detection)

                # Clean up temporary file
                if temp_file_created and os.path.exists(processed_path):
                    try:
                        shutil.rmtree(os.path.dirname(processed_path))
                    except:
                        pass

                if embedding is not None:
                    return embedding, image_path, idx  # Return original path, not temp path
                else:
                    return None, image_path, idx

            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {e}")
                return None, image_path, idx

        # Process in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with index for ordering
            futures = {executor.submit(process_single_image_with_preprocessing, (path, i)): i 
                      for i, path in enumerate(image_paths)}

            # Process completed tasks
            results = [None] * len(image_paths)  # Pre-allocate for ordering

            for future in as_completed(futures):
                try:
                    embedding, image_path, idx = future.result()
                    results[idx] = (embedding, image_path)
                    processed_count += 1

                    # Update progress
                    if progress_callback:
                        progress_callback(processed_count, len(image_paths))

                except Exception as e:
                    self.logger.error(f"Task failed: {e}")
                    processed_count += 1

                    if progress_callback:
                        progress_callback(processed_count, len(image_paths))

        # Collect successful results in order
        for embedding, image_path in results:
            if embedding is not None:
                embeddings.append(embedding)
                successful_paths.append(image_path)

        self.logger.info(f"Successfully extracted {len(embeddings)}/{len(image_paths)} embeddings")
        return embeddings, successful_paths

    def extract_batch_embeddings(self, image_paths: List[str], 
                               enforce_detection: bool = True,
                               max_workers: Optional[int] = None,
                               use_preprocessing: bool = False) -> Tuple[List[np.ndarray], List[str]]:
        """
        Extract embeddings from multiple images in parallel using optimized thread-based parallelism

        Args:
            image_paths: List of image file paths
            enforce_detection: Whether to enforce face detection
            max_workers: Maximum number of worker threads
            use_preprocessing: Whether to use preprocessing or not

        Returns:
            Tuple of (embeddings list, successful paths list)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time

        if not image_paths:
            return [], []

        # Use fewer threads for better performance with DeepFace
        if max_workers is None:
            max_workers = min(3, len(image_paths))  # Limited threads work better for DeepFace

        actual_workers = min(max_workers, len(image_paths))

        self.logger.info(f"🔧 Processing {len(image_paths)} images with {actual_workers} threads")
        start_time = time.time()

        embeddings = []
        successful_paths = []

        # Use ThreadPoolExecutor instead of ProcessPoolExecutor for better performance
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.extract_single_embedding, path, enforce_detection): path
                for path in image_paths
            }

            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                completed += 1

                try:
                    embedding = future.result(timeout=30)
                    if embedding is not None:
                        embeddings.append(embedding)
                        successful_paths.append(path)
                        self.logger.debug(f"✅ Extracted embedding from {path}")
                    else:
                        self.logger.warning(f"❌ Failed to extract embedding from {path}")
                except Exception as e:
                    self.logger.error(f"❌ Error processing {path}: {e}")

                # Progress logging with timing
                if completed % 5 == 0 or completed == len(image_paths):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    self.logger.info(f"Processed {completed}/{len(image_paths)} images ({rate:.1f} imgs/sec)")

        total_time = time.time() - start_time
        self.logger.info(f"Successfully extracted {len(embeddings)} embeddings from {len(image_paths)} images in {total_time:.1f}s")
        return embeddings, successful_paths

    def _extract_batch_sequential(self, image_paths: List[str], enforce_detection: bool) -> Tuple[List[np.ndarray], List[str]]:
        """Fallback sequential processing"""
        embeddings = []
        successful_paths = []

        for i, path in enumerate(image_paths):
            try:
                embedding = self.extract_single_embedding(path, enforce_detection)
                if embedding is not None:
                    embeddings.append(embedding)
                    successful_paths.append(path)

                if (i + 1) % 10 == 0:
                    self.logger.info(f"Sequential: Processed {i + 1}/{len(image_paths)} images")

            except Exception as e:
                self.logger.error(f"Sequential processing error for {path}: {e}")

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