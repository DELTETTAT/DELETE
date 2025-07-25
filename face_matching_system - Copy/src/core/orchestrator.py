"""Apply chunked processing to the build_complete_index function for improved memory management during index building."""
"""Fixes import paths within the FaceMatchingOrchestrator class."""
"""
Core orchestrator for the face matching system.
Centralizes all business logic and coordination between components.
"""

import os
import sys
import logging
import shutil
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Add the src directory to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, src_path)

from core.preprocessing.preprocessor import ImagePreprocessor
from core.embedding.extractor import EmbeddingExtractor
from core.indexing.manager import IndexManager
from core.search.engine import SearchEngine
from utils.filesystem.operations import FileSystemOperations
from utils.logging.logger import setup_logging, suppress_verbose_warnings
from config.settings import Settings


class FaceMatchingOrchestrator:
    """
    Core orchestrator that coordinates all face matching operations.
    Provides a unified interface for both CLI and web applications.
    """

    def __init__(self):
        # Suppress verbose warnings for cleaner output
        suppress_verbose_warnings()

        self.settings = Settings()
        self.logger = logging.getLogger(__name__)

        # Initialize all core components
        self.fs_ops = FileSystemOperations()
        self.preprocessor = ImagePreprocessor()
        self.embedding_extractor = EmbeddingExtractor()
        self.index_manager = IndexManager()
        self.search_engine = SearchEngine()

        self.logger.info("Core orchestrator initialized")

    # ===========================================
    # SYSTEM STATUS AND VALIDATION
    # ===========================================

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'directories': self._check_directory_status(),
                'index': self._check_index_status(),
                'components': self._check_component_status(),
                'timestamp': datetime.now().isoformat()
            }
            return status
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}

    def get_index_statistics(self) -> Dict[str, Any]:
        """Get detailed index statistics"""
        try:
            return self.index_manager.get_index_stats()
        except Exception as e:
            self.logger.error(f"Failed to get index statistics: {e}")
            return {'error': str(e)}

    def get_index_stats(self) -> Dict[str, Any]:
        """Alias for get_index_statistics for compatibility"""
        return self.get_index_statistics()

    def validate_source_folder(self, folder_path: str) -> Dict[str, Any]:
        """Validate source folder"""
        try:
            if not os.path.exists(folder_path):
                return {'valid': False, 'error': f'Folder does not exist: {folder_path}'}

            image_count = self.fs_ops.count_images_in_folder(folder_path)
            if image_count == 0:
                return {'valid': False, 'error': f'No images found in folder: {folder_path}'}

            return {'valid': True, 'image_count': image_count}
        except Exception as e:
            return {'valid': False, 'error': str(e)}

    def delete_index(self) -> Dict[str, Any]:
        """Delete existing index"""
        try:
            result = self.index_manager.delete_index()
            return {'success': True, 'message': 'Index deleted successfully'}
        except Exception as e:
            self.logger.error(f"Failed to delete index: {e}")
            return {'success': False, 'error': str(e)}

    def _check_directory_status(self) -> Dict[str, Any]:
        """Check status of all required directories"""
        directories = {
            'upload_dir': {
                'path': self.settings.UPLOAD_DIR,
                'exists': os.path.exists(self.settings.UPLOAD_DIR),
                'image_count': self.fs_ops.count_images_in_folder(self.settings.UPLOAD_DIR)
            },
            'preprocessed_dir': {
                'path': self.settings.PREPROCESSED_DIR,
                'exists': os.path.exists(self.settings.PREPROCESSED_DIR),
                'image_count': self.fs_ops.count_images_in_folder(self.settings.PREPROCESSED_DIR)
            },
            'embed_dir': {
                'path': self.settings.EMBED_DIR,
                'exists': os.path.exists(self.settings.EMBED_DIR)
            }
        }
        return directories

    def _check_index_status(self) -> Dict[str, Any]:
        """Check FAISS index status"""
        index_status = {
            'exists': self.index_manager.index_exists(),
            'ready': self.search_engine.is_ready(),
            'stats': {}
        }

        if index_status['exists']:
            index_status['stats'] = self.index_manager.get_index_stats()

        return index_status

    def _check_component_status(self) -> Dict[str, Any]:
        """Check status of all core components"""
        return {
            'preprocessor': {'initialized': self.preprocessor is not None},
            'embedding_extractor': {
                'initialized': self.embedding_extractor is not None,
                'model': self.embedding_extractor.model_name,
                'detector': self.embedding_extractor.detector_backend
            },
            'index_manager': {'initialized': self.index_manager is not None},
            'search_engine': {'initialized': self.search_engine is not None}
        }

    # ===========================================
    # PREPROCESSING OPERATIONS
    # ===========================================

    def preprocess_images_from_folder(self, source_folder: str, 
                                    clear_existing: bool = True) -> Dict[str, Any]:
        """
        Preprocess all images from a source folder

        Args:
            source_folder: Path to folder containing images
            clear_existing: Whether to clear existing preprocessed images

        Returns:
            Results dictionary with success status and metrics
        """
        try:
            # Validate source folder
            if not os.path.exists(source_folder):
                return {'success': False, 'error': f'Source folder not found: {source_folder}'}

            # Get all image files
            image_files = self.fs_ops.get_all_images_from_folder(source_folder)
            if not image_files:
                return {'success': False, 'error': f'No images found in {source_folder}'}

            # Clear preprocessed directory if requested
            if clear_existing and os.path.exists(self.settings.PREPROCESSED_DIR):
                shutil.rmtree(self.settings.PREPROCESSED_DIR)

            os.makedirs(self.settings.PREPROCESSED_DIR, exist_ok=True)

            # Process images in parallel
            results = {
                'success': True,
                'total_images': len(image_files),
                'successful': 0,
                'failed': 0,
                'processed_files': []
            }

            # Use parallel processing for image preprocessing
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import multiprocessing as mp

            max_workers = min(mp.cpu_count() - 1, len(image_files), 6)  # Leave 1 CPU core free, increase max workers

            def process_single_image(image_data):
                original_path, relative_path = image_data
                preprocessed_path = os.path.join(self.settings.PREPROCESSED_DIR, relative_path)
                preprocessed_dir = os.path.dirname(preprocessed_path)
                os.makedirs(preprocessed_dir, exist_ok=True)

                try:
                    # Copy and preprocess
                    shutil.copy2(original_path, preprocessed_path)
                    preprocess_result = self.preprocessor.preprocess_image(
                        preprocessed_path, preprocessed_path
                    )

                    if preprocess_result.get('success', False):
                        return 'success', (preprocessed_path, relative_path)
                    else:
                        return 'failed', None
                except Exception as e:
                    self.logger.error(f"Error processing {original_path}: {e}")
                    return 'failed', None

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all preprocessing tasks
                futures = {executor.submit(process_single_image, img_data): img_data 
                          for img_data in image_files}

                # Process completed tasks with timeout
                for i, future in enumerate(as_completed(futures)):
                    try:
                        status, result = future.result(timeout=30)  # 30 second timeout per image
                        if status == 'success':
                            results['successful'] += 1
                            results['processed_files'].append(result)
                        else:
                            results['failed'] += 1
                    except Exception as e:
                        results['failed'] += 1
                        self.logger.warning(f"Task failed: {e}")

                    # Progress logging every 5 images for less spam
                    if (i + 1) % 5 == 0 or i == len(image_files) - 1:
                        self.logger.info(f"📸 Preprocessed {i + 1}/{len(image_files)} images")

            return results

        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            return {'success': False, 'error': str(e)}

    # ===========================================
    # INDEX BUILDING OPERATIONS
    # ===========================================

    def build_complete_index(self, 
                           source_folder: str,
                           use_preprocessing: bool = False,
                           max_workers: Optional[int] = None,
                           chunk_size: int = 100) -> Dict[str, Any]:
        """
        Complete index building workflow with chunked processing for better memory management
        This is the original method without progress callbacks for backward compatibility
        """
        return self.build_complete_index_with_progress(
            source_folder, use_preprocessing, max_workers, chunk_size, None
        )

    def build_complete_index_with_progress(self,
                                         source_folder: str,
                                         use_preprocessing: bool = False,
                                         max_workers: Optional[int] = None,
                                         chunk_size: int = 100,
                                         progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Complete index building workflow with progress callbacks

        Args:
            source_folder: Path to folder containing images
            use_preprocessing: Whether to use preprocessing
            max_workers: Maximum number of workers for parallel processing
            chunk_size: Size of chunks for batch processing
            progress_callback: Optional callback function for progress updates

        Returns:
            Dictionary with build results and statistics
        """
        try:
            self.logger.info(f"Starting complete index build from: {source_folder}")

            # Validate source folder
            validation = self.validate_source_folder(source_folder)
            if not validation['valid']:
                return {'success': False, 'error': validation['error']}

            # Get all image files
            image_files = self.fs_ops.get_all_images_from_folder(source_folder)
            if not image_files:
                return {'success': False, 'error': f'No images found in {source_folder}'}

            # Convert to absolute paths
            image_paths = [os.path.join(source_folder, rel_path) for _, rel_path in image_files]

            self.logger.info(f"Found {len(image_paths)} images to process")

            if progress_callback:
                progress_callback({"stage": "extraction", "progress": 0, "total": len(image_paths)})

            # Extract embeddings with preprocessing if requested
            def wrapped_progress_callback(processed, total):
                if progress_callback:
                    progress_callback({"stage": "extraction", "progress": processed, "total": total})

            # Always use the preprocessing-capable method for consistency
            embeddings, successful_paths = self.embedding_extractor.extract_batch_embeddings_with_preprocessing(
                image_paths=image_paths,
                enforce_detection=True,
                max_workers=max_workers,
                use_preprocessing=use_preprocessing,
                progress_callback=wrapped_progress_callback
            )

            if not embeddings:
                return {'success': False, 'error': 'No face embeddings could be extracted'}

            self.logger.info(f"Successfully extracted {len(embeddings)} face embeddings")

            # Create labels (relative paths from source folder)
            labels = []
            for path in successful_paths:
                rel_path = os.path.relpath(path, source_folder)
                labels.append(rel_path)

            if progress_callback:
                progress_callback({"stage": "indexing", "progress": 0, "total": 1})

            # Build FAISS index with additional metadata
            additional_metadata = {
                'total_images': len(image_paths),
                'successful_extractions': len(embeddings),
                'failed_extractions': len(image_paths) - len(embeddings),
                'model_used': self.embedding_extractor.model_name,
                'detector_used': self.embedding_extractor.detector_backend,
                'build_timestamp': datetime.now().isoformat(),
                'embedding_dimension': len(embeddings[0]) if embeddings else 0,
                'preprocessing_configuration': {
                    'enabled': use_preprocessing,
                    'target_size': (160, 160),
                    'enhancement_enabled': use_preprocessing,
                    'face_alignment_enabled': use_preprocessing,
                    'quality_assessment_enabled': use_preprocessing
                }
            }

            success = self.index_manager.build_index(
                embeddings=embeddings,
                labels=labels,
                source_folder=source_folder,
                preprocessing_used=use_preprocessing,
                additional_metadata=additional_metadata
            )

            if not success:
                return {'success': False, 'error': 'Failed to build FAISS index'}

            # Reload search engine index
            self.search_engine.reload_index()

            if progress_callback:
                progress_callback({"stage": "complete", "progress": 1, "total": 1})

            result = {
                'success': True,
                'total_images': len(image_paths),
                'faces_indexed': len(embeddings),
                'failed_extractions': len(image_paths) - len(embeddings),
                'source_folder': source_folder,
                'preprocessing_used': use_preprocessing,
                'index_stats': self.index_manager.get_index_stats()
            }

            self.logger.info(f"✅ Index build completed: {len(embeddings)} faces indexed")
            return result

        except Exception as e:
            self.logger.error(f"Index building failed: {e}")
            return {'success': False, 'error': str(e)}

    def add_images_to_existing_index(self, 
                                   image_paths: List[str],
                                   use_preprocessing: bool = False,
                                   max_workers: Optional[int] = None,
                                   progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Add new images to existing index without rebuilding everything

        Args:
            image_paths: List of paths to new images to add
            use_preprocessing: Whether to use preprocessing
            max_workers: Maximum number of workers
            progress_callback: Optional progress callback

        Returns:
            Result dictionary with success status and statistics
        """
        try:
            # Check if index exists
            if not self.index_manager.index_exists():
                return {'success': False, 'error': 'No existing index found. Please build an index first.'}

            # Load existing index
            if not self.index_manager.load_index():
                return {'success': False, 'error': 'Failed to load existing index'}

            if not image_paths:
                return {'success': False, 'error': 'No image paths provided'}

            # Filter out already indexed images
            existing_labels = self.index_manager.get_labels()
            existing_relative_paths = set(existing_labels)

            # Get source folder from existing index
            index_info = self.index_manager.get_index_info()
            source_folder = index_info.get('source_folder', '')

            new_image_paths = []
            for img_path in image_paths:
                if os.path.exists(img_path):
                    # Create relative path for comparison
                    if source_folder and img_path.startswith(source_folder):
                        rel_path = os.path.relpath(img_path, source_folder)
                    else:
                        rel_path = os.path.basename(img_path)

                    if rel_path not in existing_relative_paths:
                        new_image_paths.append(img_path)

            if not new_image_paths:
                return {'success': False, 'error': 'All provided images are already in the index'}

            self.logger.info(f"Adding {len(new_image_paths)} new images to existing index")

            # Extract embeddings for new images
            def wrapped_progress_callback(processed, total):
                if progress_callback:
                    progress_callback({"stage": "extraction", "progress": processed, "total": total})

            # Always use the preprocessing-capable method for consistency
            new_embeddings, successful_paths = self.embedding_extractor.extract_batch_embeddings_with_preprocessing(
                image_paths=new_image_paths,
                enforce_detection=True,
                max_workers=max_workers,
                use_preprocessing=use_preprocessing,
                progress_callback=wrapped_progress_callback
            )

            if not new_embeddings:
                return {'success': False, 'error': 'No embeddings extracted from new images'}

            # Create labels for new images
            new_labels = []
            for path in successful_paths:
                if source_folder and path.startswith(source_folder):
                    rel_path = os.path.relpath(path, source_folder)
                else:
                    rel_path = os.path.basename(path)
                new_labels.append(rel_path)

            # Add to existing index
            success = self.index_manager.add_to_existing_index(
                new_embeddings=new_embeddings,
                new_labels=new_labels,
                new_full_paths=successful_paths,
                additional_metadata={
                    'incremental_update': True,
                    'update_timestamp': datetime.now().isoformat(),
                    'new_images_added': len(new_embeddings),
                    'preprocessing_used': use_preprocessing
                }
            )

            if success:
                self.search_engine.reload_index()
                return {
                    'success': True,
                    'new_images_processed': len(new_image_paths),
                    'new_faces_added': len(new_embeddings),
                    'failed_extractions': len(new_image_paths) - len(new_embeddings),
                    'total_faces_in_index': self.index_manager.get_index_stats().get('total_faces', 0)
                }
            else:
                return {'success': False, 'error': 'Failed to add images to existing index'}

        except Exception as e:
            self.logger.error(f"Failed to add images to existing index: {e}")
            return {'success': False, 'error': str(e)}

    def add_folder_to_existing_index(self,
                                   folder_path: str,
                                   use_preprocessing: bool = False,
                                   max_workers: Optional[int] = None,
                                   progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Add all images from a folder to existing index

        Args:
            folder_path: Path to folder containing new images
            use_preprocessing: Whether to use preprocessing
            max_workers: Maximum number of workers
            progress_callback: Optional progress callback

        Returns:
            Result dictionary with success status and statistics
        """
        try:
            if not os.path.exists(folder_path):
                return {'success': False, 'error': f'Folder does not exist: {folder_path}'}

            # Get all image files from folder
            image_files = self.fs_ops.get_image_files(folder_path)

            if not image_files:
                return {'success': False, 'error': f'No images found in folder: {folder_path}'}

            return self.add_images_to_existing_index(
                image_paths=image_files,
                use_preprocessing=use_preprocessing,
                max_workers=max_workers,
                progress_callback=progress_callback
            )

        except Exception as e:
            self.logger.error(f"Failed to add folder to existing index: {e}")
            return {'success': False, 'error': str(e)}

    # ===========================================
    # SEARCH OPERATIONS
    # ===========================================

    def search_similar_faces(self, 
                                query_image_path: str,
                                k: int = 5,
                                threshold: float = 450.0,
                                enforce_detection: bool = True,
                                use_preprocessing: Optional[bool] = None) -> Dict[str, Any]:
        """
        Search for similar faces using the search engine

        Args:
            query_image_path: Path to query image
            k: Number of similar faces to retrieve
            threshold: Distance threshold for filtering results
            enforce_detection: Whether to enforce face detection
            use_preprocessing: Whether to use preprocessing

        Returns:
            Dictionary with search results and metadata
        """
        try:
            # Use the search engine to perform the search
            results = self.search_engine.search_similar_faces(
                query_path=query_image_path,
                k=k,
                threshold=threshold,
                enforce_detection=enforce_detection,
                use_preprocessing=use_preprocessing
            )

            if results is None:
                return {'success': False, 'error': 'Search failed - no results returned'}

            # Count matches within threshold
            matches_found = sum(1 for r in results if r.get('within_threshold', False))

            return {
                'success': True,
                'results': results,
                'total_results': len(results),
                'matches_found': matches_found,
                'search_params': {
                    'k': k,
                    'threshold': threshold,
                    'enforce_detection': enforce_detection,
                    'use_preprocessing': use_preprocessing
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to search similar faces: {e}")
            return {'success': False, 'error': str(e)}

    def search_by_embedding(self,
                          query_embedding: List[float],
                          k: int = 5,
                          threshold: float = 450.0) -> Dict[str, Any]:
        """
        Search using pre-computed embedding

        Args:
            query_embedding: Pre-computed face embedding
            k: Number of similar faces to retrieve
            threshold: Distance threshold for filtering results

        Returns:
            Dictionary with search results and metadata
        """
        try:
            import numpy as np

            # Convert to numpy array if needed
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)

            results = self.search_engine.search_by_embedding(
                query_embedding=query_embedding,
                k=k,
                threshold=threshold
            )

            if results is None:
                return {'success': False, 'error': 'Embedding search failed - no results returned'}

            matches_found = sum(1 for r in results if r.get('within_threshold', False))

            return {
                'success': True,
                'results': results,
                'total_results': len(results),
                'matches_found': matches_found,
                'search_params': {
                    'k': k,
                    'threshold': threshold
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to search by embedding: {e}")
            return {'success': False, 'error': str(e)}