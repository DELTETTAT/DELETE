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
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
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
        Complete index building workflow with simplified processing

        Args:
            source_folder: Source folder path
            use_preprocessing: Whether to use preprocessing
            max_workers: Maximum number of workers for parallel processing
            chunk_size: Not used anymore (kept for compatibility)
            progress_callback: Optional callback for progress updates

        Returns:
            Result dictionary with success status and statistics
        """
        try:
            # Validate source folder
            validation_result = self.validate_source_folder(source_folder)
            if not validation_result['valid']:
                return {'success': False, 'error': validation_result['error']}

            # Get image files
            image_files = self.fs_ops.get_image_files(source_folder)

            if not image_files:
                return {'success': False, 'error': 'No valid image files found'}

            self.logger.info(f"Building index for {len(image_files)} images")

            # Show which images are being processed (first few)
            for i, img_path in enumerate(image_files[:5]):  # Show first 5 images
                img_name = os.path.basename(img_path)
                processing_type = "with preprocessing" if use_preprocessing else "original"
                self.logger.info(f"📸 Processing {processing_type}: {img_name}")
            if len(image_files) > 5:
                self.logger.info(f"📋 ...and {len(image_files) - 5} more images")

            # Extract embeddings from all images in one batch
            self.logger.info("🧠 Extracting embeddings from all images...")
            embeddings, successful_paths = self.embedding_extractor.extract_batch_embeddings_with_preprocessing(
                image_paths=image_files,
                enforce_detection=True,
                max_workers=max_workers,
                use_preprocessing=use_preprocessing,
                progress_callback=progress_callback
            )

            # Show embedding results
            success_count = len(embeddings)
            failed_count = len(image_files) - success_count
            self.logger.info(f"✅ Embeddings extracted: {success_count}/{len(image_files)} successful")
            if failed_count > 0:
                self.logger.warning(f"⚠️ Failed embeddings: {failed_count} images")

            if not embeddings:
                return {'success': False, 'error': 'No embeddings extracted'}

            # Create labels for successful extractions
            labels = [os.path.relpath(path, source_folder) for path in successful_paths]

            # Build final index
            self.logger.info("📊 Building final FAISS index...")
            success = self.index_manager.build_index(
                embeddings=embeddings,
                labels=labels,
                source_folder=source_folder,
                preprocessing_used=use_preprocessing,
                additional_metadata={
                    'total_images': len(image_files),
                    'failed_embeddings': failed_count,
                    'build_timestamp': datetime.now().isoformat(),
                    'simplified_processing': True
                }
            )

            if success:
                self.search_engine.reload_index()
                return {
                    'success': True,
                    'total_images': len(image_files),
                    'indexed_faces': success_count,
                    'failed_extractions': failed_count,
                    'preprocessing_used': use_preprocessing
                }
            else:
                return {'success': False, 'error': 'Index building failed'}

        except Exception as e:
            self.logger.error(f"Complete index building failed: {e}")
            return {'success': False, 'error': str(e)}

    def build_index_from_existing(self, source_folder: str, 
                                use_preprocessed: bool = True) -> Dict[str, Any]:
        """
        Build index from existing images (original or preprocessed)

        Args:
            source_folder: Path to folder containing original images
            use_preprocessed: Whether to use preprocessed images if available

        Returns:
            Results dictionary with success status and metrics
        """
        try:
            # Get image files
            image_files = self.fs_ops.get_all_images_from_folder(source_folder)
            if not image_files:
                return {'success': False, 'error': f'No images found in {source_folder}'}

            # Determine which files to use
            if use_preprocessed:
                processed_files = []
                for original_path, relative_path in image_files:
                    preprocessed_path = os.path.join(self.settings.PREPROCESSED_DIR, relative_path)
                    if os.path.exists(preprocessed_path):
                        processed_files.append((preprocessed_path, relative_path))
                    else:
                        processed_files.append((original_path, relative_path))
            else:
                processed_files = image_files

            # Extract embeddings
            image_paths = [path for path, _ in processed_files]
            embeddings, successful_paths = self.embedding_extractor.extract_batch_embeddings(
                image_paths=image_paths,
                enforce_detection=True
            )

            if not embeddings:
                return {'success': False, 'error': 'No embeddings extracted'}

            # Build index
            path_to_relative = {path: rel_path for path, rel_path in processed_files}
            labels = [path_to_relative[path] for path in successful_paths if path in path_to_relative]

            success = self.index_manager.build_index(
                embeddings=embeddings,
                labels=labels,
                source_folder=source_folder,
                preprocessing_used=use_preprocessed,
                additional_metadata={
                    'total_images': len(image_files),
                    'failed_embeddings': len(image_files) - len(embeddings),
                    'build_timestamp': datetime.now().isoformat()
                }
            )

            if success:
                self.search_engine.reload_index()
                return {
                    'success': True,
                    'total_images': len(image_files),
                    'indexed_faces': len(embeddings),
                    'failed_extractions': len(image_files) - len(embeddings),
                    'preprocessing_used': use_preprocessed
                }
            else:
                return {'success': False, 'error': 'Index building failed'}

        except Exception as e:
            self.logger.error(f"Index building from existing failed: {e}")
            return {'success': False, 'error': str(e)}

    # ===========================================
    # SEARCH OPERATIONS
    # ===========================================

    def search_similar_faces(self, query_image_path: str, 
                           k: int = 5, threshold: float = 450.0) -> Dict[str, Any]:
        """
        Search for similar faces

        Args:
            query_image_path: Path to query image
            k: Number of results to return
            threshold: Distance threshold for filtering

        Returns:
            Search results dictionary
        """
        try:
            if not self.search_engine.is_ready():
                return {'success': False, 'error': 'Search index not ready'}

            # Check if dataset has changed
            index_info = self.index_manager.get_index_info()
            if index_info:
                source_folder = index_info.get('source_folder')
                stored_hash = index_info.get('dataset_hash')

                if source_folder and stored_hash:
                    from core.embedding.cache import EmbeddingCache
                    cache = EmbeddingCache()

                    if cache.is_dataset_changed(source_folder, stored_hash):
                        return {
                            'success': False, 
                            'error': 'Dataset has changed since last index build. Please rebuild the index.',
                            'dataset_changed': True,
                            'source_folder': source_folder
                        }

            if not os.path.exists(query_image_path):
                return {'success': False, 'error': f'Query image not found: {query_image_path}'}

            results = self.search_engine.search_similar_faces(
                query_path=query_image_path,
                k=k,
                threshold=threshold
            )

            if results is None:
                return {'success': False, 'error': 'Search failed'}

            # Process results
            matches = [r for r in results if r['within_threshold']]

            return {
                'success': True,
                'query_image': query_image_path,
                'total_results': len(results),
                'matches_found': len(matches),
                'results': results,
                'matches': matches,
                'search_params': {'k': k, 'threshold': threshold}
            }

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return {'success': False, 'error': str(e)}

    def get_available_query_images(self) -> List[str]:
        """Get list of available images for querying"""
        return self.fs_ops.get_image_files_in_folder(self.settings.UPLOAD_DIR)

    # ===========================================
    # INDEX MANAGEMENT
    # ===========================================

    def delete_index(self) -> Dict[str, Any]:
        """Delete the current index"""
        try:
            success = self.index_manager.delete_index()
            if success:
                self.search_engine.reload_index()
                return {'success': True, 'message': 'Index deleted successfully'}
            else:
                return {'success': False, 'error': 'Failed to delete index'}
        except Exception as e:
            self.logger.error(f"Index deletion failed: {e}")
            return {'success': False, 'error': str(e)}

    def get_index_stats(self) -> Dict[str, Any]:
        """Get comprehensive index statistics"""
        try:
            stats = self.index_manager.get_index_stats()
            if 'error' not in stats:
                # Add search engine stats
                search_stats = self.search_engine.get_search_stats()
                stats.update(search_stats)
            return stats
        except Exception as e:
            self.logger.error(f"Failed to get index statistics: {e}")
            return {'error': str(e)}

    # ===========================================
    # UTILITY METHODS
    # ===========================================

    def validate_source_folder(self, folder_path: str) -> Dict[str, Any]:
        """Validate a source folder for processing"""
        try:
            if not os.path.exists(folder_path):
                return {'valid': False, 'error': 'Folder does not exist'}

            if not os.path.isdir(folder_path):
                return {'valid': False, 'error': 'Path is not a directory'}

            image_count = self.fs_ops.count_images_in_folder(folder_path)
            if image_count == 0:
                return {'valid': False, 'error': 'No images found in folder'}

            return {
                'valid': True,
                'image_count': image_count,
                'path': folder_path
            }

        except Exception as e:
            return {'valid': False, 'error': str(e)}

    def cleanup_temp_files(self) -> Dict[str, Any]:
        """Clean up temporary processing files"""
        try:
            cleanup_results = {
                'temp_processing_cleaned': False,
                'temp_dirs_cleaned': 0
            }

            # Clean temp_processing directory
            temp_processing = os.path.join(self.settings.BASE_DIR, 'temp_processing')
            if os.path.exists(temp_processing):
                shutil.rmtree(temp_processing)
                cleanup_results['temp_processing_cleaned'] = True

            # Add other cleanup operations as needed

            return {'success': True, 'cleanup_results': cleanup_results}

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return {'success': False, 'error': str(e)}