import os
import json
import numpy as np
import faiss
import logging
import tempfile
import shutil
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import sys
import os
# Add the src directory to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, src_path)

from core.embedding.extractor import EmbeddingExtractor
from config.settings import Settings
from utils.filesystem.operations import FileSystemOperations

class SearchEngine:
    """
    FAISS-based similarity search engine for face embeddings.
    Handles searching for similar faces and result processing.
    """

    def __init__(self, embeddings_dir: str = None):
        self.settings = Settings()
        self.embeddings_dir = embeddings_dir or self.settings.EMBED_DIR
        self.logger = logging.getLogger(__name__)
        self.fs_ops = FileSystemOperations()

        # Initialize embedding extractor
        self.embedding_extractor = EmbeddingExtractor()

        # FAISS index and metadata
        self.faiss_index = None
        self.labels = []
        self.index_info = {}

        # Try to load existing index
        self.load_index()

    def load_index(self) -> bool:
        """Load FAISS index and metadata from disk"""
        try:
            index_path = os.path.join(self.embeddings_dir, "faiss_index.bin")
            metadata_path = os.path.join(self.embeddings_dir, "labels.json")

            if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                self.logger.warning("Index files not found")
                return False

            # Load FAISS index
            self.faiss_index = faiss.read_index(index_path)

            # Load metadata
            with open(metadata_path, "r") as f:
                self.index_info = json.load(f)

            # Handle both old and new metadata formats
            if isinstance(self.index_info, list):
                self.labels = self.index_info
                self.index_info = {"labels": self.labels}
            else:
                self.labels = self.index_info.get("labels", [])

            self.logger.info(f"✅ Search engine loaded index from {self.embeddings_dir}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if search engine is ready for queries"""
        return self.faiss_index is not None and len(self.labels) > 0

    def search_similar_faces(self, 
                           query_path: str,
                           k: int = 5,
                           threshold: float = 450.0,
                           enforce_detection: bool = True,
                           use_preprocessing: Optional[bool] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Search for similar faces using FAISS index with consistent preprocessing

        Args:
            query_path: Path to query image
            k: Number of similar faces to retrieve
            threshold: Distance threshold for filtering results
            enforce_detection: Whether to enforce face detection
            use_preprocessing: Whether to use preprocessing (auto-detected from index if None)

        Returns:
            List of search results with metadata
        """
        try:
            if not self.is_ready():
                self.logger.error("Search engine not ready - no index loaded")
                return None

            if not os.path.exists(query_path):
                self.logger.error(f"Query image not found: {query_path}")
                return None

            # Determine if preprocessing should be used
            if use_preprocessing is None:
                # Auto-detect from index metadata
                use_preprocessing = self.index_info.get('preprocessing_used', False)

            self.logger.info(f"🔍 Processing query image: {os.path.basename(query_path)} (preprocessing: {use_preprocessing})")

            # Apply preprocessing if needed
            processed_query_path = query_path
            temp_file_created = False

            if use_preprocessing:
                try:
                    import tempfile
                    import shutil
                    from core.preprocessing.preprocessor import ImagePreprocessor

                    # Create temporary file for preprocessing
                    temp_dir = tempfile.mkdtemp(prefix="temp_search_preprocess_")
                    temp_filename = f"query_{os.path.basename(query_path)}"
                    temp_path = os.path.join(temp_dir, temp_filename)

                    # Copy original to temp location
                    shutil.copy2(query_path, temp_path)

                    # Apply preprocessing
                    preprocessor = ImagePreprocessor()
                    result = preprocessor.preprocess_image(temp_path, temp_path)

                    if result.get('success', False):
                        processed_query_path = temp_path
                        temp_file_created = True
                        self.logger.info(f"✅ Query image preprocessed successfully")
                    else:
                        self.logger.warning(f"⚠️ Preprocessing failed for query image, using original")
                        # Clean up failed temp file
                        shutil.rmtree(temp_dir)

                except Exception as e:
                    self.logger.error(f"❌ Preprocessing error for query image: {e}")
                    self.logger.info("🔄 Falling back to original query image")

            # Extract embedding for query image (from processed or original)
            query_embedding = self.embedding_extractor.extract_single_embedding(
                image_path=processed_query_path,
                enforce_detection=enforce_detection
            )

            # Clean up temporary preprocessed file
            if temp_file_created and os.path.exists(processed_query_path):
                try:
                    shutil.rmtree(os.path.dirname(processed_query_path))
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temp file: {e}")

            if query_embedding is None:
                self.logger.error("Failed to extract embedding from query image")
                return None

            # Perform FAISS search
            query_embedding = np.array([query_embedding]).astype("float32")
            distances, indices = self.faiss_index.search(query_embedding, k)

            # Store results with proper path information
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.labels):
                    label = self.labels[idx]

                    # Resolve the correct path for the image
                    resolved_path = self._resolve_image_path(label)

                    result = {
                        'rank': i + 1,
                        'distance': float(distance),
                        'within_threshold': distance <= threshold,
                        'filename': os.path.basename(label),
                        'relative_path': label,
                        'path': resolved_path,
                        'exists': os.path.exists(resolved_path) if resolved_path else False
                    }

                    results.append(result)

            self.logger.info(f"✅ Search completed: {len(results)} results returned")

            # Log results summary
            within_threshold = [r for r in results if r['within_threshold']]
            self.logger.info(f"📊 Matches within threshold: {len(within_threshold)}/{len(results)}")

            return results

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return None

    def _resolve_image_path(self, label: str) -> Optional[str]:
        """
        Resolve the correct full path for an image based on its label.
        Handles both original source images and incrementally added images.

        Args:
            label: The image label/relative path from the index

        Returns:
            Full path to the image file, or None if not found
        """
        try:
            # Check if we have full path mappings for incrementally added images
            full_path_mappings = self.index_info.get('full_path_mappings', {})

            # If this image has a full path mapping (was added incrementally), use it
            if label in full_path_mappings:
                mapped_path = full_path_mappings[label]
                if os.path.exists(mapped_path):
                    return mapped_path
                else:
                    self.logger.warning(f"Mapped path does not exist: {mapped_path}")

            # Try to resolve from original source folder
            source_folder = self.index_info.get('source_folder')
            if source_folder:
                source_path = os.path.join(source_folder, label)
                if os.path.exists(source_path):
                    return source_path

            # Try relative to current working directory
            if os.path.exists(label):
                return os.path.abspath(label)

            # Try in upload directory
            upload_path = os.path.join(self.settings.UPLOAD_DIR, label)
            if os.path.exists(upload_path):
                return upload_path

            # Try in preprocessed directory
            preprocessed_path = os.path.join(self.settings.PREPROCESSED_DIR, label)
            if os.path.exists(preprocessed_path):
                return preprocessed_path

            # If all else fails, try to find the file by searching common directories
            possible_dirs = [
                self.settings.UPLOAD_DIR,
                self.settings.PREPROCESSED_DIR,
                os.getcwd(),
            ]

            filename = os.path.basename(label)
            for directory in possible_dirs:
                if os.path.exists(directory):
                    for root, dirs, files in os.walk(directory):
                        if filename in files:
                            found_path = os.path.join(root, filename)
                            if os.path.exists(found_path):
                                return found_path

            self.logger.warning(f"Could not resolve path for label: {label}")
            return None

        except Exception as e:
            self.logger.error(f"Error resolving path for label {label}: {e}")
            return None

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        try:
            if not self.is_ready():
                return {"error": "Search engine not ready"}

            stats = {
                "total_indexed_faces": len(self.labels),
                "embedding_dimension": self.faiss_index.d,
                "index_ready": True,
                "last_loaded": datetime.now().isoformat()
            }

            # Add index metadata if available
            if self.index_info:
                stats.update({
                    "source_folder": self.index_info.get("source_folder", "Unknown"),
                    "created_at": self.index_info.get("created_at", "Unknown"),
                    "model_used": self.index_info.get("model_used", "Unknown"),
                    "preprocessing_used": self.index_info.get("preprocessing_used", False),
                    "preprocessing_enabled_in_search": self.index_info.get("preprocessing_used", False),
                    "has_incremental_additions": bool(self.index_info.get('full_path_mappings', {}))
                })

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get search statistics: {e}")
            return {"error": str(e)}

    def reload_index(self) -> bool:
        """Reload the index from disk"""
        self.faiss_index = None
        self.labels = []
        self.index_info = {}
        return self.load_index()

    def get_indexed_image_paths(self) -> List[str]:
        """Get all indexed image paths with proper resolution"""
        resolved_paths = []
        for label in self.labels:
            resolved_path = self._resolve_image_path(label)
            if resolved_path:
                resolved_paths.append(resolved_path)
            else:
                resolved_paths.append(label)  # Fallback to original label
        return resolved_paths

    def validate_index_integrity(self) -> Dict[str, Any]:
        """Validate the integrity of the loaded index"""
        try:
            if not self.is_ready():
                return {"valid": False, "error": "Index not loaded"}

            total_labels = len(self.labels)
            total_embeddings = self.faiss_index.ntotal

            # Check if labels and embeddings count match
            counts_match = total_labels == total_embeddings

            # Check how many image paths can be resolved
            resolved_count = 0
            missing_files = []

            for label in self.labels:
                resolved_path = self._resolve_image_path(label)
                if resolved_path and os.path.exists(resolved_path):
                    resolved_count += 1
                else:
                    missing_files.append(label)

            return {
                "valid": counts_match,
                "total_labels": total_labels,
                "total_embeddings": total_embeddings,
                "counts_match": counts_match,
                "resolved_files": resolved_count,
                "missing_files": len(missing_files),
                "missing_file_list": missing_files[:10],  # First 10 missing files
                "resolution_rate": (resolved_count / total_labels) * 100 if total_labels > 0 else 0
            }

        except Exception as e:
            self.logger.error(f"Failed to validate index integrity: {e}")
            return {"valid": False, "error": str(e)}