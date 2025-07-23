import os
import json
import numpy as np
import faiss
import logging
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

class IndexManager:
    """
    FAISS index management for face embeddings.
    Handles index creation, loading, saving, and metadata management.
    """

    def __init__(self, embeddings_dir: str = None):
        self.settings = Settings()
        self.embeddings_dir = embeddings_dir or self.settings.EMBED_DIR
        self.logger = logging.getLogger(__name__)
        self.fs_ops = FileSystemOperations()

        # Ensure embeddings directory exists
        os.makedirs(self.embeddings_dir, exist_ok=True)

        # FAISS index and metadata
        self.faiss_index = None
        self.labels = []
        self.index_info = {}

    def build_index(self, 
                    embeddings: List[np.ndarray], 
                    labels: List[str],
                    source_folder: str,
                    preprocessing_used: bool = False,
                    additional_metadata: Optional[Dict] = None) -> bool:
        """
        Build FAISS index from embeddings

        Args:
            embeddings: List of face embeddings
            labels: Corresponding labels/paths for embeddings
            source_folder: Source folder path
            preprocessing_used: Whether preprocessing was applied
            additional_metadata: Additional metadata to store

        Returns:
            True if successful, False otherwise
        """
        try:
            if not embeddings or not labels:
                self.logger.error("Empty embeddings or labels provided")
                return False

            if len(embeddings) != len(labels):
                self.logger.error("Embeddings and labels length mismatch")
                return False

            # Convert embeddings to numpy array
            np_embeddings = np.array(embeddings).astype("float32")

            # Create FAISS index
            dimension = np_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(np_embeddings)

            # Store labels
            self.labels = labels

            # Get dataset hash for change detection
            from core.embedding.cache import EmbeddingCache
            cache = EmbeddingCache()
            dataset_hash = cache.get_dataset_hash(source_folder)

            # Create index metadata
            self.index_info = {
                "labels": labels,
                "source_folder": source_folder,
                "created_at": datetime.now().isoformat(),
                "preprocessing_used": preprocessing_used,
                "total_images": len(labels),
                "successful_embeddings": len(embeddings),
                "embedding_dimension": dimension,
                "model_used": getattr(EmbeddingExtractor(), 'model_name', 'Unknown'),
                "detector_used": getattr(EmbeddingExtractor(), 'detector_backend', 'Unknown'),
                "dataset_hash": dataset_hash
            }

            # Add additional metadata if provided
            if additional_metadata:
                self.index_info.update(additional_metadata)

            # Save index and metadata
            return self.save_index()

        except Exception as e:
            self.logger.error(f"Failed to build index: {e}")
            return False

    def save_index(self) -> bool:
        """Save FAISS index and metadata to disk"""
        try:
            if self.faiss_index is None:
                self.logger.error("No index to save")
                return False

            # Save FAISS index
            index_path = os.path.join(self.embeddings_dir, "faiss_index.bin")
            faiss.write_index(self.faiss_index, index_path)

            # Save embeddings as numpy array
            embeddings_path = os.path.join(self.embeddings_dir, "embeddings.npy")
            embeddings = []
            for i in range(self.faiss_index.ntotal):
                embedding = self.faiss_index.reconstruct(i)
                embeddings.append(embedding)
            np.save(embeddings_path, np.array(embeddings))

            # Save metadata
            metadata_path = os.path.join(self.embeddings_dir, "labels.json")
            with open(metadata_path, "w") as f:
                json.dump(self.index_info, f, indent=2)

            self.logger.info(f"✅ Index saved to {self.embeddings_dir}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
            return False

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

            self.logger.info(f"✅ Index loaded from {self.embeddings_dir}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            return False

    def get_index_stats(self) -> Dict:
        """Get comprehensive index statistics"""
        try:
            if not self.index_exists():
                return {"error": "No index found"}

            if not self.faiss_index:
                self.load_index()

            stats = {
                "total_faces": len(self.labels),
                "embedding_dimension": self.faiss_index.d if self.faiss_index else 0,
                "index_size": self.faiss_index.ntotal if self.faiss_index else 0,
                **self.index_info
            }

            # Add file system stats
            index_path = os.path.join(self.embeddings_dir, "faiss_index.bin")
            if os.path.exists(index_path):
                stats["index_file_size"] = os.path.getsize(index_path)

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get index stats: {e}")
            return {"error": str(e)}

    def get_statistics(self) -> Dict[str, Any]:
        """Alias for get_index_stats for compatibility"""
        return self.get_index_stats()

    def index_exists(self) -> bool:
        """Check if index files exist"""
        index_path = os.path.join(self.embeddings_dir, "faiss_index.bin")
        metadata_path = os.path.join(self.embeddings_dir, "labels.json")
        embeddings_path = os.path.join(self.embeddings_dir, "embeddings.npy")

        return all(os.path.exists(path) for path in [index_path, metadata_path, embeddings_path])

    def delete_index(self) -> bool:
        """Delete index files"""
        try:
            files_to_delete = [
                os.path.join(self.embeddings_dir, "faiss_index.bin"),
                os.path.join(self.embeddings_dir, "labels.json"),
                os.path.join(self.embeddings_dir, "embeddings.npy")
            ]

            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)

            self.faiss_index = None
            self.labels = []
            self.index_info = {}

            self.logger.info("✅ Index files deleted")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete index: {e}")
            return False

    def get_index_info(self) -> Dict:
        """Get index metadata information"""
        return self.index_info.copy()

    def get_labels(self) -> List[str]:
        """Get all labels from the index"""
        return self.labels.copy()