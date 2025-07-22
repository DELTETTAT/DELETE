import os
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
import sys
import os
# Add the src directory to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, src_path)

from core.embedding.extractor import EmbeddingExtractor
from core.indexing.manager import IndexManager
from config.settings import Settings

class SearchEngine:
    """
    Face similarity search engine using FAISS index.
    Provides similarity search with configurable parameters.
    """
    
    def __init__(self, embeddings_dir: str = None):
        self.settings = Settings()
        self.embeddings_dir = embeddings_dir or self.settings.EMBED_DIR
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.embedding_extractor = EmbeddingExtractor()
        self.index_manager = IndexManager(self.embeddings_dir)
        
        # Load index on initialization
        self._load_index()
    
    def _load_index(self) -> bool:
        """Load the FAISS index"""
        try:
            success = self.index_manager.load_index()
            if success:
                self.logger.info("✅ Search index loaded successfully")
            else:
                self.logger.warning("⚠️ No search index found")
            return success
        except Exception as e:
            self.logger.error(f"Failed to load search index: {e}")
            return False
    
    def search_similar_faces(self, 
                             query_path: str,
                             k: int = 5,
                             threshold: float = 450.0,
                             enforce_detection: bool = True) -> Optional[List[Dict]]:
        """
        Search for similar faces
        
        Args:
            query_path: Path to query image
            k: Number of similar faces to retrieve
            threshold: Distance threshold for filtering results
            enforce_detection: Whether to enforce face detection
            
        Returns:
            List of similar face results or None if search fails
        """
        try:
            # Check if index is loaded
            if not self.index_manager.faiss_index:
                if not self._load_index():
                    return None
            
            # Extract embedding for query image
            query_embedding = self.embedding_extractor.extract_single_embedding(
                query_path, enforce_detection
            )
            
            if query_embedding is None:
                self.logger.error(f"Failed to extract embedding from {query_path}")
                return None
            
            # Perform similarity search
            distances, indices = self.index_manager.faiss_index.search(
                np.array([query_embedding]).astype("float32"), k
            )
            
            # Process results
            results = []
            labels = self.index_manager.get_labels()
            index_info = self.index_manager.get_index_info()
            source_folder = index_info.get("source_folder", "")
            
            for rank, idx in enumerate(indices[0]):
                if idx < len(labels):  # Ensure valid index
                    distance = float(distances[0][rank])
                    relative_path = labels[idx]
                    
                    # Construct original path
                    original_path = None
                    if source_folder:
                        original_path = os.path.join(source_folder, relative_path)
                    
                    result = {
                        'rank': rank + 1,
                        'filename': os.path.basename(relative_path),
                        'relative_path': relative_path,
                        'distance': distance,
                        'path': original_path,
                        'within_threshold': distance <= threshold,
                        'similarity_score': max(0, 1.0 - distance / 1000.0)  # Normalized similarity
                    }
                    
                    results.append(result)
            
            # Filter results by threshold if requested
            filtered_results = [r for r in results if r['within_threshold']]
            
            self.logger.info(f"Found {len(filtered_results)} matches within threshold from {len(results)} total results")
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return None
    
    def search_by_embedding(self,
                           query_embedding: np.ndarray,
                           k: int = 5,
                           threshold: float = 450.0) -> Optional[List[Dict]]:
        """
        Search using pre-computed embedding
        
        Args:
            query_embedding: Pre-computed face embedding
            k: Number of similar faces to retrieve  
            threshold: Distance threshold for filtering results
            
        Returns:
            List of similar face results or None if search fails
        """
        try:
            if not self.index_manager.faiss_index:
                if not self._load_index():
                    return None
            
            # Perform similarity search
            distances, indices = self.index_manager.faiss_index.search(
                np.array([query_embedding]).astype("float32"), k
            )
            
            # Process results (same as search_similar_faces)
            results = []
            labels = self.index_manager.get_labels()
            index_info = self.index_manager.get_index_info()
            source_folder = index_info.get("source_folder", "")
            
            for rank, idx in enumerate(indices[0]):
                if idx < len(labels):
                    distance = float(distances[0][rank])
                    relative_path = labels[idx]
                    
                    original_path = None
                    if source_folder:
                        original_path = os.path.join(source_folder, relative_path)
                    
                    result = {
                        'rank': rank + 1,
                        'filename': os.path.basename(relative_path),
                        'relative_path': relative_path,
                        'distance': distance,
                        'path': original_path,
                        'within_threshold': distance <= threshold,
                        'similarity_score': max(0, 1.0 - distance / 1000.0)
                    }
                    
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Embedding search failed: {e}")
            return None
    
    def get_search_stats(self) -> Dict:
        """Get search engine statistics"""
        try:
            stats = {
                'index_loaded': self.index_manager.faiss_index is not None,
                'total_indexed_faces': len(self.index_manager.get_labels()),
                'embedding_dimension': self.embedding_extractor.get_embedding_dimension(),
                'model_name': self.embedding_extractor.model_name,
                'detector_backend': self.embedding_extractor.detector_backend
            }
            
            # Add index stats if available
            if self.index_manager.faiss_index:
                index_stats = self.index_manager.get_index_stats()
                stats.update(index_stats)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get search stats: {e}")
            return {"error": str(e)}
    
    def reload_index(self) -> bool:
        """Reload the search index"""
        return self._load_index()
    
    def is_ready(self) -> bool:
        """Check if search engine is ready to perform searches"""
        return (self.index_manager.faiss_index is not None and 
                len(self.index_manager.get_labels()) > 0)
