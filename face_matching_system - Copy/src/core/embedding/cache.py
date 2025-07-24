
import os
import pickle
import hashlib
import numpy as np
import logging
from typing import Optional, Dict, Set
from datetime import datetime

class EmbeddingCache:
    """
    Cache for face embeddings to avoid recomputation.
    Uses file hash as key for cache invalidation.
    """
    
    def __init__(self, cache_dir: str = "cache/embeddings", max_cache_size: int = 1000):
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # In-memory cache for frequently accessed embeddings
        self._memory_cache: Dict[str, np.ndarray] = {}
        
    def _get_file_hash(self, file_path: str) -> str:
        """Get MD5 hash of file for cache key"""
        try:
            with open(file_path, 'rb') as f:
                # Read file in chunks for large files
                hash_md5 = hashlib.md5()
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to compute hash for {file_path}: {e}")
            return None
    
    def get_embedding(self, file_path: str) -> Optional[np.ndarray]:
        """Get cached embedding if available"""
        try:
            file_hash = self._get_file_hash(file_path)
            if not file_hash:
                return None
            
            # Check memory cache first
            if file_hash in self._memory_cache:
                return self._memory_cache[file_hash]
            
            # Check disk cache
            cache_file = os.path.join(self.cache_dir, f"{file_hash}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                
                # Add to memory cache
                self._memory_cache[file_hash] = embedding
                self._cleanup_memory_cache()
                
                return embedding
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting cached embedding for {file_path}: {e}")
            return None
    
    def store_embedding(self, file_path: str, embedding: np.ndarray):
        """Store embedding in cache"""
        try:
            file_hash = self._get_file_hash(file_path)
            if not file_hash:
                return
            
            # Store in memory cache
            self._memory_cache[file_hash] = embedding
            self._cleanup_memory_cache()
            
            # Store in disk cache
            cache_file = os.path.join(self.cache_dir, f"{file_hash}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
            
        except Exception as e:
            self.logger.error(f"Error storing embedding for {file_path}: {e}")
    
    def _cleanup_memory_cache(self):
        """Keep memory cache size under limit"""
        if len(self._memory_cache) > self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            excess = len(self._memory_cache) - self.max_cache_size
            keys_to_remove = list(self._memory_cache.keys())[:excess]
            for key in keys_to_remove:
                del self._memory_cache[key]
    
    def get_dataset_hash(self, source_folder: str) -> str:
        """Get hash of entire dataset for change detection"""
        try:
            from utils.filesystem.operations import FileSystemOperations
            fs_ops = FileSystemOperations()
            image_files = fs_ops.get_all_images_from_folder(source_folder)
            
            # Combine all file hashes and modification times
            combined_data = []
            for file_path, _ in image_files:
                file_hash = self._get_file_hash(file_path)
                mod_time = os.path.getmtime(file_path)
                combined_data.append(f"{file_hash}:{mod_time}")
            
            # Create hash of combined data
            combined_str = "|".join(sorted(combined_data))
            return hashlib.md5(combined_str.encode()).hexdigest()
            
        except Exception as e:
            self.logger.error(f"Error computing dataset hash: {e}")
            return None
    
    def is_dataset_changed(self, source_folder: str, stored_hash: str) -> bool:
        """Check if dataset has changed since last index build"""
        current_hash = self.get_dataset_hash(source_folder)
        return current_hash != stored_hash if current_hash else True
    
    def clear_cache(self):
        """Clear all cached embeddings"""
        try:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            
            self._memory_cache.clear()
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
