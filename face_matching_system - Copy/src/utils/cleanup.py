
"""
Cleanup utilities for the face matching system.
Manages temporary files, preprocessed images, and cache cleanup.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional
from config.settings import Settings

logger = logging.getLogger(__name__)

class CleanupManager:
    """Manages cleanup operations for the face matching system"""
    
    def __init__(self):
        self.settings = Settings()
    
    def cleanup_preprocessed_images(self, keep_recent: bool = False, hours: int = 24) -> int:
        """
        Clean up preprocessed images after embeddings are generated.
        
        Args:
            keep_recent: Whether to keep recent files
            hours: How many hours to keep recent files
            
        Returns:
            Number of files cleaned
        """
        cleaned_count = 0
        preprocessed_dir = self.settings.PREPROCESSED_DIR
        
        if not preprocessed_dir.exists():
            return cleaned_count
            
        try:
            import time
            current_time = time.time()
            cutoff_time = current_time - (hours * 3600) if keep_recent else 0
            
            for file_path in preprocessed_dir.rglob("*"):
                if file_path.is_file():
                    if not keep_recent or file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.debug(f"Cleaned preprocessed file: {file_path}")
            
            # Remove empty directories
            for dir_path in preprocessed_dir.rglob("*"):
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    logger.debug(f"Removed empty directory: {dir_path}")
                    
        except Exception as e:
            logger.error(f"Error cleaning preprocessed images: {e}")
            
        return cleaned_count
    
    def cleanup_temp_files(self) -> int:
        """Clean up temporary files"""
        cleaned_count = 0
        temp_dir = self.settings.TEMP_DIR
        
        if not temp_dir.exists():
            return cleaned_count
            
        try:
            for file_path in temp_dir.rglob("*"):
                if file_path.is_file():
                    file_path.unlink()
                    cleaned_count += 1
                    logger.debug(f"Cleaned temp file: {file_path}")
                    
        except Exception as e:
            logger.error(f"Error cleaning temp files: {e}")
            
        return cleaned_count
    
    def cleanup_cache(self, max_age_days: int = 7) -> int:
        """Clean up old cache files"""
        cleaned_count = 0
        cache_dir = self.settings.CACHE_DIR
        
        if not cache_dir.exists():
            return cleaned_count
            
        try:
            import time
            current_time = time.time()
            cutoff_time = current_time - (max_age_days * 24 * 3600)
            
            for file_path in cache_dir.rglob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    cleaned_count += 1
                    logger.debug(f"Cleaned cache file: {file_path}")
                    
        except Exception as e:
            logger.error(f"Error cleaning cache: {e}")
            
        return cleaned_count
    
    def auto_cleanup_after_indexing(self):
        """Automatic cleanup after indexing operations"""
        try:
            # Clean preprocessed images immediately after indexing
            cleaned_preprocessed = self.cleanup_preprocessed_images()
            
            # Clean old temp files
            cleaned_temp = self.cleanup_temp_files()
            
            logger.info(f"Auto cleanup completed: {cleaned_preprocessed} preprocessed, {cleaned_temp} temp files")
            
        except Exception as e:
            logger.error(f"Auto cleanup failed: {e}")
    
    def get_storage_stats(self) -> dict:
        """Get storage statistics for data directories"""
        stats = {}
        
        directories = {
            'temp': self.settings.TEMP_DIR,
            'preprocessed': self.settings.PREPROCESSED_DIR,
            'embeddings': self.settings.EMBEDDINGS_DIR,
            'cache': self.settings.CACHE_DIR,
            'indexes': self.settings.INDEXES_DIR
        }
        
        for name, path in directories.items():
            if path.exists():
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                count = len([f for f in path.rglob('*') if f.is_file()])
                stats[name] = {'size_mb': round(size / (1024 * 1024), 2), 'file_count': count}
            else:
                stats[name] = {'size_mb': 0, 'file_count': 0}
                
        return stats

# Global cleanup manager instance
cleanup_manager = CleanupManager()
