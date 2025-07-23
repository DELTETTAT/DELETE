import os
import shutil
import logging
from pathlib import Path
from typing import List, Tuple

class FileSystemOperations:
    """
    File system operations for face matching system.
    Handles file discovery, directory management, and path operations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp')
    
    def get_all_images_from_folder(self, folder_path: str, recursive: bool = True) -> List[Tuple[str, str]]:
        """
        Get all image files from folder and optionally subfolders
        
        Args:
            folder_path: Path to the folder to search
            recursive: Whether to search recursively in subfolders
            
        Returns:
            List of tuples (full_path, relative_path)
        """
        image_files = []
        
        if not os.path.exists(folder_path):
            self.logger.warning(f"Folder does not exist: {folder_path}")
            return []
        
        try:
            if recursive:
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        if file.lower().endswith(self.image_extensions):
                            full_path = os.path.join(root, file)
                            # Create relative path for unique naming
                            rel_path = os.path.relpath(full_path, folder_path)
                            image_files.append((full_path, rel_path))
            else:
                for file in os.listdir(folder_path):
                    if file.lower().endswith(self.image_extensions):
                        full_path = os.path.join(folder_path, file)
                        image_files.append((full_path, file))
        
        except Exception as e:
            self.logger.error(f"Error scanning folder {folder_path}: {e}")
        
        return image_files
    
    def get_image_files_in_folder(self, folder_path: str) -> List[str]:
        """
        Get list of image filenames in a folder (non-recursive)
        
        Args:
            folder_path: Path to the folder
            
        Returns:
            List of image filenames
        """
        try:
            if not os.path.exists(folder_path):
                return []
            
            files = []
            for file in os.listdir(folder_path):
                if file.lower().endswith(self.image_extensions):
                    files.append(file)
            
            return sorted(files)
            
        except Exception as e:
            self.logger.error(f"Error listing files in {folder_path}: {e}")
            return []
    
    def count_images_in_folder(self, folder_path: str, recursive: bool = True) -> int:
        """
        Count image files in a folder
        
        Args:
            folder_path: Path to the folder
            recursive: Whether to count recursively
            
        Returns:
            Number of image files found
        """
        try:
            if not os.path.exists(folder_path):
                return 0
            
            count = 0
            if recursive:
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        if file.lower().endswith(self.image_extensions):
                            count += 1
            else:
                for file in os.listdir(folder_path):
                    if file.lower().endswith(self.image_extensions):
                        count += 1
            
            return count
            
        except Exception as e:
            self.logger.error(f"Error counting images in {folder_path}: {e}")
            return 0
    
    def ensure_directory_exists(self, directory_path: str) -> bool:
        """
        Ensure a directory exists, create if it doesn't
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            True if directory exists or was created successfully
        """
        try:
            os.makedirs(directory_path, exist_ok=True)
            return True
        except Exception as e:
            self.logger.error(f"Failed to create directory {directory_path}: {e}")
            return False
    
    def copy_file(self, source_path: str, destination_path: str, 
                  create_dirs: bool = True) -> bool:
        """
        Copy a file from source to destination
        
        Args:
            source_path: Source file path
            destination_path: Destination file path
            create_dirs: Whether to create destination directories
            
        Returns:
            True if copy was successful
        """
        try:
            if not os.path.exists(source_path):
                self.logger.error(f"Source file does not exist: {source_path}")
                return False
            
            if create_dirs:
                dest_dir = os.path.dirname(destination_path)
                if dest_dir:
                    self.ensure_directory_exists(dest_dir)
            
            shutil.copy2(source_path, destination_path)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to copy {source_path} to {destination_path}: {e}")
            return False
    
    def move_file(self, source_path: str, destination_path: str,
                  create_dirs: bool = True) -> bool:
        """
        Move a file from source to destination
        
        Args:
            source_path: Source file path
            destination_path: Destination file path
            create_dirs: Whether to create destination directories
            
        Returns:
            True if move was successful
        """
        try:
            if not os.path.exists(source_path):
                self.logger.error(f"Source file does not exist: {source_path}")
                return False
            
            if create_dirs:
                dest_dir = os.path.dirname(destination_path)
                if dest_dir:
                    self.ensure_directory_exists(dest_dir)
            
            shutil.move(source_path, destination_path)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to move {source_path} to {destination_path}: {e}")
            return False
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            else:
                self.logger.warning(f"File does not exist: {file_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete {file_path}: {e}")
            return False
    
    def delete_directory(self, directory_path: str, recursive: bool = False) -> bool:
        """
        Delete a directory
        
        Args:
            directory_path: Path to the directory to delete
            recursive: Whether to delete recursively (including contents)
            
        Returns:
            True if deletion was successful
        """
        try:
            if not os.path.exists(directory_path):
                self.logger.warning(f"Directory does not exist: {directory_path}")
                return False
            
            if recursive:
                shutil.rmtree(directory_path)
            else:
                os.rmdir(directory_path)  # Only works if directory is empty
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete directory {directory_path}: {e}")
            return False
    
    def get_file_size(self, file_path: str) -> int:
        """
        Get file size in bytes
        
        Args:
            file_path: Path to the file
            
        Returns:
            File size in bytes, 0 if file doesn't exist or error occurs
        """
        try:
            if os.path.exists(file_path):
                return os.path.getsize(file_path)
            else:
                return 0
        except Exception as e:
            self.logger.error(f"Failed to get size of {file_path}: {e}")
            return 0
    
    def get_directory_size(self, directory_path: str) -> int:
        """
        Get total size of directory in bytes
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            Total directory size in bytes
        """
        try:
            total_size = 0
            if os.path.exists(directory_path):
                for dirpath, dirnames, filenames in os.walk(directory_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        try:
                            total_size += os.path.getsize(filepath)
                        except (OSError, FileNotFoundError):
                            # Skip files that can't be accessed
                            pass
            return total_size
        except Exception as e:
            self.logger.error(f"Failed to get directory size for {directory_path}: {e}")
            return 0
    
    def clean_directory(self, directory_path: str, keep_structure: bool = True) -> bool:
        """
        Clean/empty a directory
        
        Args:
            directory_path: Path to the directory to clean
            keep_structure: Whether to keep the directory structure (only delete files)
            
        Returns:
            True if cleaning was successful
        """
        try:
            if not os.path.exists(directory_path):
                self.logger.warning(f"Directory does not exist: {directory_path}")
                return False
            
            if keep_structure:
                # Remove only files, keep subdirectories
                for root, dirs, files in os.walk(directory_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        os.remove(file_path)
            else:
                # Remove everything and recreate directory
                shutil.rmtree(directory_path)
                os.makedirs(directory_path, exist_ok=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clean directory {directory_path}: {e}")
            return False
    
    def is_image_file(self, file_path: str) -> bool:
        """
        Check if a file is an image based on extension
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file has an image extension
        """
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.image_extensions
    
    def get_relative_path(self, file_path: str, base_path: str) -> str:
        """
        Get relative path from base path
        
        Args:
            file_path: Full file path
            base_path: Base directory path
            
        Returns:
            Relative path
        """
        try:
            return os.path.relpath(file_path, base_path)
        except Exception:
            return file_path
    
    def create_unique_filename(self, directory: str, filename: str) -> str:
        """
        Create a unique filename in a directory by adding suffix if needed
        
        Args:
            directory: Target directory
            filename: Desired filename
            
        Returns:
            Unique filename that doesn't exist in the directory
        """
        base_name, ext = os.path.splitext(filename)
        counter = 1
        unique_filename = filename
        
        while os.path.exists(os.path.join(directory, unique_filename)):
            unique_filename = f"{base_name}_{counter}{ext}"
            counter += 1
        
        return unique_filename

    def get_image_files(self, folder_path: str) -> List[str]:
        """Get list of all image files in folder and subfolders"""
        image_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if self.is_image_file(file):
                    image_files.append(os.path.join(root, file))
        return image_files