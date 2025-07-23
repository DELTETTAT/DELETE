import os
import cv2
import numpy as np
import logging
from PIL import Image
from typing import Tuple, Optional

class ImageProcessor:
    """
    Common image processing utilities.
    Handles loading, saving, resizing, and format conversion.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp')
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load image from file path
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as numpy array in RGB format or None if loading fails
        """
        try:
            if not os.path.exists(image_path):
                self.logger.error(f"Image not found: {image_path}")
                return None
            
            # Load with PIL first to handle various formats
            pil_image = Image.open(image_path)
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            image = np.array(pil_image)
            
            # Basic validation
            if image.shape[0] < 10 or image.shape[1] < 10:
                self.logger.warning(f"Image too small: {image.shape}")
                return None
            
            return image
            
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def save_image(self, image: np.ndarray, output_path: str, quality: int = 95) -> bool:
        """
        Save image to file
        
        Args:
            image: Image as numpy array
            output_path: Path to save the image
            quality: JPEG quality (for JPEG files)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert to PIL and save
            if image.dtype != np.uint8:
                # Normalize to 0-255 range if needed
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image)
            
            # Determine format from extension
            ext = os.path.splitext(output_path)[1].lower()
            if ext in ['.jpg', '.jpeg']:
                pil_image.save(output_path, 'JPEG', quality=quality)
            elif ext == '.png':
                pil_image.save(output_path, 'PNG')
            elif ext == '.bmp':
                pil_image.save(output_path, 'BMP')
            elif ext == '.tiff':
                pil_image.save(output_path, 'TIFF')
            else:
                # Default to JPEG
                pil_image.save(output_path, 'JPEG', quality=quality)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save image to {output_path}: {e}")
            return False
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int], 
                     maintain_aspect_ratio: bool = True) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image as numpy array
            target_size: (width, height) tuple
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized image as numpy array
        """
        try:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            if maintain_aspect_ratio:
                # Calculate scaling factor to maintain aspect ratio
                scale = min(target_w / w, target_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                # Resize image
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                
                # Create canvas and center the image
                if len(image.shape) == 3:
                    canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
                else:
                    canvas = np.zeros((target_h, target_w), dtype=image.dtype)
                
                # Calculate position to center the image
                start_h = (target_h - new_h) // 2
                start_w = (target_w - new_w) // 2
                
                if len(image.shape) == 3:
                    canvas[start_h:start_h + new_h, start_w:start_w + new_w] = resized
                else:
                    canvas[start_h:start_h + new_h, start_w:start_w + new_w] = resized
                
                return canvas
            else:
                # Direct resize without maintaining aspect ratio
                return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
                
        except Exception as e:
            self.logger.error(f"Resizing failed: {e}")
            return image
    
    def normalize_image(self, image: np.ndarray, method: str = 'standard') -> np.ndarray:
        """
        Normalize image pixel values
        
        Args:
            image: Input image as numpy array
            method: Normalization method ('standard', 'minmax', 'zscore')
            
        Returns:
            Normalized image as numpy array
        """
        try:
            if method == 'standard':
                # Standard normalization to [0, 1]
                if image.dtype == np.uint8:
                    return image.astype(np.float32) / 255.0
                else:
                    return image.astype(np.float32)
            
            elif method == 'minmax':
                # Min-max normalization to [0, 1]
                image_float = image.astype(np.float32)
                min_val = image_float.min()
                max_val = image_float.max()
                if max_val > min_val:
                    return (image_float - min_val) / (max_val - min_val)
                else:
                    return image_float
            
            elif method == 'zscore':
                # Z-score normalization (mean=0, std=1)
                image_float = image.astype(np.float32)
                mean = np.mean(image_float)
                std = np.std(image_float)
                if std > 0:
                    return (image_float - mean) / std
                else:
                    return image_float - mean
            
            else:
                self.logger.warning(f"Unknown normalization method: {method}")
                return image
                
        except Exception as e:
            self.logger.error(f"Normalization failed: {e}")
            return image
    
    def convert_color_space(self, image: np.ndarray, 
                           from_space: str = 'RGB', to_space: str = 'BGR') -> np.ndarray:
        """
        Convert image between color spaces
        
        Args:
            image: Input image
            from_space: Source color space
            to_space: Target color space
            
        Returns:
            Converted image
        """
        try:
            if from_space == to_space:
                return image
            
            conversion_map = {
                ('RGB', 'BGR'): cv2.COLOR_RGB2BGR,
                ('BGR', 'RGB'): cv2.COLOR_BGR2RGB,
                ('RGB', 'GRAY'): cv2.COLOR_RGB2GRAY,
                ('BGR', 'GRAY'): cv2.COLOR_BGR2GRAY,
                ('RGB', 'HSV'): cv2.COLOR_RGB2HSV,
                ('BGR', 'HSV'): cv2.COLOR_BGR2HSV,
                ('HSV', 'RGB'): cv2.COLOR_HSV2RGB,
                ('HSV', 'BGR'): cv2.COLOR_HSV2BGR,
            }
            
            conversion_key = (from_space, to_space)
            if conversion_key in conversion_map:
                return cv2.cvtColor(image, conversion_map[conversion_key])
            else:
                self.logger.warning(f"Unsupported color conversion: {from_space} -> {to_space}")
                return image
                
        except Exception as e:
            self.logger.error(f"Color conversion failed: {e}")
            return image
    
    def is_supported_format(self, file_path: str) -> bool:
        """Check if file format is supported"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.supported_extensions
    
    def get_image_info(self, image_path: str) -> dict:
        """Get basic image information"""
        try:
            if not os.path.exists(image_path):
                return {"error": "File not found"}
            
            # Get file size
            file_size = os.path.getsize(image_path)
            
            # Load image to get dimensions
            image = self.load_image(image_path)
            if image is None:
                return {"error": "Failed to load image"}
            
            info = {
                "path": image_path,
                "filename": os.path.basename(image_path),
                "file_size": file_size,
                "file_size_mb": file_size / (1024 * 1024),
                "dimensions": image.shape[:2],  # (height, width)
                "channels": image.shape[2] if len(image.shape) == 3 else 1,
                "dtype": str(image.dtype),
                "format": os.path.splitext(image_path)[1].lower(),
                "supported": self.is_supported_format(image_path)
            }
            
            return info
            
        except Exception as e:
            return {"error": str(e)}

