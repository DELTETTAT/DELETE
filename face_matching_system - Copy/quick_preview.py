
#!/usr/bin/env python3
"""
Simple Preprocessing Preview
Shows before/after comparison of preprocessing
"""

import os
import sys
import cv2
import numpy as np
import tempfile
import shutil

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from core.preprocessing.preprocessor import ImagePreprocessor
from utils.image.processing import ImageProcessor
from config.settings import Settings

def analyze_image(image_path):
    """Simple analysis of preprocessing effects"""
    
    print(f"\n🔍 Analyzing: {os.path.basename(image_path)}")
    print("=" * 50)
    
    # Load original
    image_processor = ImageProcessor()
    original = image_processor.load_image(image_path)
    
    if original is None:
        print("❌ Failed to load image")
        return
    
    # Create temp copy for preprocessing
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, os.path.basename(image_path))
    shutil.copy2(image_path, temp_path)
    
    # Apply preprocessing
    preprocessor = ImagePreprocessor()
    result = preprocessor.preprocess_image(temp_path, temp_path)
    
    if not result.get('success', False):
        print("❌ Preprocessing failed")
        shutil.rmtree(temp_dir)
        return
    
    # Load preprocessed
    preprocessed = image_processor.load_image(temp_path)
    
    # Show comparison stats
    print(f"📊 Original size: {original.shape}")
    print(f"📊 Preprocessed size: {preprocessed.shape}")
    
    # Calculate differences
    if original.shape != preprocessed.shape:
        original_resized = cv2.resize(original, preprocessed.shape[:2][::-1])
    else:
        original_resized = original
    
    diff = cv2.absdiff(original_resized, preprocessed)
    mean_diff = np.mean(diff)
    
    print(f"📊 Average pixel difference: {mean_diff:.2f}")
    
    # Brightness comparison
    orig_brightness = np.mean(cv2.cvtColor(original_resized, cv2.COLOR_BGR2GRAY))
    prep_brightness = np.mean(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY))
    
    print(f"📊 Brightness change: {prep_brightness - orig_brightness:+.2f}")
    
    # Quality assessment
    if mean_diff < 15:
        print("✅ Conservative processing - good for accuracy")
    elif mean_diff < 35:
        print("⚠️  Moderate processing - check results")
    else:
        print("🔴 Heavy processing - may affect accuracy")
    
    # Create side-by-side comparison
    comparison_path = f"preview_{os.path.basename(image_path)}"
    
    height = max(original.shape[0], preprocessed.shape[0])
    width = original.shape[1] + preprocessed.shape[1]
    comparison = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Place images side by side
    comparison[:original.shape[0], :original.shape[1]] = original
    comparison[:preprocessed.shape[0], original.shape[1]:] = preprocessed
    
    # Add divider
    cv2.line(comparison, (original.shape[1], 0), (original.shape[1], height), (255, 255, 255), 2)
    
    cv2.imwrite(comparison_path, comparison)
    print(f"💾 Comparison saved: {comparison_path}")
    
    # Cleanup
    shutil.rmtree(temp_dir)

def main():
    print("🔍 Simple Preprocessing Preview")
    print("=" * 40)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            analyze_image(image_path)
        else:
            print(f"❌ Image not found: {image_path}")
        return
    
    # Check upload directory
    settings = Settings()
    
    if os.path.exists(settings.UPLOAD_DIR):
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend([
                os.path.join(settings.UPLOAD_DIR, f) 
                for f in os.listdir(settings.UPLOAD_DIR) 
                if f.lower().endswith(ext)
            ])
        
        if image_files:
            print(f"📁 Found {len(image_files)} images")
            
            # Process first 3 images
            for i, image_path in enumerate(image_files[:3]):
                analyze_image(image_path)
                if i < min(2, len(image_files) - 1):
                    print("\n" + "-" * 50)
            
            print(f"\n💡 To analyze specific image:")
            print(f"python quick_preview.py path/to/image.jpg")
        else:
            print("❌ No images found in upload directory")
    else:
        print("❌ Upload directory not found")
        print("💡 Run the web app first or specify an image:")
        print("python quick_preview.py path/to/image.jpg")

if __name__ == "__main__":
    main()
