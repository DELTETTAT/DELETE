"""
Simple Preprocessing Visualizer
Essential functions for checking preprocessing effects
"""

import os
import sys
import cv2
import numpy as np
import tempfile
import shutil
from typing import Optional, Dict, Any

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path)

from core.preprocessing.preprocessor import ImagePreprocessor
from utils.image.processing import ImageProcessor

def quick_compare(image_path: str) -> Dict[str, Any]:
    """
    Quick comparison of original vs preprocessed image

    Args:
        image_path: Path to image

    Returns:
        Dict with comparison stats
    """

    image_processor = ImageProcessor()
    original = image_processor.load_image(image_path)

    if original is None:
        return {'success': False, 'error': 'Failed to load image'}

    # Create temp preprocessed version
    temp_dir = tempfile.mkdtemp(prefix="quick_compare_")
    temp_path = os.path.join(temp_dir, os.path.basename(image_path))
    shutil.copy2(image_path, temp_path)

    preprocessor = ImagePreprocessor()
    result = preprocessor.preprocess_image(temp_path, temp_path)

    if not result.get('success', False):
        shutil.rmtree(temp_dir)
        return {'success': False, 'error': 'Preprocessing failed'}

    preprocessed = image_processor.load_image(temp_path)

    # Calculate differences
    if original.shape != preprocessed.shape:
        original_resized = cv2.resize(original, preprocessed.shape[:2][::-1])
    else:
        original_resized = original

    diff = cv2.absdiff(original_resized, preprocessed)
    mean_diff = np.mean(diff)

    # Brightness comparison
    orig_brightness = np.mean(cv2.cvtColor(original_resized, cv2.COLOR_BGR2GRAY))
    prep_brightness = np.mean(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY))

    # Cleanup
    shutil.rmtree(temp_dir)

    return {
        'success': True,
        'original_shape': original.shape,
        'preprocessed_shape': preprocessed.shape,
        'mean_difference': float(mean_diff),
        'brightness_change': float(prep_brightness - orig_brightness),
        'processing_applied': result.get('processing_applied', []),
        'quality_assessment': 'minimal' if mean_diff < 10 else 'moderate' if mean_diff < 30 else 'heavy'
    }

def create_side_by_side(image_path: str, output_path: Optional[str] = None) -> bool:
    """
    Create simple side-by-side comparison image

    Args:
        image_path: Path to original image
        output_path: Where to save comparison (optional)

    Returns:
        bool: Success status
    """

    try:
        image_processor = ImageProcessor()
        original = image_processor.load_image(image_path)

        if original is None:
            return False

        # Create temp preprocessed version
        temp_dir = tempfile.mkdtemp(prefix="side_by_side_")
        temp_path = os.path.join(temp_dir, os.path.basename(image_path))
        shutil.copy2(image_path, temp_path)

        preprocessor = ImagePreprocessor()
        result = preprocessor.preprocess_image(temp_path, temp_path)

        if not result.get('success', False):
            shutil.rmtree(temp_dir)
            return False

        preprocessed = image_processor.load_image(temp_path)

        # Create side-by-side image
        height = max(original.shape[0], preprocessed.shape[0])
        width = original.shape[1] + preprocessed.shape[1]
        comparison = np.zeros((height, width, 3), dtype=np.uint8)

        # Place images side by side
        comparison[:original.shape[0], :original.shape[1]] = original
        comparison[:preprocessed.shape[0], original.shape[1]:] = preprocessed

        # Add divider line
        cv2.line(comparison, (original.shape[1], 0), (original.shape[1], height), (255, 255, 255), 2)

        # Save comparison
        if output_path is None:
            output_path = f"comparison_{os.path.basename(image_path)}"

        cv2.imwrite(output_path, comparison)

        # Cleanup
        shutil.rmtree(temp_dir)
        return True

    except Exception:
        return False

def main():
    """CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(description='Simple preprocessing visualization')
    parser.add_argument('image', help='Image to analyze')
    parser.add_argument('--output', '-o', help='Output path for comparison image')

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return

    # Quick analysis
    stats = quick_compare(args.image)

    if stats['success']:
        print(f"\n📊 Preprocessing Analysis for {os.path.basename(args.image)}")
        print("=" * 50)
        print(f"Original size: {stats['original_shape']}")
        print(f"Preprocessed size: {stats['preprocessed_shape']}")
        print(f"Average difference: {stats['mean_difference']:.2f}")
        print(f"Brightness change: {stats['brightness_change']:+.2f}")
        print(f"Quality: {stats['quality_assessment']}")
        print(f"Processing: {stats['processing_applied']}")

        # Create comparison image
        output_path = args.output or f"comparison_{os.path.basename(args.image)}"
        if create_side_by_side(args.image, output_path):
            print(f"\n💾 Comparison saved: {output_path}")
        else:
            print("\n❌ Failed to create comparison image")
    else:
        print(f"❌ Analysis failed: {stats.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()