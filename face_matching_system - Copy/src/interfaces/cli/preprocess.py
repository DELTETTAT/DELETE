#!/usr/bin/env python3
"""
CLI tool for preprocessing images.
"""

import os
import sys
import argparse
import logging
import shutil

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.preprocessing.preprocessor import ImagePreprocessor
from utils.filesystem.operations import FileSystemOperations
from utils.logging.logger import setup_logging
from config.settings import Settings

def main():
    """Main entry point for preprocessing CLI"""
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Preprocess images for face recognition')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to folder containing images')
    parser.add_argument('--output', type=str,
                        help='Output directory for preprocessed images')
    parser.add_argument('--target-size', type=int, nargs=2, default=[160, 160],
                        help='Target size for images (width height)')
    parser.add_argument('--no-alignment', action='store_true',
                        help='Disable face alignment')
    parser.add_argument('--no-enhancement', action='store_true',
                        help='Disable image enhancement')
    parser.add_argument('--quality-threshold', type=float, default=0.7,
                        help='Quality threshold for processing')
    parser.add_argument('--assess-quality', action='store_true',
                        help='Assess image quality without processing')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    # Initialize components
    settings = Settings()
    fs_ops = FileSystemOperations()
    
    # Validate source folder
    if not os.path.exists(args.source):
        logger.error(f"❌ Source folder not found: {args.source}")
        sys.exit(1)
    
    # Set output directory
    output_dir = args.output or settings.PREPROCESSED_DIR
    
    try:
        # Get all image files
        image_files = fs_ops.get_all_images_from_folder(args.source)
        if not image_files:
            logger.error(f"🖼️ No images found in: {args.source}")
            sys.exit(1)
        
        logger.info(f"🖼️ Found {len(image_files)} images in {args.source}")
        
        # Initialize preprocessor
        preprocessor = ImagePreprocessor(
            target_size=tuple(args.target_size),
            enable_face_alignment=not args.no_alignment,
            enable_enhancement=not args.no_enhancement,
            quality_threshold=args.quality_threshold
        )
        
        if args.assess_quality:
            # Quality assessment mode
            logger.info("📊 Assessing image quality...")
            
            quality_results = []
            for original_path, relative_path in image_files:
                quality = preprocessor.assess_quality(original_path)
                
                if "error" not in quality:
                    quality_results.append({
                        'path': relative_path,
                        'overall_score': quality.get('overall_score', 0),
                        'is_acceptable': quality.get('is_acceptable', False),
                        **quality
                    })
                    
                    status = "✅" if quality.get('is_acceptable', False) else "❌"
                    score = quality.get('overall_score', 0)
                    logger.info(f"{status} {relative_path}: Quality = {score:.3f}")
                else:
                    logger.error(f"❌ {relative_path}: {quality['error']}")
            
            # Summary
            acceptable_count = sum(1 for r in quality_results if r['is_acceptable'])
            logger.info(f"\n📊 Quality Assessment Summary:")
            logger.info(f"   Total images: {len(quality_results)}")
            logger.info(f"   Acceptable quality: {acceptable_count}")
            logger.info(f"   Below threshold: {len(quality_results) - acceptable_count}")
            
        else:
            # Preprocessing mode
            logger.info("🔄 Preprocessing images...")
            
            # Clear and create output directory
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            successful_count = 0
            failed_count = 0
            
            for i, (original_path, relative_path) in enumerate(image_files):
                # Create output path
                output_path = os.path.join(output_dir, relative_path)
                output_dir_path = os.path.dirname(output_path)
                os.makedirs(output_dir_path, exist_ok=True)
                
                # Preprocess image
                success = preprocessor.preprocess_image(original_path, output_path)
                
                if success:
                    successful_count += 1
                    logger.debug(f"✅ Processed: {relative_path} ({i+1}/{len(image_files)})")
                else:
                    failed_count += 1
                    logger.warning(f"❌ Failed: {relative_path}")
            
            # Summary
            logger.info(f"\n📊 Preprocessing Summary:")
            logger.info(f"   Total images: {len(image_files)}")
            logger.info(f"   Successfully processed: {successful_count}")
            logger.info(f"   Failed: {failed_count}")
            logger.info(f"   Output directory: {output_dir}")
            
            if successful_count == 0:
                logger.error("❌ No images were successfully preprocessed")
                sys.exit(1)
            else:
                logger.info("✅ Preprocessing completed!")
    
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
