#!/usr/bin/env python3
"""
CLI tool for preprocessing images.
"""

import os
import sys
import argparse
import logging

# Add src to path for imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, src_path)

from core.orchestrator import FaceMatchingOrchestrator
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

    # Initialize settings
    settings = Settings()

    # Validate source folder
    if not os.path.exists(args.source):
        logger.error(f"❌ Source folder not found: {args.source}")
        sys.exit(1)

    # Set output directory
    output_dir = args.output or settings.PREPROCESSED_DIR

    try:
        # Initialize orchestrator
        orchestrator = FaceMatchingOrchestrator()

        # Validate source folder
        validation_result = orchestrator.validate_source_folder(args.source)
        if not validation_result['valid']:
            logger.error(f"❌ {validation_result['error']}")
            sys.exit(1)

        logger.info(f"🖼️ Found {validation_result['image_count']} images in {args.source}")

        if args.assess_quality:
            # Quality assessment mode
            logger.info("📊 Assessing image quality...")
            logger.warning("⚠️ Quality assessment mode requires direct preprocessor access")
            logger.info("Tip: Use preprocessing mode to apply quality filtering automatically")

            # For now, suggest using preprocessing mode instead
            logger.info("Consider using preprocessing mode which applies quality filtering:")
            logger.info(f"python {__file__} --source {args.source} --quality-threshold {args.quality_threshold}")

        else:
            # Preprocessing mode using orchestrator
            logger.info("🔄 Preprocessing images...")

            result = orchestrator.preprocess_images_from_folder(
                source_folder=args.source,
                clear_existing=True
            )

            if result['success']:
                logger.info(f"\n📊 Preprocessing Summary:")
                logger.info(f"   Total images: {result['total_images']}")
                logger.info(f"   Successfully processed: {result['successful']}")
                logger.info(f"   Failed: {result['failed']}")
                logger.info(f"   Output directory: {settings.PREPROCESSED_DIR}")

                if result['successful'] == 0:
                    logger.error("❌ No images were successfully preprocessed")
                    sys.exit(1)
                else:
                    logger.info("✅ Preprocessing completed!")

                    # Show some processed files
                    if result.get('processed_files'):
                        logger.info("📁 Sample processed files:")
                        for i, (full_path, relative_path) in enumerate(result['processed_files'][:5]):
                            logger.info(f"   ✓ {relative_path}")
                        if len(result['processed_files']) > 5:
                            logger.info(f"   ... and {len(result['processed_files']) - 5} more")
            else:
                logger.error(f"❌ Preprocessing failed: {result['error']}")
                sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()