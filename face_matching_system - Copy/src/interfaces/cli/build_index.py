#!/usr/bin/env python3
"""
CLI tool for building FAISS index from images.
"""

import os
import sys
import argparse
import logging
import shutil

# Add src to path for imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, src_path)

from core.orchestrator import FaceMatchingOrchestrator
from utils.logging.logger import setup_logging
from config.settings import Settings

def main():
    """Main entry point for index building CLI"""
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Build face index from images')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to folder containing images')
    parser.add_argument('--preprocess', action='store_true',
                        help='Enable image preprocessing')
    parser.add_argument('--embeddings-dir', type=str,
                        help='Custom embeddings directory')
    parser.add_argument('--max-workers', type=int,
                        help='Maximum number of worker processes')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    # Initialize settings and components
    settings = Settings()

    # Initialize orchestrator
    orchestrator = FaceMatchingOrchestrator()

    try:
        # Validate source folder using orchestrator
        validation_result = orchestrator.validate_source_folder(args.source)
        if not validation_result['valid']:
            logger.error(f"❌ {validation_result['error']}")
            sys.exit(1)

        logger.info(f"🖼️ Found {validation_result['image_count']} images in {args.source}")

        # Build complete index using orchestrator
        logger.info("🏗️ Building complete index...")
        result = orchestrator.build_complete_index(
            source_folder=args.source,
            use_preprocessing=args.preprocess
        )

        if result['success']:
            logger.info("✅ Index built successfully!")
            logger.info(f"🧠 Indexed {result['indexed_faces']} faces from {result['total_images']} images")
            logger.info(f"📁 Failed: {result['failed_extractions']} images")

            # Display summary
            stats = orchestrator.get_index_statistics()
            logger.info("📊 Index Summary:")
            logger.info(f"   Source: {args.source}")
            logger.info(f"   Preprocessing: {'Yes' if args.preprocess else 'No'}")
            logger.info(f"   Model: {stats.get('model_used', 'Unknown')}")
            logger.info(f"   Created: {stats.get('created_at', 'Unknown')}")
        else:
            logger.error(f"❌ {result['error']}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Index building failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()