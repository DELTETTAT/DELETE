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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.preprocessing.preprocessor import ImagePreprocessor
from core.embedding.extractor import EmbeddingExtractor
from core.indexing.manager import IndexManager
from utils.filesystem.operations import FileSystemOperations
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
    fs_ops = FileSystemOperations()
    
    # Validate source folder
    if not os.path.exists(args.source):
        logger.error(f"❌ Source folder not found: {args.source}")
        sys.exit(1)
    
    try:
        # Get all image files from source folder
        image_files = fs_ops.get_all_images_from_folder(args.source)
        if not image_files:
            logger.error(f"🖼️ No images found in: {args.source}")
            sys.exit(1)
        
        logger.info(f"🖼️ Found {len(image_files)} images in {args.source}")
        
        # Initialize components
        preprocessor = ImagePreprocessor() if args.preprocess else None
        embedding_extractor = EmbeddingExtractor()
        index_manager = IndexManager(embeddings_dir=args.embeddings_dir)
        
        # Setup preprocessed directory if needed
        preprocessed_dir = settings.PREPROCESSED_DIR
        if args.preprocess:
            if os.path.exists(preprocessed_dir):
                shutil.rmtree(preprocessed_dir)
            os.makedirs(preprocessed_dir, exist_ok=True)
            logger.info("🔄 Preprocessing enabled")
        
        # Process images
        processed_files = []
        if args.preprocess and preprocessor:
            logger.info("🔄 Preprocessing images...")
            for i, (original_path, relative_path) in enumerate(image_files):
                # Create preprocessed copy
                preprocessed_path = os.path.join(preprocessed_dir, relative_path)
                preprocessed_dir_path = os.path.dirname(preprocessed_path)
                os.makedirs(preprocessed_dir_path, exist_ok=True)
                
                # Copy and preprocess
                shutil.copy2(original_path, preprocessed_path)
                if preprocessor.preprocess_image(preprocessed_path, preprocessed_path):
                    processed_files.append((preprocessed_path, relative_path))
                    logger.debug(f"✅ Preprocessed: {relative_path} ({i+1}/{len(image_files)})")
                else:
                    logger.warning(f"⚠️ Preprocessing failed: {relative_path}")
        else:
            # Use original files without preprocessing
            for original_path, relative_path in image_files:
                processed_files.append((original_path, relative_path))
        
        if not processed_files:
            logger.error("❌ No images available for processing")
            sys.exit(1)
        
        # Extract embeddings
        logger.info("🧠 Extracting face embeddings...")
        image_paths = [path for path, _ in processed_files]
        embeddings, successful_paths = embedding_extractor.extract_batch_embeddings(
            image_paths=image_paths,
            enforce_detection=True,
            max_workers=args.max_workers
        )
        
        if not embeddings:
            logger.error("⚠️ No embeddings created. Check logs above.")
            sys.exit(1)
        
        # Create labels for successful extractions
        path_to_relative = {path: rel_path for path, rel_path in processed_files}
        labels = []
        for successful_path in successful_paths:
            if successful_path in path_to_relative:
                labels.append(path_to_relative[successful_path])
        
        # Build and save index
        logger.info("💾 Building FAISS index...")
        additional_metadata = {
            "total_images": len(image_files),
            "failed_embeddings": len(image_files) - len(embeddings)
        }
        
        success = index_manager.build_index(
            embeddings=embeddings,
            labels=labels,
            source_folder=args.source,
            preprocessing_used=args.preprocess,
            additional_metadata=additional_metadata
        )
        
        if success:
            logger.info("✅ Index built successfully!")
            logger.info(f"🧠 Indexed {len(embeddings)} faces from {len(image_files)} images")
            logger.info(f"📁 Failed: {len(image_files) - len(embeddings)} images")
            
            # Display summary
            stats = index_manager.get_index_stats()
            logger.info("📊 Index Summary:")
            logger.info(f"   Source: {args.source}")
            logger.info(f"   Preprocessing: {'Yes' if args.preprocess else 'No'}")
            logger.info(f"   Model: {stats.get('model_used', 'Unknown')}")
            logger.info(f"   Created: {stats.get('created_at', 'Unknown')}")
        else:
            logger.error("❌ Failed to build index")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ Index building failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
