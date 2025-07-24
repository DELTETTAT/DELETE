#!/usr/bin/env python3
"""
CLI tool for searching similar faces using the face matching system.
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
    """Main entry point for face search CLI"""
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Search for similar faces')
    parser.add_argument('--query', type=str, required=True, 
                        help='Path to query image')
    parser.add_argument('--k', type=int, default=3, 
                        help='Number of top matches to retrieve (default: 3)')
    parser.add_argument('--threshold', type=float, default=450.0, 
                        help='Distance threshold for filtering (default: 450.0)')
    parser.add_argument('--embeddings-dir', type=str, 
                        help='Custom embeddings directory')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Enable verbose logging')
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    # Validate query image exists
    if not os.path.exists(args.query):
        logger.error(f"❌ Query image not found: {args.query}")
        sys.exit(1)
    
    try:
        # Initialize orchestrator
        orchestrator = FaceMatchingOrchestrator()

        # Check system status
        system_status = orchestrator.get_system_status()
        index_status = system_status.get('index', {})

        if not index_status.get('ready', False):
            logger.error("❌ Search index not found. Please build an index first.")
            sys.exit(1)

        # Get index stats
        stats = orchestrator.get_index_statistics()
        logger.info(f"🔍 Search engine ready with {stats.get('total_faces', 0)} indexed faces")

        if stats.get('preprocessing_used'):
            logger.info("ℹ️ Preprocessing was used when building the index")

        # Perform search
        logger.info(f"🔍 Searching for similar faces to: {args.query}")
        logger.info(f"📊 Parameters: k={args.k}, threshold={args.threshold}")

        search_result = orchestrator.search_similar_faces(
            query_image_path=args.query,
            k=args.k,
            threshold=args.threshold,
            use_preprocessing=None  # Auto-detect from index
        )

        if not search_result['success']:
            logger.error(f"❌ Search failed: {search_result['error']}")
            sys.exit(1)

        results = search_result['results']

        # Display results
        logger.info("🔎 Search Results:")
        logger.info("=" * 60)

        found_matches = False
        for result in results:
            status = "✅" if result['within_threshold'] else "❌"
            logger.info(f"{status} #{result['rank']}: {result['filename']}")
            logger.info(f"   Distance: {result['distance']:.4f}")
            logger.info(f"   Similarity: {result['similarity_score']:.3f}")

            if result['within_threshold']:
                found_matches = True
                if result['path']:
                    logger.info(f"   Path: {result['path']}")
                else:
                    logger.info(f"   Relative Path: {result['relative_path']}")
            else:
                logger.info(f"   Status: Outside threshold")
            logger.info("")

        if not found_matches:
            logger.warning("🚫 No matches found within the specified threshold")
        else:
            logger.info(f"✅ Found {search_result['matches_found']} matches within threshold")
    
    except Exception as e:
        logger.error(f"❌ Search failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()