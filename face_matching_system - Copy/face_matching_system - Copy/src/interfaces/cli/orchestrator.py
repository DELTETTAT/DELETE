#!/usr/bin/env python3
"""
Interactive orchestrator for the face matching system.
Provides a menu-driven interface for all system operations.
"""

import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.preprocessing.preprocessor import ImagePreprocessor
from core.embedding.extractor import EmbeddingExtractor
from core.indexing.manager import IndexManager
from core.search.engine import SearchEngine
from utils.filesystem.operations import FileSystemOperations
from utils.logging.logger import setup_logging
from config.settings import Settings

class InteractiveFaceMatchingOrchestrator:
    """Interactive orchestrator for the face matching system"""
    
    def __init__(self):
        self.settings = Settings()
        self.source_folder = ""
        
        # Initialize components
        self.fs_ops = FileSystemOperations()
        self.preprocessor = ImagePreprocessor()
        self.embedding_extractor = EmbeddingExtractor()
        self.index_manager = IndexManager()
        self.search_engine = SearchEngine()
        
        # Setup logging
        setup_logging()
        
        self.logger = self.settings.logger
    
    def _print_header(self):
        """Print application header"""
        print("\n" + "="*60)
        print("🎯 FACE MATCHING SYSTEM")
        print("="*60)
        if self.source_folder:
            print(f"📁 Source Folder: {self.source_folder}")
    
    def _print_menu(self):
        """Print main menu options"""
        print("\n📋 Choose an option:")
        print("-" * 30)
        print("1. 🔄 Rebuild Complete Index (Preprocess + Build)")
        print("2. 📊 Build Index from Existing Images")
        print("3. 🔍 Search for Face")
        print("4. 📈 View Index Statistics")
        print("5. 🖼️  Preprocess Images Only")
        print("6. 📁 Check Directory Status")
        print("7. 📂 Set Source Folder")
        print("8. ❌ Exit")
        print("-" * 30)
    
    def _get_user_choice(self) -> int:
        """Get user menu choice"""
        while True:
            try:
                choice = int(input("Enter your choice (1-8): ").strip())
                if 1 <= choice <= 8:
                    return choice
                else:
                    print("❌ Please enter a number between 1 and 8")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def _set_source_folder(self):
        """Set the source folder for images"""
        print("\n📂 SET SOURCE FOLDER")
        print("-" * 30)
        folder = input("Enter path to folder containing images: ").strip()
        
        if not folder:
            print("❌ No path entered")
            return
        
        if not os.path.exists(folder):
            print("❌ Path does not exist")
            return
        
        self.source_folder = folder
        print(f"✅ Source folder set to: {folder}")
    
    def _check_directory_status(self):
        """Check status of all directories"""
        print("\n📁 Directory Status:")
        print("-" * 40)
        
        # Check source folder
        if self.source_folder:
            source_images = self.fs_ops.count_images_in_folder(self.source_folder)
            print(f"Source Folder: {source_images} images in '{self.source_folder}'")
        else:
            print("Source Folder: ❌ Not set")
        
        # Check uploads directory
        upload_images = self.fs_ops.count_images_in_folder(self.settings.UPLOAD_DIR)
        print(f"Uploads: {upload_images} images in '{self.settings.UPLOAD_DIR}'")
        
        # Check index status
        index_ready = self.index_manager.index_exists()
        print(f"\n{'✅' if index_ready else '❌'} Index Ready: {'Yes' if index_ready else 'No'}")
        
        if index_ready:
            stats = self.index_manager.get_index_stats()
            print(f"Indexed Faces: {stats.get('total_faces', 0)}")
        
        return index_ready
    
    def _rebuild_complete_index(self):
        """Interactive complete rebuild"""
        print("\n🔄 COMPLETE INDEX REBUILD")
        print("-" * 30)
        
        if not self.source_folder:
            print("❌ Source folder not set")
            return
        
        source_images = self.fs_ops.count_images_in_folder(self.source_folder)
        if source_images == 0:
            print(f"❌ No images found in '{self.source_folder}'")
            return
        
        print(f"Found {source_images} images to process")
        confirm = input("This will preprocess all images and rebuild the index. Continue? (y/n): ")
        
        if confirm.lower() != 'y':
            print("❌ Operation cancelled")
            return
        
        try:
            # Get all image files
            image_files = self.fs_ops.get_all_images_from_folder(self.source_folder)
            
            print("\n🔄 Step 1: Preprocessing images...")
            processed_files = []
            
            # Clear preprocessed directory
            import shutil
            if os.path.exists(self.settings.PREPROCESSED_DIR):
                shutil.rmtree(self.settings.PREPROCESSED_DIR)
            os.makedirs(self.settings.PREPROCESSED_DIR, exist_ok=True)
            
            # Preprocess images
            for i, (original_path, relative_path) in enumerate(image_files):
                preprocessed_path = os.path.join(self.settings.PREPROCESSED_DIR, relative_path)
                preprocessed_dir = os.path.dirname(preprocessed_path)
                os.makedirs(preprocessed_dir, exist_ok=True)
                
                # Copy and preprocess
                shutil.copy2(original_path, preprocessed_path)
                if self.preprocessor.preprocess_image(preprocessed_path, preprocessed_path):
                    processed_files.append((preprocessed_path, relative_path))
                
                if i % 10 == 0 or i == len(image_files) - 1:
                    print(f"   Processed: {i+1}/{len(image_files)}")
            
            if not processed_files:
                print("❌ No images successfully preprocessed")
                return
            
            print("\n🔄 Step 2: Extracting embeddings...")
            image_paths = [path for path, _ in processed_files]
            embeddings, successful_paths = self.embedding_extractor.extract_batch_embeddings(
                image_paths=image_paths,
                enforce_detection=True
            )
            
            if not embeddings:
                print("❌ No embeddings extracted")
                return
            
            print("\n🔄 Step 3: Building FAISS index...")
            
            # Create labels for successful extractions
            path_to_relative = {path: rel_path for path, rel_path in processed_files}
            labels = [path_to_relative[path] for path in successful_paths if path in path_to_relative]
            
            success = self.index_manager.build_index(
                embeddings=embeddings,
                labels=labels,
                source_folder=self.source_folder,
                preprocessing_used=True,
                additional_metadata={
                    "total_images": len(image_files),
                    "failed_embeddings": len(image_files) - len(embeddings)
                }
            )
            
            if success:
                print("\n🎉 Complete rebuild successful!")
                print(f"✅ Indexed {len(embeddings)} faces from {len(image_files)} images")
            else:
                print("❌ Index building failed")
        
        except Exception as e:
            print(f"❌ Rebuild failed: {e}")
    
    def _build_index_only(self):
        """Interactive index building"""
        print("\n📊 BUILD INDEX FROM EXISTING IMAGES")
        print("-" * 40)
        
        if not self.source_folder:
            print("❌ Source folder not set")
            return
        
        source_images = self.fs_ops.count_images_in_folder(self.source_folder)
        if source_images == 0:
            print(f"❌ No images found in '{self.source_folder}'")
            return
        
        print(f"Found {source_images} images")
        use_preprocessing = input("Use preprocessing? (y/n): ").lower() == 'y'
        
        try:
            # Get all image files
            image_files = self.fs_ops.get_all_images_from_folder(self.source_folder)
            
            # Determine which files to use
            if use_preprocessing:
                print("\n🔄 Using preprocessed images (if available)...")
                # Check if preprocessed images exist
                processed_files = []
                for original_path, relative_path in image_files:
                    preprocessed_path = os.path.join(self.settings.PREPROCESSED_DIR, relative_path)
                    if os.path.exists(preprocessed_path):
                        processed_files.append((preprocessed_path, relative_path))
                    else:
                        processed_files.append((original_path, relative_path))
            else:
                processed_files = image_files
            
            print("\n🔄 Building FAISS index...")
            
            # Extract embeddings
            image_paths = [path for path, _ in processed_files]
            embeddings, successful_paths = self.embedding_extractor.extract_batch_embeddings(
                image_paths=image_paths,
                enforce_detection=True
            )
            
            if not embeddings:
                print("❌ No embeddings extracted")
                return
            
            # Create labels
            path_to_relative = {path: rel_path for path, rel_path in processed_files}
            labels = [path_to_relative[path] for path in successful_paths if path in path_to_relative]
            
            # Build index
            success = self.index_manager.build_index(
                embeddings=embeddings,
                labels=labels,
                source_folder=self.source_folder,
                preprocessing_used=use_preprocessing,
                additional_metadata={
                    "total_images": len(image_files),
                    "failed_embeddings": len(image_files) - len(embeddings)
                }
            )
            
            if success:
                print("✅ Index built successfully!")
                print(f"📊 Indexed {len(embeddings)} faces from {len(image_files)} images")
            else:
                print("❌ Index building failed")
        
        except Exception as e:
            print(f"❌ Index building failed: {e}")
    
    def _search_face_interactive(self):
        """Interactive face search"""
        print("\n🔍 FACE SEARCH")
        print("-" * 20)
        
        # Check if index exists
        if not self.search_engine.is_ready():
            print("❌ No index found. Please build an index first.")
            return
        
        # List available images in uploads
        upload_images = self.fs_ops.get_image_files_in_folder(self.settings.UPLOAD_DIR)
        
        if not upload_images:
            print(f"❌ No images found in '{self.settings.UPLOAD_DIR}' directory")
            print("Please add images to search and try again.")
            return
        
        print(f"\n📁 Available images in '{self.settings.UPLOAD_DIR}':")
        for i, img in enumerate(upload_images, 1):
            print(f"{i}. {img}")
        
        # Get user choice
        while True:
            try:
                choice = int(input(f"\nSelect image to search (1-{len(upload_images)}): "))
                if 1 <= choice <= len(upload_images):
                    selected_image = upload_images[choice - 1]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(upload_images)}")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Get search parameters
        print("\n⚙️ Search Parameters:")
        
        try:
            k = int(input("Number of results to return (default 3): ").strip() or "3")
        except ValueError:
            k = 3
        
        try:
            threshold = float(input("Distance threshold (default 450.0): ").strip() or "450.0")
        except ValueError:
            threshold = 450.0
        
        # Perform search
        query_path = os.path.join(self.settings.UPLOAD_DIR, selected_image)
        print(f"\n🔍 Searching for: {selected_image}")
        print(f"📊 Parameters: k={k}, threshold={threshold}")
        print("-" * 50)
        
        results = self.search_engine.search_similar_faces(
            query_path=query_path,
            k=k,
            threshold=threshold
        )
        
        if not results:
            print("❌ Search failed or no results found")
            return
        
        # Display results
        print("\n📋 Search Results:")
        print("=" * 50)
        
        found_matches = False
        for result in results:
            status = "✅" if result['within_threshold'] else "❌"
            print(f"{status} #{result['rank']}: {result['filename']}")
            print(f"   Distance: {result['distance']:.2f}")
            print(f"   Similarity: {result['similarity_score']:.3f}")
            
            if result['within_threshold']:
                found_matches = True
                if result['path']:
                    print(f"   Path: {result['path']}")
                else:
                    print(f"   Relative Path: {result['relative_path']}")
            else:
                print(f"   Status: Outside threshold")
            print()
        
        if not found_matches:
            print("🚫 No matches found within the specified threshold")
    
    def _view_statistics(self):
        """View index statistics"""
        print("\n📈 INDEX STATISTICS")
        print("-" * 25)
        
        stats = self.index_manager.get_index_stats()
        
        if "error" in stats:
            print(f"❌ {stats['error']}")
            return
        
        print(f"Source Folder: {stats.get('source_folder', 'N/A')}")
        print(f"Total Images: {stats.get('total_images', 0)}")
        print(f"Indexed Faces: {stats['total_faces']}")
        print(f"Failed: {stats.get('failed_embeddings', 0)}")
        print(f"Preprocessing: {'Yes' if stats.get('preprocessing_used', False) else 'No'}")
        print(f"Created: {stats.get('created_at', 'Unknown')}")
        print(f"Embedding Dimension: {stats.get('embedding_dimension', 'Unknown')}")
        print(f"Model Used: {stats.get('model_used', 'Unknown')}")
    
    def _preprocess_only(self):
        """Interactive preprocessing only"""
        print("\n🖼️ PREPROCESS IMAGES ONLY")
        print("-" * 30)
        
        if not self.source_folder:
            print("❌ Source folder not set")
            return
        
        source_images = self.fs_ops.count_images_in_folder(self.source_folder)
        if source_images == 0:
            print(f"❌ No images found in '{self.source_folder}'")
            return
        
        print(f"Found {source_images} images to preprocess")
        confirm = input("Preprocess all images? (y/n): ")
        
        if confirm.lower() != 'y':
            print("❌ Operation cancelled")
            return
        
        try:
            # Get all image files
            image_files = self.fs_ops.get_all_images_from_folder(self.source_folder)
            
            print("\n🔄 Preprocessing images...")
            
            # Clear preprocessed directory
            import shutil
            if os.path.exists(self.settings.PREPROCESSED_DIR):
                shutil.rmtree(self.settings.PREPROCESSED_DIR)
            os.makedirs(self.settings.PREPROCESSED_DIR, exist_ok=True)
            
            successful_count = 0
            
            for i, (original_path, relative_path) in enumerate(image_files):
                preprocessed_path = os.path.join(self.settings.PREPROCESSED_DIR, relative_path)
                preprocessed_dir = os.path.dirname(preprocessed_path)
                os.makedirs(preprocessed_dir, exist_ok=True)
                
                # Copy and preprocess
                shutil.copy2(original_path, preprocessed_path)
                if self.preprocessor.preprocess_image(preprocessed_path, preprocessed_path):
                    successful_count += 1
                
                if i % 10 == 0 or i == len(image_files) - 1:
                    print(f"   Processed: {i+1}/{len(image_files)}")
            
            print(f"✅ Preprocessing completed! ({successful_count}/{len(image_files)} successful)")
        
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
    
    def run_interactive_menu(self):
        """Main interactive menu loop"""
        while True:
            try:
                self._print_header()
                self._print_menu()
                
                choice = self._get_user_choice()
                
                if choice == 1:
                    self._rebuild_complete_index()
                elif choice == 2:
                    self._build_index_only()
                elif choice == 3:
                    self._search_face_interactive()
                elif choice == 4:
                    self._view_statistics()
                elif choice == 5:
                    self._preprocess_only()
                elif choice == 6:
                    self._check_directory_status()
                elif choice == 7:
                    self._set_source_folder()
                elif choice == 8:
                    print("\n👋 Goodbye!")
                    break
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                input("Press Enter to continue...")

def main():
    """Main entry point for orchestrator CLI"""
    orchestrator = InteractiveFaceMatchingOrchestrator()
    orchestrator.run_interactive_menu()

if __name__ == "__main__":
    main()
