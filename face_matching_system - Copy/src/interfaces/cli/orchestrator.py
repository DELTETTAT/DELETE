#!/usr/bin/env python3
"""
Interactive CLI orchestrator for face matching system.
"""
import os
import sys
import logging

# Add src to path for imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, src_path)

from core.orchestrator import FaceMatchingOrchestrator
from utils.filesystem.operations import FileSystemOperations

class CLIOrchestrator:
    """Interactive CLI for face matching system"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.orchestrator = FaceMatchingOrchestrator()
        self.fs_ops = FileSystemOperations()
        self.source_folder = None

    def _print_header(self):
        """Print application header"""
        print("\n" + "="*60)
        print("🔍 FACE MATCHING SYSTEM - CLI ORCHESTRATOR")
        print("="*60)
        if self.source_folder:
            print(f"📁 Source: {self.source_folder}")
        print()

    def _print_menu(self):
        """Print main menu"""
        print("Available Operations:")
        print("1. 🔄 Complete Rebuild (preprocess + index)")
        print("2. 📊 Build Index Only")
        print("3. 🔍 Search Face")
        print("4. 📈 View Statistics")
        print("5. 🖼️ Preprocess Only")
        print("6. 📋 Check Status")
        print("7. 📁 Set Source Folder")
        print("8. 🚪 Exit")
        print()

    def _get_user_choice(self):
        """Get and validate user choice"""
        try:
            choice = int(input("Enter your choice (1-8): "))
            if 1 <= choice <= 8:
                return choice
            else:
                print("❌ Invalid choice. Please enter 1-8.")
                return self._get_user_choice()
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
            return self._get_user_choice()

    def _set_source_folder(self):
        """Set source folder interactively"""
        print("\n📁 SET SOURCE FOLDER")
        print("-" * 25)

        current = self.source_folder or "Not set"
        print(f"Current folder: {current}")

        new_folder = input("Enter new source folder path: ").strip()

        if not new_folder:
            print("❌ No folder specified")
            return

        if not os.path.exists(new_folder):
            print(f"❌ Folder '{new_folder}' does not exist")
            return

        if not os.path.isdir(new_folder):
            print(f"❌ '{new_folder}' is not a directory")
            return

        self.source_folder = new_folder
        print(f"✅ Source folder set: {new_folder}")

    def _check_directory_status(self):
        """Check and display directory status"""
        print("\n📋 DIRECTORY STATUS")
        print("-" * 25)

        # Source folder
        if self.source_folder:
            source_images = self.fs_ops.count_images_in_folder(self.source_folder)
            print(f"Source: {source_images} images in '{self.source_folder}'")
        else:
            print("Source: Not set")

        # Check index status
        system_status = self.orchestrator.get_system_status()
        index_status = system_status.get('index', {})
        index_ready = index_status.get('ready', False)

        print(f"{'✅' if index_ready else '❌'} Index Ready: {'Yes' if index_ready else 'No'}")

        if index_ready:
            stats = self.orchestrator.get_index_stats()
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
            print("\n🔄 Building complete index...")
            result = self.orchestrator.build_complete_index(
                source_folder=self.source_folder,
                use_preprocessing=True
            )

            if result['success']:
                print("\n🎉 Complete rebuild successful!")
                print(f"✅ Indexed {result['indexed_faces']} faces from {result['total_images']} images")
                if result['failed_extractions'] > 0:
                    print(f"⚠️ Failed: {result['failed_extractions']} images")
            else:
                print(f"❌ {result['error']}")

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
            result = self.orchestrator.build_complete_index(
                source_folder=self.source_folder,
                use_preprocessing=use_preprocessing
            )

            if result['success']:
                print("✅ Index built successfully!")
                print(f"📊 Indexed {result['indexed_faces']} faces from {result['total_images']} images")
                if result['failed_extractions'] > 0:
                    print(f"⚠️ Failed: {result['failed_extractions']} images")
            else:
                print(f"❌ {result['error']}")

        except Exception as e:
            print(f"❌ Index building failed: {e}")

    def _search_face_interactive(self):
        """Interactive face search"""
        print("\n🔍 FACE SEARCH")
        print("-" * 15)

        # Check if index exists
        if not self.orchestrator.index_manager.index_exists():
            print("❌ No index found. Please build an index first.")
            return

        query_path = input("Enter path to query image: ").strip()

        if not os.path.exists(query_path):
            print(f"❌ File '{query_path}' not found")
            return

        try:
            results = self.orchestrator.search_face(query_path)

            if results['success']:
                matches = results['matches']
                print(f"\n✅ Found {len(matches)} matches:")

                for i, match in enumerate(matches[:5], 1):  # Show top 5
                    print(f"{i}. {match['path']} (distance: {match['distance']:.2f})")
            else:
                print(f"❌ Search failed: {results['error']}")

        except Exception as e:
            print(f"❌ Search failed: {e}")

    def _view_statistics(self):
        """View system statistics"""
        print("\n📈 SYSTEM STATISTICS")
        print("-" * 25)

        try:
            stats = self.orchestrator.get_index_stats()

            if 'error' in stats:
                print(f"❌ {stats['error']}")
                return

            print(f"Total Faces: {stats.get('total_faces', 0)}")
            print(f"Total Images: {stats.get('total_images', 0)}")
            print(f"Embedding Dimension: {stats.get('embedding_dimension', 0)}")
            print(f"Model Used: {stats.get('model_used', 'Unknown')}")
            print(f"Created: {stats.get('created_at', 'Unknown')}")
            print(f"Source Folder: {stats.get('source_folder', 'Unknown')}")

        except Exception as e:
            print(f"❌ Failed to get statistics: {e}")

    def _preprocess_only(self):
        """Preprocessing only operation"""
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
        confirm = input("Continue with preprocessing? (y/n): ")

        if confirm.lower() != 'y':
            print("❌ Operation cancelled")
            return

        try:
            image_files = self.fs_ops.get_all_images_from_folder(self.source_folder)
            processed_files = self.orchestrator.batch_processor.preprocess_batch(image_files)
            successful_count = len(processed_files)

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
                print(f"\n❌ Unexpected error: {e}")
                input("Press Enter to continue...")

def main():
    """Main entry point"""
    cli = CLIOrchestrator()
    cli.run_interactive_menu()

if __name__ == "__main__":
    main()