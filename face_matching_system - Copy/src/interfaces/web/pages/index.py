import os
import streamlit as st
import shutil
from datetime import datetime
import sys
import os
# Add the src directory to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, src_path)

from core.preprocessing.preprocessor import ImagePreprocessor
from core.embedding.extractor import EmbeddingExtractor
from core.indexing.manager import IndexManager
from utils.filesystem.operations import FileSystemOperations
from config.settings import Settings

def render_index_page():
    """Render the index management page"""
    
    settings = Settings()
    fs_ops = FileSystemOperations()
    
    st.markdown('<h1 class="main-header">📊 Index Management</h1>', unsafe_allow_html=True)
    
    # Initialize components
    index_manager = IndexManager()
    
    # Display current index status
    display_index_status(index_manager, fs_ops, settings)
    
    st.markdown("---")
    
    # Index management operations
    st.markdown("## 🔧 Index Operations")
    
    # Source folder configuration
    st.markdown("### 📁 Source Folder Configuration")
    
    current_folder = st.session_state.get('custom_folder_path', '')
    new_folder = st.text_input(
        "Source folder path:",
        value=current_folder,
        help="Path to folder containing images (searches recursively in subfolders)"
    )
    
    if st.button("📂 Set Source Folder"):
        if os.path.exists(new_folder):
            st.session_state.custom_folder_path = new_folder
            st.success(f"✅ Source folder set: {new_folder}")
            st.rerun()
        else:
            st.error("❌ Invalid folder path")
    
    if st.session_state.get('custom_folder_path'):
        st.markdown(f'<div class="folder-path">📁 {st.session_state.custom_folder_path}</div>',
                   unsafe_allow_html=True)
        
        # Show folder statistics
        source_images = fs_ops.count_images_in_folder(st.session_state.custom_folder_path)
        st.info(f"Found {source_images} images in source folder")
    
    st.markdown("---")
    
    # Index building options
    st.markdown("### 🏗️ Build Index")
    
    if not st.session_state.get('custom_folder_path'):
        st.warning("⚠️ Please set a source folder first")
        return
    
    # Build options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Quick Build")
        use_preprocessing = st.checkbox("Enable preprocessing", value=True)
        
        if st.button("🚀 Build Index", type="primary"):
            build_index_interactive(
                st.session_state.custom_folder_path,
                use_preprocessing,
                settings,
                fs_ops
            )
    
    with col2:
        st.markdown("#### Advanced Options")
        max_workers = st.number_input("Max worker processes", min_value=1, max_value=8, value=4)
        force_rebuild = st.checkbox("Force rebuild (delete existing index)")
        
        if st.button("🔧 Advanced Build"):
            if force_rebuild and index_manager.index_exists():
                if st.button("⚠️ Confirm Delete Existing Index", type="secondary"):
                    index_manager.delete_index()
                    st.success("✅ Existing index deleted")
                    st.rerun()
            else:
                build_index_interactive(
                    st.session_state.custom_folder_path,
                    use_preprocessing,
                    settings,
                    fs_ops,
                    max_workers
                )
    
    st.markdown("---")
    
    # Index management
    st.markdown("### 🗂️ Index Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reload Index"):
            # Force reload of cached functions
            st.cache_data.clear()
            st.success("✅ Index reloaded")
            st.rerun()
    
    with col2:
        if st.button("📈 View Detailed Stats"):
            display_detailed_index_stats(index_manager)
    
    with col3:
        if st.button("🗑️ Delete Index", type="secondary"):
            if index_manager.index_exists():
                if st.button("⚠️ Confirm Delete", type="secondary"):
                    if index_manager.delete_index():
                        st.success("✅ Index deleted successfully")
                        st.rerun()
                    else:
                        st.error("❌ Failed to delete index")
            else:
                st.info("No index to delete")

def display_index_status(index_manager, fs_ops, settings):
    """Display current index status"""
    
    st.markdown("## 📊 Current Status")
    
    # Get system statistics
    stats = get_system_stats(index_manager, fs_ops, settings)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Source Images", stats['source_images'])
    with col2:
        st.metric("Uploaded Images", stats['uploaded_images'])
    with col3:
        st.metric("Indexed Faces", stats['indexed_faces'])
    with col4:
        index_status = "✅ Ready" if stats['index_ready'] else "❌ Not Ready"
        st.metric("Index Status", index_status)
    
    # Display detailed status
    if stats['index_ready']:
        index_info = index_manager.get_index_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Index Information")
            st.write(f"**Created:** {index_info.get('created_at', 'Unknown')}")
            st.write(f"**Source Folder:** {index_info.get('source_folder', 'Unknown')}")
            st.write(f"**Preprocessing Used:** {'Yes' if index_info.get('preprocessing_used') else 'No'}")
        
        with col2:
            st.markdown("#### Technical Details")
            st.write(f"**Model:** {index_info.get('model_used', 'Unknown')}")
            st.write(f"**Detector:** {index_info.get('detector_used', 'Unknown')}")
            st.write(f"**Embedding Dimension:** {index_info.get('embedding_dimension', 'Unknown')}")

@st.cache_data
def get_system_stats(_index_manager, _fs_ops, _settings):
    """Get cached system statistics"""
    
    custom_folder = st.session_state.get('custom_folder_path', '')
    
    # Count source images
    source_images = 0
    if custom_folder and os.path.exists(custom_folder):
        source_images = _fs_ops.count_images_in_folder(custom_folder)
    
    # Count uploaded images
    uploaded_images = _fs_ops.count_images_in_folder(_settings.UPLOAD_DIR)
    
    # Check index status
    index_ready = _index_manager.index_exists()
    indexed_faces = 0
    
    if index_ready:
        stats = _index_manager.get_index_stats()
        indexed_faces = stats.get('total_faces', 0)
    
    return {
        'source_images': source_images,
        'uploaded_images': uploaded_images,
        'indexed_faces': indexed_faces,
        'index_ready': index_ready,
        'custom_folder': custom_folder
    }

def build_index_interactive(source_folder, use_preprocessing, settings, fs_ops, max_workers=None):
    """Interactive index building with progress display"""
    
    # Validate source folder
    if not os.path.exists(source_folder):
        st.error(f"❌ Source folder not found: {source_folder}")
        return
    
    # Get image files
    image_files = fs_ops.get_all_images_from_folder(source_folder)
    if not image_files:
        st.error(f"❌ No images found in {source_folder}")
        return
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize components
        preprocessor = ImagePreprocessor() if use_preprocessing else None
        embedding_extractor = EmbeddingExtractor()
        index_manager = IndexManager()
        
        status_text.text(f"Found {len(image_files)} images")
        progress_bar.progress(10)
        
        # Preprocessing step
        processed_files = []
        if use_preprocessing and preprocessor:
            status_text.text("Preprocessing images...")
            
            # Clear preprocessed directory
            if os.path.exists(settings.PREPROCESSED_DIR):
                shutil.rmtree(settings.PREPROCESSED_DIR)
            os.makedirs(settings.PREPROCESSED_DIR, exist_ok=True)
            
            for i, (original_path, relative_path) in enumerate(image_files):
                # Create preprocessed path
                preprocessed_path = os.path.join(settings.PREPROCESSED_DIR, relative_path)
                preprocessed_dir = os.path.dirname(preprocessed_path)
                os.makedirs(preprocessed_dir, exist_ok=True)
                
                # Copy and preprocess
                shutil.copy2(original_path, preprocessed_path)
                if preprocessor.preprocess_image(preprocessed_path, preprocessed_path):
                    processed_files.append((preprocessed_path, relative_path))
                
                progress = 10 + int(30 * (i + 1) / len(image_files))
                progress_bar.progress(progress)
        else:
            # Use original files
            processed_files = image_files
            progress_bar.progress(40)
        
        if not processed_files:
            st.error("❌ No images available for processing")
            return
        
        # Extract embeddings
        status_text.text("Extracting face embeddings...")
        image_paths = [path for path, _ in processed_files]
        
        embeddings, successful_paths = embedding_extractor.extract_batch_embeddings(
            image_paths=image_paths,
            enforce_detection=True,
            max_workers=max_workers
        )
        
        progress_bar.progress(80)
        
        if not embeddings:
            st.error("❌ No face embeddings extracted")
            return
        
        # Create labels for successful extractions
        path_to_relative = {path: rel_path for path, rel_path in processed_files}
        labels = [path_to_relative[path] for path in successful_paths if path in path_to_relative]
        
        # Build index
        status_text.text("Building FAISS index...")
        
        additional_metadata = {
            "total_images": len(image_files),
            "failed_embeddings": len(image_files) - len(embeddings)
        }
        
        success = index_manager.build_index(
            embeddings=embeddings,
            labels=labels,
            source_folder=source_folder,
            preprocessing_used=use_preprocessing,
            additional_metadata=additional_metadata
        )
        
        progress_bar.progress(100)
        
        if success:
            st.success(f"✅ Index built successfully!")
            st.info(f"📊 Indexed {len(embeddings)} faces from {len(image_files)} images")
            if len(image_files) - len(embeddings) > 0:
                st.warning(f"⚠️ {len(image_files) - len(embeddings)} images failed to process")
            
            # Clear cache and rerun to show updated stats
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("❌ Failed to build index")
    
    except Exception as e:
        st.error(f"❌ Index building failed: {e}")
    finally:
        progress_bar.empty()
        status_text.empty()

def display_detailed_index_stats(index_manager):
    """Display detailed index statistics"""
    
    if not index_manager.index_exists():
        st.warning("❌ No index found")
        return
    
    stats = index_manager.get_index_stats()
    
    st.markdown("### 📈 Detailed Index Statistics")
    
    # Basic info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Basic Information")
        st.write(f"**Total Faces:** {stats.get('total_faces', 0)}")
        st.write(f"**Total Images:** {stats.get('total_images', 0)}")
        st.write(f"**Failed Embeddings:** {stats.get('failed_embeddings', 0)}")
        st.write(f"**Success Rate:** {(stats.get('successful_embeddings', 0) / stats.get('total_images', 1) * 100):.1f}%")
    
    with col2:
        st.markdown("#### Technical Details")
        st.write(f"**Embedding Dimension:** {stats.get('embedding_dimension', 0)}")
        st.write(f"**Model Used:** {stats.get('model_used', 'Unknown')}")
        st.write(f"**Detector Used:** {stats.get('detector_used', 'Unknown')}")
        st.write(f"**Index Size:** {stats.get('index_size', 0)}")
    
    # Metadata
    st.markdown("#### Metadata")
    st.write(f"**Source Folder:** {stats.get('source_folder', 'Unknown')}")
    st.write(f"**Created At:** {stats.get('created_at', 'Unknown')}")
    st.write(f"**Preprocessing Used:** {'Yes' if stats.get('preprocessing_used') else 'No'}")
    
    # File sizes
    if 'index_file_size' in stats:
        st.write(f"**Index File Size:** {stats['index_file_size'] / 1024 / 1024:.2f} MB")
