import os
import streamlit as st
import shutil
from datetime import datetime
import sys
import os
# Add the src directory to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, src_path)

from core.orchestrator import FaceMatchingOrchestrator
from utils.filesystem.operations import FileSystemOperations
from config.settings import Settings

def render_index_page():
    """Render the index management page"""

    settings = Settings()
    fs_ops = FileSystemOperations()

    st.markdown('<h1 class="main-header">📊 Index Management</h1>', unsafe_allow_html=True)

    # Initialize orchestrator
    orchestrator = FaceMatchingOrchestrator()

    # Display current index status
    display_index_status(orchestrator, fs_ops, settings)

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
            build_index_with_orchestrator(
                st.session_state.custom_folder_path,
                use_preprocessing
            )

    with col2:
        st.markdown("#### Advanced Options")
        max_workers = st.number_input("Max worker processes", min_value=1, max_value=8, value=4)
        force_rebuild = st.checkbox("Force rebuild (delete existing index)")

        if st.button("🔧 Advanced Build"):
            if force_rebuild:
                orchestrator = FaceMatchingOrchestrator()
                if st.button("⚠️ Confirm Delete Existing Index", type="secondary"):
                    result = orchestrator.delete_index()
                    if result['success']:
                        st.success("✅ Existing index deleted")
                        st.rerun()
                    else:
                        st.error(f"❌ {result['error']}")
            else:
                build_index_with_orchestrator(
                    st.session_state.custom_folder_path,
                    use_preprocessing
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
            orchestrator = FaceMatchingOrchestrator()
            display_detailed_index_stats(orchestrator)

    with col3:
        if st.button("🗑️ Delete Index", type="secondary"):
            orchestrator = FaceMatchingOrchestrator()
            if st.button("⚠️ Confirm Delete", type="secondary"):
                result = orchestrator.delete_index()
                if result['success']:
                    st.success("✅ Index deleted successfully")
                    st.rerun()
                else:
                    st.error(f"❌ {result['error']}")

def display_index_status(orchestrator, fs_ops, settings):
    """Display current index status"""

    st.markdown("## 📊 Current Status")

    # Get system statistics
    stats = get_system_stats(orchestrator, fs_ops, settings)

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
        index_info = orchestrator.get_index_stats()

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
def get_system_stats(_orchestrator, _fs_ops, _settings):
    """Get cached system statistics"""

    custom_folder = st.session_state.get('custom_folder_path', '')

    # Get system status from orchestrator
    system_status = _orchestrator.get_system_status()

    # Count source images
    source_images = 0
    if custom_folder and os.path.exists(custom_folder):
        source_images = _fs_ops.count_images_in_folder(custom_folder)

    # Count uploaded images
    uploaded_images = _fs_ops.count_images_in_folder(_settings.UPLOAD_DIR)

    # Check index status
    index_status = system_status.get('index', {})
    index_ready = index_status.get('ready', False)
    indexed_faces = index_status.get('stats', {}).get('total_faces', 0)

    return {
        'source_images': source_images,
        'uploaded_images': uploaded_images,
        'indexed_faces': indexed_faces,
        'index_ready': index_ready,
        'custom_folder': custom_folder
    }

def build_index_with_orchestrator(source_folder, use_preprocessing):
    """Interactive index building using orchestrator with progress display"""

    # Initialize orchestrator
    orchestrator = FaceMatchingOrchestrator()

    # Validate source folder
    validation_result = orchestrator.validate_source_folder(source_folder)
    if not validation_result['valid']:
        st.error(f"❌ {validation_result['error']}")
        return

    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text(f"Found {validation_result['image_count']} images")
        progress_bar.progress(10)

        status_text.text("Building index using orchestrator...")
        progress_bar.progress(50)

        # Build complete index using orchestrator
        result = orchestrator.build_complete_index(
            source_folder=source_folder,
            use_preprocessing=use_preprocessing
        )

        progress_bar.progress(100)

        if result['success']:
            st.success(f"✅ Index built successfully!")
            st.info(f"📊 Indexed {result['indexed_faces']} faces from {result['total_images']} images")
            if result['failed_extractions'] > 0:
                st.warning(f"⚠️ {result['failed_extractions']} images failed to process")

            # Clear cache and rerun to show updated stats
            st.cache_data.clear()
            st.rerun()
        else:
            st.error(f"❌ {result['error']}")

    except Exception as e:
        st.error(f"❌ Index building failed: {e}")
    finally:
        progress_bar.empty()
        status_text.empty()

def display_detailed_index_stats(orchestrator):
    """Display detailed index statistics"""

    stats = orchestrator.get_index_statistics()

    if 'error' in stats:
        st.warning(f"❌ {stats['error']}")
        return

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