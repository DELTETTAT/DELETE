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
    """Interactive index building using orchestrator with enhanced real-time progress display"""

    # Initialize orchestrator
    orchestrator = FaceMatchingOrchestrator()

    # Validate source folder
    validation_result = orchestrator.validate_source_folder(source_folder)
    if not validation_result['valid']:
        st.error(f"❌ {validation_result['error']}")
        return

    # Create enhanced progress containers with better layout
    progress_container = st.container()
    with progress_container:
        st.markdown("### 🚀 Building Index Progress")

        # Overall progress section
        st.markdown("#### Overall Progress")
        overall_progress = st.progress(0)
        overall_status = st.empty()

        # Current chunk progress section  
        st.markdown("#### Current Chunk Progress")
        chunk_progress = st.progress(0)
        chunk_status = st.empty()

        # Detailed status section
        st.markdown("#### Processing Details")
        details_text = st.empty()

        # Performance metrics
        metrics_container = st.empty()

    # Tracking variables for performance metrics
    import time
    start_time = time.time()
    last_update_time = start_time
    processed_per_second = 0

    def enhanced_progress_callback(chunk_idx, total_chunks, chunk_processed, chunk_total, overall_processed, overall_total):
        nonlocal last_update_time, processed_per_second

        current_time = time.time()
        elapsed = current_time - start_time

        # Calculate processing rate
        if elapsed > 0:
            processed_per_second = overall_processed / elapsed

        # Calculate progress percentages
        overall_pct = min(95, int((overall_processed / overall_total) * 100)) if overall_total > 0 else 0
        chunk_pct = int((chunk_processed / chunk_total) * 100) if chunk_total > 0 else 0

        # Update progress bars
        overall_progress.progress(overall_pct)
        chunk_progress.progress(chunk_pct)

        # Enhanced status messages
        if chunk_processed == 0:
            phase = "🚀 Initializing"
        elif chunk_processed == chunk_total:
            phase = "✅ Completed"
        elif chunk_processed < chunk_total // 3:
            phase = "🔄 Processing"
        elif chunk_processed < chunk_total * 2 // 3:
            phase = "⚡ Accelerating"
        else:
            phase = "🏁 Finalizing"

        # Update status displays
        overall_status.text(f"Overall: {overall_processed}/{overall_total} images ({overall_pct}%)")
        chunk_status.text(f"{phase} chunk {chunk_idx + 1}/{total_chunks} - {chunk_processed}/{chunk_total} ({chunk_pct}%)")

        # Enhanced details with timing and performance
        remaining = overall_total - overall_processed
        eta = remaining / processed_per_second if processed_per_second > 0 else 0

        details_text.text(
            f"📊 Chunk {chunk_idx + 1}/{total_chunks} | "
            f"⏱️ Elapsed: {elapsed:.1f}s | "
            f"⚡ Rate: {processed_per_second:.1f} imgs/sec | "
            f"🔮 ETA: {eta:.1f}s"
        )

        # Performance metrics in columns
        with metrics_container.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Processed", f"{overall_processed}", f"+{chunk_processed}")
            with col2:
                st.metric("Rate", f"{processed_per_second:.1f}/sec")
            with col3:
                st.metric("Remaining", f"{remaining}")
            with col4:
                st.metric("ETA", f"{eta:.0f}s")

        last_update_time = current_time

    try:
        total_images = validation_result['image_count']
        overall_status.text(f"🔍 Found {total_images} images to process")
        overall_progress.progress(5)

        # Calculate chunk size and number of chunks
        chunk_size = 100  # Same as in orchestrator
        total_chunks = (total_images + chunk_size - 1) // chunk_size

        chunk_status.text(f"📊 Will process {total_images} images in {total_chunks} chunks")
        details_text.text("Initializing processing pipeline...")
        overall_progress.progress(10)

        # Call orchestrator with progress callback
        if hasattr(orchestrator, 'build_complete_index_with_progress'):
            result = orchestrator.build_complete_index_with_progress(
                source_folder=source_folder,
                use_preprocessing=use_preprocessing,
                max_workers=4,  # Optimal for most systems
                chunk_size=50,  # Smaller chunks for better progress granularity
                progress_callback=enhanced_progress_callback
            )
        else:
            # Fallback to standard method if progress method not available
            result = orchestrator.build_complete_index(
                source_folder=source_folder,
                use_preprocessing=use_preprocessing,
                max_workers=4,
                chunk_size=50
            )

        # Final progress update
        overall_progress.progress(100)
        chunk_progress.progress(100)

        total_time = time.time() - start_time

        if result['success']:
            overall_status.text("✅ Index building completed successfully!")
            chunk_status.text("🎉 All chunks processed!")
            details_text.text(f"🏆 Completed in {total_time:.1f}s - Indexed {result['indexed_faces']} faces from {result['total_images']} images")

            # Final metrics
            with metrics_container.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Processed", f"{result['indexed_faces']}")
                with col2:
                    st.metric("Success Rate", f"{(result['indexed_faces']/result['total_images']*100):.1f}%")
                with col3:
                    st.metric("Total Time", f"{total_time:.1f}s")
                with col4:
                    st.metric("Avg Rate", f"{result['indexed_faces']/total_time:.1f}/sec")

            st.success(f"✅ Index built successfully!")
            st.info(f"📊 Indexed {result['indexed_faces']} faces from {result['total_images']} images in {total_time:.1f} seconds")
            if result['failed_extractions'] > 0:
                st.warning(f"⚠️ {result['failed_extractions']} images failed to process")

            # Clear cache and rerun to show updated stats
            st.cache_data.clear()
            st.rerun()
        else:
            st.error(f"❌ {result['error']}")

    except Exception as e:
        st.error(f"❌ Index building failed: {e}")
        import traceback
        st.error(f"Debug info: {traceback.format_exc()}")
    finally:
        # Keep progress display visible for a moment to show completion
        import time
        time.sleep(3)
        # Clean up progress elements
        progress_container.empty()

def update_progress_display(overall_progress, chunk_progress, status_text, details_text,
                          chunk_idx, total_chunks, chunk_processed, chunk_total, 
                          overall_processed, overall_total):
    """Update progress display with real-time information and enhanced feedback"""

    # Calculate progress percentages
    overall_pct = min(95, int((overall_processed / overall_total) * 100)) if overall_total > 0 else 0
    chunk_pct = int((chunk_processed / chunk_total) * 100) if chunk_total > 0 else 0

    # Update progress bars with dynamic colors
    overall_progress.progress(overall_pct)
    chunk_progress.progress(chunk_pct)

    # Enhanced status messages with emojis and timing
    if chunk_processed == 0:
        phase = "🚀 Starting"
    elif chunk_processed < chunk_total // 2:
        phase = "🔄 Processing"
    elif chunk_processed < chunk_total:
        phase = "⚡ Accelerating"
    else:
        phase = "✅ Completing"

    # Update status with more detailed information
    status_text.text(f"{phase} chunk {chunk_idx + 1}/{total_chunks} ({chunk_pct}% complete)")

    # Enhanced details with rates and ETA estimation
    if overall_processed > 0 and overall_total > 0:
        completion_rate = (overall_processed / overall_total) * 100
        remaining = overall_total - overall_processed

        details_text.text(
            f"📊 Overall: {overall_processed}/{overall_total} ({completion_rate:.1f}%) | "
            f"Current chunk: {chunk_processed}/{chunk_total} | "
            f"Remaining: {remaining} images"
        )
    else:
        details_text.text(f"📈 Current chunk: {chunk_processed}/{chunk_total}")

    # Force UI update
    st.empty()

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