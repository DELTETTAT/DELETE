import os
import streamlit as st
import shutil
import time
from datetime import datetime
from typing import List
import sys

# Add the src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', '..', '..')
src_path = os.path.normpath(src_path)

if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from core.orchestrator import FaceMatchingOrchestrator
    from utils.filesystem.operations import FileSystemOperations
    from config.settings import Settings
except ImportError as e:
    st.error(f"❌ Failed to import required modules: {e}")
    st.info(
        "Please ensure all dependencies are installed and the project structure is correct."
    )
    st.stop()


def render_index_page():
    """Render the index management page with Netflix-style UI"""

    settings = Settings()
    fs_ops = FileSystemOperations()

    # Page header
    st.markdown('<div class="section-header">📊 Index Management Center</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Build, manage, and monitor your face recognition index with professional tools."
    )

    # Initialize orchestrator
    orchestrator = FaceMatchingOrchestrator()

    # Create main tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🏗️ Build Index", "➕ Add to Index", "📊 Index Status", "⚙️ Settings"])

    with tab1:
        render_build_index_tab(orchestrator, fs_ops, settings)

    with tab2:
        render_add_to_index_tab(orchestrator, fs_ops, settings)

    with tab3:
        render_index_status_tab(orchestrator, fs_ops, settings)

    with tab4:
        render_settings_tab(orchestrator, fs_ops, settings)


def perform_build_operation(orchestrator, folder_path, use_preprocessing,
                            max_workers, chunk_size, force_rebuild,
                            build_type):
    """Perform the actual build operation with progress tracking"""
    try:
        # Delete existing index if force rebuild is enabled
        if force_rebuild:
            with st.spinner("🗑️ Deleting existing index..."):
                orchestrator.delete_index()
                st.success("✅ Existing index deleted")

        # Create progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(data, total=None):
            # Handle multiple callback formats
            try:
                if isinstance(data, dict):
                    # New format: dictionary with stage information
                    stage = data.get('stage', 'processing')
                    progress = data.get('progress', 0)
                    total_items = data.get('total', 1)

                    if stage == 'extraction':
                        percentage = (progress / total_items) if total_items > 0 else 0
                        progress_bar.progress(min(percentage, 0.8))  # Reserve 20% for indexing
                        status_text.text(f"🔍 Extracting embeddings: {progress}/{total_items}")
                    elif stage == 'indexing':
                        progress_bar.progress(0.9)
                        status_text.text("📊 Building FAISS index...")
                    elif stage == 'complete':
                        progress_bar.progress(1.0)
                        status_text.text("✅ Build completed!")
                elif total is not None:
                    # Legacy format: (processed, total) as separate arguments
                    processed = data
                    percentage = (processed / total) if total > 0 else 0
                    progress_bar.progress(min(percentage, 0.95))
                    status_text.text(f"🔍 Processing images: {processed}/{total}")
                else:
                    # Single argument format - assume it's processed count
                    progress_bar.progress(0.5)  # Show some progress
                    status_text.text(f"🔍 Processing images: {data}")
            except Exception as e:
                # Fallback - just show something is happening
                progress_bar.progress(0.3)
                status_text.text("🔍 Processing images...")

        status_text.text(f"🚀 Starting {build_type}...")

        # Start the build process
        result = orchestrator.build_complete_index_with_progress(
            source_folder=folder_path,
            use_preprocessing=use_preprocessing,
            max_workers=max_workers,
            chunk_size=chunk_size,
            progress_callback=progress_callback)

        if result.get('success', False):
            st.success(f"🎉 {build_type} completed successfully!")

            # Display results
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Images", result.get('total_images', 0))
            with col2:
                st.metric("Faces Indexed", result.get('faces_indexed', 0))
            with col3:
                st.metric("Failed Extractions",
                          result.get('failed_extractions', 0))
            with col4:
                preprocessing_text = "Yes" if result.get(
                    'preprocessing_used', False) else "No"
                st.metric("Preprocessing", preprocessing_text)

            # Show index statistics
            if 'index_stats' in result:
                stats = result['index_stats']
                st.markdown("### 📊 Index Statistics")
                st.json(stats)

        else:
            st.error(
                f"❌ {build_type} failed: {result.get('error', 'Unknown error')}"
            )

    except Exception as e:
        st.error(f"❌ Build operation failed: {str(e)}")


def render_build_index_tab(orchestrator, fs_ops, settings):
    """Render the build index tab"""

    st.markdown("### 🏗️ Build Complete Index")
    st.markdown(
        "Create a new face recognition index from your image collection.")

    # Source folder selection with file browser
    col1, col2 = st.columns([3, 1])

    with col1:
        current_folder = st.session_state.get('custom_folder_path', '')
        new_folder = st.text_input(
            "📁 Source Folder Path",
            value=current_folder,
            help="Enter the full path to your image folder",
            placeholder="/path/to/your/images")

    with col2:
        if st.button("📂 Browse", help="Set the source folder path"):
            if new_folder and os.path.exists(new_folder):
                st.session_state.custom_folder_path = new_folder
                st.success(f"✅ Folder set: {os.path.basename(new_folder)}")
                st.rerun()
            elif new_folder:
                st.error("❌ Folder not found")
            else:
                st.warning("⚠️ Please enter a folder path")

    # Display current folder info
    if st.session_state.get('custom_folder_path'):
        folder_path = st.session_state.custom_folder_path
        st.markdown(
            f'<div class="folder-path">📁 Current Folder: {folder_path}</div>',
            unsafe_allow_html=True)

        # Folder statistics
        image_count = fs_ops.count_images_in_folder(folder_path)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                '<div class="metric-container"><div class="metric-value">{}</div><div class="metric-label">Images Found</div></div>'
                .format(image_count),
                unsafe_allow_html=True)
        with col2:
            folder_size = get_folder_size(folder_path)
            st.markdown(
                '<div class="metric-container"><div class="metric-value">{:.1f} MB</div><div class="metric-label">Folder Size</div></div>'
                .format(folder_size),
                unsafe_allow_html=True)
        with col3:
            subfolders = count_subfolders(folder_path)
            st.markdown(
                '<div class="metric-container"><div class="metric-value">{}</div><div class="metric-label">Subfolders</div></div>'
                .format(subfolders),
                unsafe_allow_html=True)

        if image_count > 0:
            # Build options
            st.markdown("---")
            st.markdown("### ⚙️ Build Configuration")

            col1, col2 = st.columns(2)

            with col1:
                use_preprocessing = st.checkbox(
                    "🔧 Enable Image Preprocessing",
                    value=True,
                    help="Enhance image quality before processing")
                force_rebuild = st.checkbox(
                    "🔄 Force Complete Rebuild",
                    help="Delete existing index and rebuild from scratch")

            with col2:
                max_workers = st.slider("🚀 Worker Processes",
                                        min_value=1,
                                        max_value=8,
                                        value=4,
                                        help="Number of parallel processes")
                chunk_size = st.slider("📦 Chunk Size",
                                       min_value=10,
                                       max_value=200,
                                       value=50,
                                       help="Images processed per batch")

            # Build options explanation
            st.markdown("---")
            st.markdown("### 🎯 Build Options")

            # Create expandable sections for each build type
            with st.expander("🚀 Quick Build - Uses Your Current Settings",
                             expanded=True):
                st.markdown(f"""
                **Configuration:**
                - Preprocessing: {'✅ Enabled' if use_preprocessing else '❌ Disabled'}
                - Workers: {max_workers} parallel processes
                - Chunk Size: {chunk_size} images per batch
                - Force Rebuild: {'✅ Yes' if force_rebuild else '❌ No'}

                **Best for:** Standard processing with your custom settings
                """)
                if st.button("🚀 Start Quick Build",
                             type="primary",
                             use_container_width=True,
                             key="quick_build"):
                    perform_build_operation(orchestrator, folder_path,
                                            use_preprocessing, max_workers,
                                            chunk_size, force_rebuild,
                                            "Quick Build")

            with st.expander("⚡ Fast Build - Optimized for Speed"):
                st.markdown("""
                **Configuration:**
                - Preprocessing: ❌ Disabled (faster processing)
                - Workers: 6 parallel processes (high parallelism)
                - Chunk Size: 100 images per batch (larger batches)
                - Force Rebuild: Same as your setting

                **Best for:** Large datasets where speed is priority over accuracy
                **Trade-off:** May have slightly lower face detection accuracy
                """)
                if st.button("⚡ Start Fast Build",
                             use_container_width=True,
                             key="fast_build"):
                    perform_build_operation(orchestrator, folder_path, False,
                                            6, 100, force_rebuild,
                                            "Fast Build")

            with st.expander("🎯 Quality Build - Optimized for Accuracy"):
                st.markdown("""
                **Configuration:**
                - Preprocessing: ✅ Enabled (image enhancement)
                - Workers: 2 parallel processes (lower parallelism for stability)
                - Chunk Size: 25 images per batch (smaller batches)
                - Force Rebuild: Same as your setting

                **Best for:** High-quality results, important datasets
                **Trade-off:** Slower processing but better face detection and matching
                """)
                if st.button("🎯 Start Quality Build",
                             use_container_width=True,
                             key="quality_build"):
                    perform_build_operation(orchestrator, folder_path, True, 2,
                                            25, force_rebuild, "Quality Build")
        else:
            st.warning("⚠️ No images found in the selected folder")
    else:
        st.info("📁 Please select a source folder to begin")


def render_add_to_index_tab(orchestrator, fs_ops, settings):
    """Render the add to index tab"""

    st.markdown("### ➕ Add Images to Existing Index")
    st.markdown(
        "Expand your existing index by adding new images without rebuilding everything."
    )

    # Check if index exists
    if not orchestrator.index_manager.index_exists():
        st.error("❌ No existing index found. Please build an index first.")
        return

    # Load and display current index info
    index_stats = orchestrator.get_index_stats()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="metric-container"><div class="metric-value">{}</div><div class="metric-label">Current Faces</div></div>'
            .format(index_stats.get('total_faces', 0)),
            unsafe_allow_html=True)
    with col2:
        st.markdown(
            '<div class="metric-container"><div class="metric-value">{}</div><div class="metric-label">Source Folder</div></div>'
            .format(
                os.path.basename(index_stats.get('source_folder', 'Unknown'))),
            unsafe_allow_html=True)
    with col3:
        created_date = index_stats.get('created_at', 'Unknown')
        if created_date != 'Unknown':
            try:
                created_date = datetime.fromisoformat(
                    created_date.replace('Z', '+00:00')).strftime('%Y-%m-%d')
            except:
                pass
        st.markdown(
            '<div class="metric-container"><div class="metric-value">{}</div><div class="metric-label">Created</div></div>'
            .format(created_date),
            unsafe_allow_html=True)

    st.markdown("---")

    # Add options
    add_method = st.radio("📥 Addition Method",
                          ["📁 Add Folder", "📄 Add Individual Files"],
                          help="Choose how to add new images to your index")

    if add_method == "📁 Add Folder":
        # Folder addition
        col1, col2 = st.columns([3, 1])

        with col1:
            new_folder_path = st.text_input(
                "📁 New Images Folder",
                placeholder="/path/to/new/images",
                help="Path to folder containing new images to add")

        with col2:
            if st.button("📂 Browse"):
                if new_folder_path and os.path.exists(new_folder_path):
                    image_count = fs_ops.count_images_in_folder(
                        new_folder_path)
                    st.success(f"✅ Found {image_count} images")
                else:
                    st.error("❌ Folder not found")

        if new_folder_path and os.path.exists(new_folder_path):
            image_count = fs_ops.count_images_in_folder(new_folder_path)

            if image_count > 0:
                st.info(f"📊 Found {image_count} images in the folder")

                # Addition options
                col1, col2 = st.columns(2)
                with col1:
                    use_preprocessing = st.checkbox("🔧 Enable Preprocessing",
                                                    value=True)
                with col2:
                    max_workers = st.slider("🚀 Workers",
                                            min_value=1,
                                            max_value=6,
                                            value=3)

                if st.button("➕ Add Folder to Index",
                             type="primary",
                             use_container_width=True):
                    add_folder_to_index(orchestrator, new_folder_path,
                                        use_preprocessing, max_workers)
            else:
                st.warning("⚠️ No images found in the selected folder")

    else:
        # Individual file addition
        uploaded_files = st.file_uploader(
            "📄 Select Images to Add",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Upload individual images to add to the index")

        if uploaded_files:
            st.success(f"✅ Selected {len(uploaded_files)} files")

            # Show preview of uploaded files
            if len(uploaded_files) <= 5:
                cols = st.columns(len(uploaded_files))
                for i, file in enumerate(uploaded_files):
                    with cols[i]:
                        st.image(file, caption=file.name, width=150)
            else:
                st.info(
                    f"📊 {len(uploaded_files)} files selected (showing first 3)"
                )
                cols = st.columns(3)
                for i, file in enumerate(uploaded_files[:3]):
                    with cols[i]:
                        st.image(file, caption=file.name, width=150)

            # Addition options
            col1, col2 = st.columns(2)
            with col1:
                use_preprocessing = st.checkbox("🔧 Enable Preprocessing",
                                                value=True,
                                                key="individual_preprocess")
            with col2:
                max_workers = st.slider("🚀 Workers",
                                        min_value=1,
                                        max_value=4,
                                        value=2,
                                        key="individual_workers")

            if st.button("➕ Add Files to Index",
                         type="primary",
                         use_container_width=True):
                add_files_to_index(orchestrator, uploaded_files,
                                   use_preprocessing, max_workers, settings)


def render_index_status_tab(orchestrator, fs_ops, settings):
    """Render the index status tab"""

    st.markdown("### 📊 Index Status & Statistics")

    # Get comprehensive stats
    system_status = orchestrator.get_system_status()
    index_status = system_status.get('index', {})

    if index_status.get('exists', False):
        stats = orchestrator.get_index_stats()

        # Main statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                '<div class="metric-container"><div class="metric-value">{}</div><div class="metric-label">Total Faces</div></div>'
                .format(stats.get('total_faces', 0)),
                unsafe_allow_html=True)

        with col2:
            st.markdown(
                '<div class="metric-container"><div class="metric-value">{}</div><div class="metric-label">Total Images</div></div>'
                .format(stats.get('total_images', 0)),
                unsafe_allow_html=True)

        with col3:
            success_rate = (stats.get('successful_embeddings', 0) /
                            max(stats.get('total_images', 1), 1)) * 100
            st.markdown(
                '<div class="metric-container"><div class="metric-value">{:.1f}%</div><div class="metric-label">Success Rate</div></div>'
                .format(success_rate),
                unsafe_allow_html=True)

        with col4:
            dimension = stats.get('embedding_dimension', 0)
            st.markdown(
                '<div class="metric-container"><div class="metric-value">{}</div><div class="metric-label">Dimensions</div></div>'
                .format(dimension),
                unsafe_allow_html=True)

        st.markdown("---")

        # Detailed information
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                '<div class="section-header">📋 Index Information</div>',
                unsafe_allow_html=True)
            info_data = {
                "Source Folder": stats.get('source_folder', 'Unknown'),
                "Created": format_date(stats.get('created_at')),
                "Last Updated": format_date(stats.get('last_updated')),
                "Preprocessing":
                "Yes" if stats.get('preprocessing_used') else "No"
            }

            for key, value in info_data.items():
                st.markdown(f"**{key}:** {value}")

        with col2:
            st.markdown(
                '<div class="section-header">🔧 Technical Details</div>',
                unsafe_allow_html=True)
            tech_data = {
                "Model Used":
                stats.get('model_used', 'Unknown'),
                "Detector":
                stats.get('detector_used', 'Unknown'),
                "Index Size":
                f"{stats.get('index_size', 0):,} entries",
                "File Size":
                f"{stats.get('index_file_size', 0) / 1024 / 1024:.2f} MB"
                if stats.get('index_file_size') else "Unknown"
            }

            for key, value in tech_data.items():
                st.markdown(f"**{key}:** {value}")

        # Update history if available
        if 'update_history' in stats:
            st.markdown("---")
            st.markdown('<div class="section-header">📈 Update History</div>',
                        unsafe_allow_html=True)

            history = stats['update_history']
            if history:
                for update in history[-5:]:  # Show last 5 updates
                    timestamp = format_date(update.get('timestamp'))
                    images_added = update.get('images_added', 0)
                    st.markdown(
                        f"• **{timestamp}:** Added {images_added} images")
            else:
                st.info("No update history available")

        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Reload Index", use_container_width=True):
                st.cache_data.clear()
                st.success("✅ Index reloaded")
                st.rerun()

        with col2:
            if st.button("📊 Detailed Analysis", use_container_width=True):
                show_detailed_analysis(stats)

        with col3:
            if st.button("🗑️ Delete Index", use_container_width=True):
                show_delete_confirmation(orchestrator)

    else:
        st.markdown(
            '<div class="metric-container"><div class="status-warning">⚠️ No Index Found</div><div class="metric-label">Build an index to see statistics</div></div>',
            unsafe_allow_html=True)

        if st.button("🏗️ Go to Build Index", type="primary"):
            st.session_state.current_page = "📊 Index Management"
            st.rerun()


def render_settings_tab(orchestrator, fs_ops, settings):
    """Render the settings tab"""

    st.markdown("### ⚙️ System Settings")

    # Directory settings
    st.markdown('<div class="section-header">📁 Directory Configuration</div>',
                unsafe_allow_html=True)

    directories = {
        "Upload Directory": settings.UPLOAD_DIR,
        "Preprocessed Directory": settings.PREPROCESSED_DIR,
        "Embeddings Directory": settings.EMBED_DIR,
        "Temporary Directory": getattr(settings, 'TEMP_DIR', '/tmp')
    }

    for name, path in directories.items():
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"**{name}:**")
        with col2:
            st.code(path)
        with col3:
            exists = os.path.exists(path)
            status = "✅" if exists else "❌"
            st.markdown(f"{status} {'Exists' if exists else 'Missing'}")

    # Performance settings
    st.markdown("---")
    st.markdown('<div class="section-header">⚡ Performance Settings</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Processing Options:**")
        st.markdown("• Default chunk size: 50 images")
        st.markdown("• Max workers: 4-6 processes")
        st.markdown("• Memory optimization: Enabled")

    with col2:
        st.markdown("**Recommended Settings:**")
        st.markdown("• Small datasets (<1000): chunk_size=25")
        st.markdown("• Medium datasets (1000-5000): chunk_size=50")
        st.markdown("• Large datasets (>5000): chunk_size=100")

    # Cleanup options
    st.markdown("---")
    st.markdown('<div class="section-header">🧹 Cleanup Options</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("✅ Cache cleared")

    with col2:
        if st.button("🧹 Clean Temp Files", use_container_width=True):
            # Implement temp file cleanup
            temp_dir = getattr(settings, 'TEMP_DIR', '/tmp')
            if os.path.exists(temp_dir):
                try:
                    for file in os.listdir(temp_dir):
                        if file.startswith('streamlit'):
                            os.remove(os.path.join(temp_dir, file))
                    st.success("✅ Temporary files cleaned")
                except Exception as e:
                    st.error(f"❌ Cleanup failed: {e}")
            else:
                st.info("No temporary files to clean")

    with col3:
        if st.button("📊 System Info", use_container_width=True):
            show_system_info()


# Helper functions
def get_folder_size(folder_path):
    """Get folder size in MB"""
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # Convert to MB
    except:
        return 0


def count_subfolders(folder_path):
    """Count subfolders in a directory"""
    try:
        return len([
            d for d in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, d))
        ])
    except:
        return 0


def format_date(date_string):
    """Format date string for display"""
    if not date_string or date_string == 'Unknown':
        return 'Unknown'
    try:
        dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M')
    except:
        return date_string


def build_index_interactive(orchestrator, source_folder, use_preprocessing,
                            max_workers, chunk_size, force_rebuild):
    """Interactive index building with enhanced progress display"""

    if force_rebuild:
        with st.spinner("🗑️ Removing existing index..."):
            orchestrator.delete_index()
        st.success("✅ Existing index removed")

    # Create progress containers
    progress_container = st.container()
    with progress_container:
        st.markdown('<div class="section-header">🚀 Building Index</div>',
                    unsafe_allow_html=True)

        overall_progress = st.progress(0)
        status_text = st.empty()
        metrics_container = st.empty()

    start_time = time.time()

    def progress_callback(processed, total):
        """Enhanced progress callback"""
        try:
            progress = min(95, int(
                (processed / total) * 100)) if total > 0 else 0
            overall_progress.progress(progress)

            elapsed_time = time.time() - start_time
            rate = processed / elapsed_time if elapsed_time > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0

            status_text.markdown(
                f"**Processing:** {processed:,}/{total:,} images ({progress}% complete)"
            )

            # Update metrics
            with metrics_container.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Processed", f"{processed:,}")
                with col2:
                    st.metric("Rate",
                              f"{rate:.1f}/sec" if rate > 0 else "0/sec")
                with col3:
                    st.metric(
                        "ETA",
                        f"{eta/60:.1f}m" if eta > 0 else "Calculating...")
        except:
            pass

    try:
        # Build index with progress tracking
        result = orchestrator.build_complete_index_with_progress(
            source_folder=source_folder,
            use_preprocessing=use_preprocessing,
            max_workers=max_workers,
            chunk_size=chunk_size,
            progress_callback=progress_callback)

        # Final progress update
        overall_progress.progress(100)
        total_time = time.time() - start_time

        if result['success']:
            status_text.markdown(
                "**✅ Index building completed successfully!**")

            # Final metrics
            with metrics_container.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Faces", result.get('faces_indexed', 0))
                with col2:
                    success_rate = (result.get('faces_indexed', 0) / result.get('total_images', 1) * 100) if result.get('total_images', 0) > 0 else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                with col3:
                    st.metric("Total Time", f"{total_time:.1f}s")
                with col4:
                    rate = result.get('faces_indexed', 0) / total_time if total_time > 0 else 0
                    st.metric("Average Rate", f"{rate:.1f}/sec")

            st.success(
                f"🎉 Successfully indexed {result.get('faces_indexed', 0)} faces from {result.get('total_images', 0)} images!"
            )

            if result.get('failed_extractions', 0) > 0:
                st.warning(
                    f"⚠️ {result['failed_extractions']} images failed to process"
                )

            # Clear cache and rerun
            st.cache_data.clear()
            time.sleep(2)
            st.rerun()
        else:
            st.error(
                f"❌ Index building failed: {result.get('error', 'Unknown error')}"
            )

    except Exception as e:
        st.error(f"❌ Index building failed: {e}")


def add_folder_to_index(orchestrator, folder_path, use_preprocessing,
                        max_workers):
    """Add folder to existing index"""

    with st.spinner("➕ Adding images to index..."):
        result = orchestrator.add_folder_to_existing_index(
            folder_path=folder_path,
            use_preprocessing=use_preprocessing,
            max_workers=max_workers)

    if result['success']:
        st.success(f"✅ Added {result['new_faces_added']} faces to the index!")
        st.info(f"📊 Total faces in index: {result['total_faces_in_index']}")

        if result.get('failed_extractions', 0) > 0:
            st.warning(
                f"⚠️ {result['failed_extractions']} images failed to process")

        st.cache_data.clear()
        time.sleep(1)
        st.rerun()
    else:
        st.error(
            f"❌ Failed to add images: {result.get('error', 'Unknown error')}")


def add_files_to_index(orchestrator, uploaded_files, use_preprocessing,
                       max_workers, settings):
    """Add uploaded files to existing index"""

    # Save uploaded files temporarily
    temp_paths = []
    temp_dir = getattr(settings, 'TEMP_DIR', '/tmp/streamlit_uploads')
    os.makedirs(temp_dir, exist_ok=True)

    try:
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            temp_paths.append(temp_path)

        with st.spinner("➕ Adding files to index..."):
            result = orchestrator.add_images_to_existing_index(
                image_paths=temp_paths,
                use_preprocessing=use_preprocessing,
                max_workers=max_workers)

        if result['success']:
            st.success(
                f"✅ Added {result['new_faces_added']} faces from {len(uploaded_files)} files!"
            )
            st.info(
                f"📊 Total faces in index: {result['total_faces_in_index']}")

            if result.get('failed_extractions', 0) > 0:
                st.warning(
                    f"⚠️ {result['failed_extractions']} files failed to process"
                )

            st.cache_data.clear()
            time.sleep(1)
            st.rerun()
        else:
            st.error(
                f"❌ Failed to add files: {result.get('error', 'Unknown error')}"
            )

    finally:
        # Clean up temporary files
        for temp_path in temp_paths:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass


def show_detailed_analysis(stats):
    """Show detailed index analysis"""

    st.markdown("---")
    st.markdown('<div class="section-header">📈 Detailed Analysis</div>',
                unsafe_allow_html=True)

    # Performance metrics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Quality Metrics:**")
        total_images = stats.get('total_images', 0)
        successful = stats.get('successful_embeddings', 0)
        failed = stats.get('failed_embeddings', 0)

        if total_images > 0:
            success_rate = (successful / total_images) * 100
            failure_rate = (failed / total_images) * 100

            st.metric("Success Rate", f"{success_rate:.1f}%")
            st.metric("Failure Rate", f"{failure_rate:.1f}%")
            st.metric("Processing Efficiency", f"{successful}/{total_images}")

    with col2:
        st.markdown("**Storage Information:**")
        file_size = stats.get('index_file_size', 0)
        dimension = stats.get('embedding_dimension', 0)
        total_faces = stats.get('total_faces', 0)

        if file_size > 0:
            st.metric("Index File Size", f"{file_size / 1024 / 1024:.2f} MB")
            if total_faces > 0:
                avg_size_per_face = file_size / total_faces
                st.metric("Avg Size per Face",
                          f"{avg_size_per_face / 1024:.2f} KB")

        st.metric("Embedding Dimension", dimension)


def show_delete_confirmation(orchestrator):
    """Show delete confirmation dialog"""

    st.markdown("---")
    st.markdown(
        '<div class="section-header">⚠️ Delete Index Confirmation</div>',
        unsafe_allow_html=True)
    st.warning(
        "This action cannot be undone. All indexed data will be permanently deleted."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("❌ Cancel", use_container_width=True):
            st.rerun()

    with col2:
        if st.button("🗑️ Delete Index",
                     type="primary",
                     use_container_width=True):
            with st.spinner("🗑️ Deleting index..."):
                result = orchestrator.delete_index()

            if result['success']:
                st.success("✅ Index deleted successfully")
                st.cache_data.clear()
                time.sleep(1)
                st.rerun()
            else:
                st.error(
                    f"❌ Failed to delete index: {result.get('error', 'Unknown error')}"
                )


def show_system_info():
    """Show system information"""

    st.markdown("---")
    st.markdown('<div class="section-header">💻 System Information</div>',
                unsafe_allow_html=True)

    import platform
    import psutil

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**System:**")
        st.markdown(f"• OS: {platform.system()} {platform.release()}")
        st.markdown(f"• Python: {platform.python_version()}")
        st.markdown(f"• Architecture: {platform.machine()}")

    with col2:
        st.markdown("**Resources:**")
        st.markdown(f"• CPU Cores: {psutil.cpu_count()}")
        memory = psutil.virtual_memory()
        st.markdown(f"• Total RAM: {memory.total / 1024**3:.1f} GB")
        st.markdown(f"• Available RAM: {memory.available / 1024**3:.1f} GB")
