import os
import streamlit as st
from PIL import Image
import shutil
import sys

# Add the src directory to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, src_path)

from core.orchestrator import FaceMatchingOrchestrator
from utils.filesystem.operations import FileSystemOperations
from config.settings import Settings

def render_files_page():
    """Render the file management page"""

    st.markdown('<h1 class="main-header">📁 File Management</h1>', unsafe_allow_html=True)

    # Initialize components
    orchestrator = FaceMatchingOrchestrator()
    fs_ops = FileSystemOperations()
    settings = Settings()

    # File upload section
    st.markdown("## 📤 Upload Images")

    uploaded_files = st.file_uploader(
        "Choose image files",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        accept_multiple_files=True,
        help="Upload images to the system for processing and searching"
    )

    if uploaded_files:
        if st.button("💾 Save Uploaded Files", type="primary"):
            save_uploaded_files(uploaded_files, settings.UPLOAD_DIR)

    st.markdown("---")

    # Directory browser
    st.markdown("## 🗂️ Browse Directories")

    # Directory tabs
    tab1, tab2, tab3 = st.tabs(["📥 Uploads", "🔧 Preprocessed", "📊 System Info"])

    with tab1:
        display_directory_contents("Upload Directory", settings.UPLOAD_DIR, fs_ops)

    with tab2:
        display_directory_contents("Preprocessed Directory", settings.PREPROCESSED_DIR, fs_ops)

    with tab3:
        display_system_directories(orchestrator, fs_ops, settings)

    st.markdown("---")

    # Cleanup operations
    st.markdown("## 🧹 Cleanup Operations")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🗑️ Clear Uploads"):
            if st.button("⚠️ Confirm Clear Uploads"):
                clear_directory(settings.UPLOAD_DIR, "uploads")

    with col2:
        if st.button("🗑️ Clear Preprocessed"):
            if st.button("⚠️ Confirm Clear Preprocessed"):
                clear_directory(settings.PREPROCESSED_DIR, "preprocessed images")

    with col3:
        if st.button("🧹 Clean Temp Files"):
            result = orchestrator.cleanup_temp_files()
            if result['success']:
                st.success("✅ Temporary files cleaned")
            else:
                st.error(f"❌ Cleanup failed: {result['error']}")

def save_uploaded_files(uploaded_files, upload_dir):
    """Save uploaded files to the upload directory"""
    try:
        os.makedirs(upload_dir, exist_ok=True)

        saved_count = 0
        for uploaded_file in uploaded_files:
            file_path = os.path.join(upload_dir, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_count += 1

        st.success(f"✅ Saved {saved_count} files to uploads directory")
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error saving files: {e}")

def display_directory_contents(title, directory_path, fs_ops):
    """Display contents of a directory"""

    st.markdown(f"### {title}")
    st.markdown(f'<div class="folder-path">📁 {directory_path}</div>', unsafe_allow_html=True)

    if not os.path.exists(directory_path):
        st.warning(f"Directory does not exist: {directory_path}")
        return

    # Get image files
    try:
        image_files = fs_ops.get_image_files_in_folder(directory_path)

        if not image_files:
            st.info("No images found in this directory")
            return

        st.info(f"Found {len(image_files)} images")

        # Display images in a grid
        cols = st.columns(4)

        for i, filename in enumerate(image_files[:20]):  # Limit to first 20 for performance
            with cols[i % 4]:
                image_path = os.path.join(directory_path, filename)

                try:
                    # Display thumbnail
                    image = Image.open(image_path)
                    st.image(image, caption=filename, width=150)

                    # File info
                    file_size = os.path.getsize(image_path)
                    st.caption(f"Size: {file_size / 1024:.1f} KB")

                    # Delete button
                    if st.button(f"🗑️", key=f"delete_{filename}_{i}"):
                        if st.button(f"⚠️ Confirm", key=f"confirm_delete_{filename}_{i}"):
                            delete_file(image_path, filename)

                except Exception as e:
                    st.error(f"Error loading {filename}: {e}")

        if len(image_files) > 20:
            st.info(f"... and {len(image_files) - 20} more images")

    except Exception as e:
        st.error(f"Error reading directory: {e}")

def display_system_directories(orchestrator, fs_ops, settings):
    """Display system directory information"""

    st.markdown("### System Directory Status")

    # Get system status
    system_status = orchestrator.get_system_status()
    directories = system_status.get('directories', {})

    # Create directory status table
    dir_data = []
    for dir_name, dir_info in directories.items():
        dir_data.append({
            'Directory': dir_name.replace('_', ' ').title(),
            'Path': dir_info.get('path', 'Unknown'),
            'Exists': '✅' if dir_info.get('exists', False) else '❌',
            'Image Count': dir_info.get('image_count', 0)
        })

    if dir_data:
        import pandas as pd
        df = pd.DataFrame(dir_data)
        st.dataframe(df, use_container_width=True)

    # Storage usage
    st.markdown("### Storage Information")

    total_images = 0
    total_size = 0

    for dir_name, dir_info in directories.items():
        if dir_info.get('exists', False):
            dir_path = dir_info.get('path')
            if dir_path and os.path.exists(dir_path):
                try:
                    for root, dirs, files in os.walk(dir_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                                total_images += 1
                                total_size += os.path.getsize(file_path)
                except Exception:
                    pass

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Images", total_images)
    with col2:
        st.metric("Total Size", f"{total_size / 1024 / 1024:.1f} MB")

def delete_file(file_path, filename):
    """Delete a file"""
    try:
        os.remove(file_path)
        st.success(f"✅ Deleted {filename}")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Error deleting {filename}: {e}")

def clear_directory(directory_path, directory_name):
    """Clear all files in a directory"""
    try:
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
            os.makedirs(directory_path, exist_ok=True)
            st.success(f"✅ Cleared {directory_name} directory")
            st.rerun()
        else:
            st.warning(f"Directory does not exist: {directory_path}")
    except Exception as e:
        st.error(f"❌ Error clearing {directory_name}: {e}")