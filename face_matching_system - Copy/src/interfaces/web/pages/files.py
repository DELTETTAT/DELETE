import os
import streamlit as st
import shutil
from PIL import Image
from utils.filesystem.operations import FileSystemOperations
from config.settings import Settings

def render_files_page():
    """Render the file management page"""
    
    settings = Settings()
    fs_ops = FileSystemOperations()
    
    st.markdown('<h1 class="main-header">📁 File Management</h1>', unsafe_allow_html=True)
    
    # File management tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload Images", "📂 Browse Files", "🗂️ Manage Directories"])
    
    with tab1:
        render_upload_section(settings, fs_ops)
    
    with tab2:
        render_browse_section(settings, fs_ops)
    
    with tab3:
        render_directory_management(settings, fs_ops)

def render_upload_section(settings, fs_ops):
    """Render image upload section"""
    
    st.markdown("## 📤 Upload Images")
    st.markdown("Upload images for face matching. Supported formats: PNG, JPG, JPEG, BMP, TIFF")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        accept_multiple_files=True,
        help="Select one or more image files to upload"
    )
    
    if uploaded_files:
        st.markdown(f"### Selected {len(uploaded_files)} file(s)")
        
        # Preview uploaded files
        cols = st.columns(min(len(uploaded_files), 3))
        for i, uploaded_file in enumerate(uploaded_files[:3]):  # Show first 3 previews
            with cols[i % 3]:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, width=150)
                except Exception:
                    st.write(f"📄 {uploaded_file.name}")
        
        if len(uploaded_files) > 3:
            st.write(f"... and {len(uploaded_files) - 3} more files")
        
        # Upload options
        col1, col2 = st.columns(2)
        with col1:
            overwrite = st.checkbox("Overwrite existing files", value=False)
        with col2:
            organize_by_date = st.checkbox("Organize by upload date", value=False)
        
        # Upload button
        if st.button("📤 Upload Files", type="primary"):
            upload_files(uploaded_files, settings.UPLOAD_DIR, overwrite, organize_by_date)

def render_browse_section(settings, fs_ops):
    """Render file browsing section"""
    
    st.markdown("## 📂 Browse Files")
    
    # Directory selection
    browse_dirs = {
        "Uploads": settings.UPLOAD_DIR,
        "Preprocessed": settings.PREPROCESSED_DIR,
        "Temp": settings.TEMP_DIR
    }
    
    # Add custom folder if set
    if st.session_state.get('custom_folder_path'):
        browse_dirs["Source Folder"] = st.session_state.custom_folder_path
    
    selected_dir = st.selectbox("Select directory to browse:", list(browse_dirs.keys()))
    dir_path = browse_dirs[selected_dir]
    
    if not os.path.exists(dir_path):
        st.warning(f"Directory does not exist: {dir_path}")
        return
    
    # Display directory contents
    display_directory_contents(dir_path, fs_ops)

def render_directory_management(settings, fs_ops):
    """Render directory management section"""
    
    st.markdown("## 🗂️ Manage Directories")
    
    # Directory information
    directories = {
        "Uploads": settings.UPLOAD_DIR,
        "Embeddings": settings.EMBED_DIR,
        "Preprocessed": settings.PREPROCESSED_DIR,
        "Temp": settings.TEMP_DIR
    }
    
    for name, path in directories.items():
        with st.expander(f"📁 {name} Directory"):
            st.code(path, language=None)
            
            if os.path.exists(path):
                # Directory statistics
                image_count = fs_ops.count_images_in_folder(path)
                total_files = len(os.listdir(path))
                dir_size = get_directory_size(path)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Images", image_count)
                with col2:
                    st.metric("Total Files", total_files)
                with col3:
                    st.metric("Size", f"{dir_size:.2f} MB")
                
                # Directory actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"📂 Open {name}", key=f"open_{name}"):
                        st.session_state[f'browse_{name}'] = True
                
                with col2:
                    if name != "Embeddings":  # Don't allow clearing embeddings easily
                        if st.button(f"🗑️ Clear {name}", key=f"clear_{name}", type="secondary"):
                            if st.button(f"⚠️ Confirm Clear {name}", key=f"confirm_{name}"):
                                clear_directory(path, name)
            else:
                st.info("Directory does not exist")
                if st.button(f"📁 Create {name}", key=f"create_{name}"):
                    create_directory(path, name)

def upload_files(uploaded_files, upload_dir, overwrite=False, organize_by_date=False):
    """Upload files to the specified directory"""
    
    try:
        os.makedirs(upload_dir, exist_ok=True)
        
        uploaded_count = 0
        skipped_count = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Determine target path
            if organize_by_date:
                from datetime import datetime
                date_folder = datetime.now().strftime("%Y-%m-%d")
                target_dir = os.path.join(upload_dir, date_folder)
                os.makedirs(target_dir, exist_ok=True)
                target_path = os.path.join(target_dir, uploaded_file.name)
            else:
                target_path = os.path.join(upload_dir, uploaded_file.name)
            
            # Check if file exists
            if os.path.exists(target_path) and not overwrite:
                status_text.text(f"Skipping {uploaded_file.name} (already exists)")
                skipped_count += 1
            else:
                # Save file
                with open(target_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                uploaded_count += 1
                status_text.text(f"Uploaded {uploaded_file.name}")
            
            # Update progress
            progress = int((i + 1) * 100 / len(uploaded_files))
            progress_bar.progress(progress)
        
        # Show results
        progress_bar.empty()
        status_text.empty()
        
        if uploaded_count > 0:
            st.success(f"✅ Uploaded {uploaded_count} files successfully")
        if skipped_count > 0:
            st.info(f"ℹ️ Skipped {skipped_count} existing files")
    
    except Exception as e:
        st.error(f"❌ Upload failed: {e}")

def display_directory_contents(dir_path, fs_ops):
    """Display contents of a directory"""
    
    try:
        # Get image files
        image_files = fs_ops.get_image_files_in_folder(dir_path)
        
        if not image_files:
            st.info("No image files found in this directory")
            return
        
        st.write(f"Found {len(image_files)} image files:")
        
        # Pagination
        items_per_page = 12
        total_pages = (len(image_files) - 1) // items_per_page + 1
        
        if total_pages > 1:
            page = st.selectbox("Page", range(1, total_pages + 1))
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(image_files))
            page_files = image_files[start_idx:end_idx]
        else:
            page_files = image_files
        
        # Display images in grid
        cols = st.columns(4)
        for i, filename in enumerate(page_files):
            with cols[i % 4]:
                image_path = os.path.join(dir_path, filename)
                
                try:
                    image = Image.open(image_path)
                    st.image(image, caption=filename, width=150)
                    
                    # File info
                    file_size = os.path.getsize(image_path)
                    st.caption(f"Size: {file_size / 1024:.1f} KB")
                    
                    # Actions
                    if st.button(f"🗑️", key=f"delete_{filename}_{i}"):
                        if st.button(f"⚠️ Confirm", key=f"confirm_delete_{filename}_{i}"):
                            delete_file(image_path, filename)
                
                except Exception as e:
                    st.error(f"Error loading {filename}: {e}")
    
    except Exception as e:
        st.error(f"Error browsing directory: {e}")

def get_directory_size(dir_path):
    """Calculate directory size in MB"""
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(dir_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # Convert to MB
    except Exception:
        return 0.0

def clear_directory(dir_path, dir_name):
    """Clear all contents of a directory"""
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)
            st.success(f"✅ {dir_name} directory cleared")
            st.rerun()
        else:
            st.error(f"❌ Directory does not exist: {dir_path}")
    except Exception as e:
        st.error(f"❌ Failed to clear {dir_name} directory: {e}")

def create_directory(dir_path, dir_name):
    """Create a directory"""
    try:
        os.makedirs(dir_path, exist_ok=True)
        st.success(f"✅ {dir_name} directory created")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Failed to create {dir_name} directory: {e}")

def delete_file(file_path, filename):
    """Delete a single file"""
    try:
        os.remove(file_path)
        st.success(f"✅ Deleted {filename}")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Failed to delete {filename}: {e}")
