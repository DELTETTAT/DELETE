
import os
import streamlit as st
from PIL import Image
import shutil
import sys
from datetime import datetime
import zipfile
from io import BytesIO

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
    st.info("Please ensure all dependencies are installed and the project structure is correct.")
    st.stop()

def render_files_page():
    """Render the file management page with Netflix-style UI"""

    settings = Settings()
    fs_ops = FileSystemOperations()

    # Page header
    st.markdown('<div class="section-header">📁 File Management Center</div>', unsafe_allow_html=True)
    st.markdown("Upload, organize, and manage your image files with professional tools.")

    # Initialize orchestrator
    orchestrator = FaceMatchingOrchestrator()

    # Create main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Files", "🗂️ Browse Files", "🔧 File Operations", "📊 Storage Analytics"])

    with tab1:
        render_upload_tab(settings, fs_ops)
    
    with tab2:
        render_browse_tab(settings, fs_ops)
    
    with tab3:
        render_operations_tab(settings, fs_ops, orchestrator)
    
    with tab4:
        render_analytics_tab(settings, fs_ops)

def render_upload_tab(settings, fs_ops):
    """Render the upload files tab"""
    
    st.markdown("### 📤 Upload Image Files")
    st.markdown("Upload individual files or entire folders to your image collection.")
    
    # Upload method selection
    upload_method = st.radio(
        "📥 Upload Method",
        ["📄 Individual Files", "📁 Multiple Files", "🗂️ ZIP Archive"],
        help="Choose how to upload your images"
    )
    
    if upload_method == "📄 Individual Files":
        # Single file upload
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
            help="Upload a single image file"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(uploaded_file, caption=uploaded_file.name, width=300)
            
            with col2:
                st.markdown("**File Information:**")
                st.markdown(f"• **Name:** {uploaded_file.name}")
                st.markdown(f"• **Size:** {uploaded_file.size / 1024:.1f} KB")
                st.markdown(f"• **Type:** {uploaded_file.type}")
                
                # Upload options
                subfolder = st.text_input("📁 Subfolder (optional)", placeholder="e.g., portraits, group_photos")
                
                if st.button("📤 Upload File", type="primary", use_container_width=True):
                    upload_single_file(uploaded_file, settings, subfolder)
    
    elif upload_method == "📁 Multiple Files":
        # Multiple files upload
        uploaded_files = st.file_uploader(
            "Choose multiple image files",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
            accept_multiple_files=True,
            help="Upload multiple image files at once"
        )
        
        if uploaded_files:
            st.success(f"✅ Selected {len(uploaded_files)} files")
            
            # Show preview of first few files
            if len(uploaded_files) <= 6:
                cols = st.columns(min(len(uploaded_files), 3))
                for i, file in enumerate(uploaded_files[:3]):
                    with cols[i % 3]:
                        st.image(file, caption=file.name[:20], width=150)
            else:
                st.info(f"📊 {len(uploaded_files)} files selected (showing first 3)")
                cols = st.columns(3)
                for i, file in enumerate(uploaded_files[:3]):
                    with cols[i]:
                        st.image(file, caption=file.name[:20], width=150)
            
            # Upload options
            col1, col2 = st.columns(2)
            with col1:
                subfolder = st.text_input("📁 Subfolder (optional)", placeholder="e.g., batch_2024", key="multi_subfolder")
            with col2:
                overwrite = st.checkbox("🔄 Overwrite existing files", help="Replace files with the same name")
            
            # File size summary
            total_size = sum(file.size for file in uploaded_files) / (1024 * 1024)
            st.info(f"📊 Total size: {total_size:.2f} MB")
            
            if st.button("📤 Upload All Files", type="primary", use_container_width=True):
                upload_multiple_files(uploaded_files, settings, subfolder, overwrite)
    
    else:  # ZIP Archive
        # ZIP file upload
        uploaded_zip = st.file_uploader(
            "Choose a ZIP archive",
            type=['zip'],
            help="Upload a ZIP archive containing images"
        )
        
        if uploaded_zip is not None:
            st.success(f"✅ ZIP file selected: {uploaded_zip.name}")
            
            # Analyze ZIP contents
            try:
                zip_contents = analyze_zip_contents(uploaded_zip)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Files", zip_contents['total_files'])
                with col2:
                    st.metric("Image Files", zip_contents['image_files'])
                with col3:
                    st.metric("Archive Size", f"{uploaded_zip.size / (1024*1024):.2f} MB")
                
                # Show some file names
                if zip_contents['image_names']:
                    with st.expander("📋 Image Files in Archive"):
                        for name in zip_contents['image_names'][:20]:  # Show first 20
                            st.markdown(f"• {name}")
                        if len(zip_contents['image_names']) > 20:
                            st.markdown(f"... and {len(zip_contents['image_names']) - 20} more files")
                
                # Extract options
                col1, col2 = st.columns(2)
                with col1:
                    extract_folder = st.text_input("📁 Extract to subfolder", value=os.path.splitext(uploaded_zip.name)[0])
                with col2:
                    preserve_structure = st.checkbox("🗂️ Preserve folder structure", value=True)
                
                if st.button("📦 Extract and Upload", type="primary", use_container_width=True):
                    extract_and_upload_zip(uploaded_zip, settings, extract_folder, preserve_structure)
            
            except Exception as e:
                st.error(f"❌ Error analyzing ZIP file: {e}")

def render_browse_tab(settings, fs_ops):
    """Render the browse files tab"""
    
    st.markdown("### 🗂️ Browse Image Collection")
    st.markdown("Explore and manage your uploaded image files.")
    
    # Get directory statistics
    upload_stats = get_directory_stats(settings.UPLOAD_DIR)
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-container"><div class="metric-value">{}</div><div class="metric-label">Total Images</div></div>'.format(upload_stats['total_images']), unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-container"><div class="metric-value">{:.1f} MB</div><div class="metric-label">Total Size</div></div>'.format(upload_stats['total_size_mb']), unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-container"><div class="metric-value">{}</div><div class="metric-label">Subfolders</div></div>'.format(upload_stats['subfolders']), unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-container"><div class="metric-value">{}</div><div class="metric-label">File Types</div></div>'.format(len(upload_stats['file_types'])), unsafe_allow_html=True)
    
    if upload_stats['total_images'] == 0:
        st.markdown('<div class="metric-container"><div class="status-warning">⚠️ No Images Found</div><div class="metric-label">Upload images to get started</div></div>', unsafe_allow_html=True)
        return
    
    st.markdown("---")
    
    # Folder navigation
    st.markdown("### 📁 Folder Navigation")
    
    # Get all subdirectories
    subdirs = get_subdirectories(settings.UPLOAD_DIR)
    
    if subdirs:
        selected_folder = st.selectbox(
            "Select folder to browse",
            ["📁 Root Directory"] + [f"📁 {folder}" for folder in subdirs],
            help="Choose a folder to explore"
        )
        
        if selected_folder == "📁 Root Directory":
            current_path = settings.UPLOAD_DIR
        else:
            folder_name = selected_folder.replace("📁 ", "")
            current_path = os.path.join(settings.UPLOAD_DIR, folder_name)
    else:
        current_path = settings.UPLOAD_DIR
    
    # Display current folder contents
    display_folder_contents(current_path, fs_ops)

def render_operations_tab(settings, fs_ops, orchestrator):
    """Render the file operations tab"""
    
    st.markdown("### 🔧 File Operations")
    st.markdown("Perform bulk operations on your image collection.")
    
    # Operation selection
    operation = st.selectbox(
        "🛠️ Select Operation",
        [
            "🗑️ Clean Up Files",
            "📁 Organize by Date",
            "🔄 Convert Formats",
            "📏 Resize Images",
            "🧹 Remove Duplicates",
            "📦 Create Archive"
        ],
        help="Choose the operation to perform"
    )
    
    st.markdown("---")
    
    if operation == "🗑️ Clean Up Files":
        render_cleanup_operation(settings, fs_ops, orchestrator)
    elif operation == "📁 Organize by Date":
        render_organize_operation(settings, fs_ops)
    elif operation == "🔄 Convert Formats":
        render_convert_operation(settings, fs_ops)
    elif operation == "📏 Resize Images":
        render_resize_operation(settings, fs_ops)
    elif operation == "🧹 Remove Duplicates":
        render_duplicate_operation(settings, fs_ops)
    elif operation == "📦 Create Archive":
        render_archive_operation(settings, fs_ops)

def render_analytics_tab(settings, fs_ops):
    """Render the storage analytics tab"""
    
    st.markdown("### 📊 Storage Analytics")
    st.markdown("Analyze your storage usage and file distribution.")
    
    # Get comprehensive analytics
    analytics = get_comprehensive_analytics(settings, fs_ops)
    
    # Storage overview
    st.markdown("#### 💾 Storage Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-container"><div class="metric-value">{}</div><div class="metric-label">Total Files</div></div>'.format(analytics['total_files']), unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-container"><div class="metric-value">{:.1f} GB</div><div class="metric-label">Total Size</div></div>'.format(analytics['total_size_gb']), unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-container"><div class="metric-value">{:.1f} MB</div><div class="metric-label">Average Size</div></div>'.format(analytics['avg_file_size_mb']), unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-container"><div class="metric-value">{}</div><div class="metric-label">Largest File</div></div>'.format(f"{analytics['largest_file_mb']:.1f} MB"), unsafe_allow_html=True)
    
    # File type distribution
    st.markdown("---")
    st.markdown("#### 📊 File Type Distribution")
    
    if analytics['file_types']:
        for file_type, count in analytics['file_types'].items():
            percentage = (count / analytics['total_files']) * 100 if analytics['total_files'] > 0 else 0
            st.markdown(f"**{file_type.upper()}:** {count} files ({percentage:.1f}%)")
            st.progress(percentage / 100)
    
    # Folder analysis
    st.markdown("---")
    st.markdown("#### 📁 Folder Analysis")
    
    if analytics['folder_stats']:
        for folder, stats in analytics['folder_stats'].items():
            with st.expander(f"📁 {folder} ({stats['file_count']} files)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("File Count", stats['file_count'])
                    st.metric("Total Size", f"{stats['total_size_mb']:.1f} MB")
                with col2:
                    st.metric("Average Size", f"{stats['avg_size_mb']:.1f} MB")
                    st.metric("Last Modified", stats['last_modified'])

# Helper functions

def upload_single_file(uploaded_file, settings, subfolder=""):
    """Upload a single file"""
    try:
        # Create target directory
        if subfolder:
            target_dir = os.path.join(settings.UPLOAD_DIR, subfolder)
        else:
            target_dir = settings.UPLOAD_DIR
        
        os.makedirs(target_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(target_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Upload failed: {e}")

def upload_multiple_files(uploaded_files, settings, subfolder="", overwrite=False):
    """Upload multiple files"""
    try:
        # Create target directory
        if subfolder:
            target_dir = os.path.join(settings.UPLOAD_DIR, subfolder)
        else:
            target_dir = settings.UPLOAD_DIR
        
        os.makedirs(target_dir, exist_ok=True)
        
        # Upload progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        successful = 0
        failed = 0
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Uploading {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
            progress_bar.progress((i + 1) / len(uploaded_files))
            
            try:
                file_path = os.path.join(target_dir, uploaded_file.name)
                
                # Check if file exists
                if os.path.exists(file_path) and not overwrite:
                    st.warning(f"⚠️ Skipped {uploaded_file.name} (already exists)")
                    continue
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                successful += 1
                
            except Exception as e:
                st.error(f"❌ Failed to upload {uploaded_file.name}: {e}")
                failed += 1
        
        status_text.text("✅ Upload completed!")
        st.success(f"✅ Successfully uploaded {successful} files")
        
        if failed > 0:
            st.warning(f"⚠️ {failed} files failed to upload")
        
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Upload failed: {e}")

def analyze_zip_contents(uploaded_zip):
    """Analyze contents of uploaded ZIP file"""
    zip_contents = {
        'total_files': 0,
        'image_files': 0,
        'image_names': [],
        'other_files': []
    }
    
    try:
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if not file_info.is_dir():
                    zip_contents['total_files'] += 1
                    
                    # Check if it's an image file
                    file_ext = os.path.splitext(file_info.filename)[1].lower()
                    if file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
                        zip_contents['image_files'] += 1
                        zip_contents['image_names'].append(file_info.filename)
                    else:
                        zip_contents['other_files'].append(file_info.filename)
    
    except Exception as e:
        st.error(f"Error analyzing ZIP: {e}")
    
    return zip_contents

def extract_and_upload_zip(uploaded_zip, settings, extract_folder, preserve_structure):
    """Extract and upload ZIP contents"""
    try:
        # Create target directory
        if extract_folder:
            target_dir = os.path.join(settings.UPLOAD_DIR, extract_folder)
        else:
            target_dir = settings.UPLOAD_DIR
        
        os.makedirs(target_dir, exist_ok=True)
        
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
            
            # Get list of image files
            image_files = [f for f in zip_ref.namelist() 
                          if os.path.splitext(f)[1].lower() in image_extensions and not f.endswith('/')]
            
            if not image_files:
                st.warning("⚠️ No image files found in the archive")
                return
            
            # Extract with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            extracted = 0
            
            for i, file_path in enumerate(image_files):
                status_text.text(f"Extracting {os.path.basename(file_path)} ({i+1}/{len(image_files)})")
                progress_bar.progress((i + 1) / len(image_files))
                
                try:
                    # Determine target path
                    if preserve_structure:
                        target_path = os.path.join(target_dir, file_path)
                    else:
                        target_path = os.path.join(target_dir, os.path.basename(file_path))
                    
                    # Create directory if needed
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    
                    # Extract file
                    with zip_ref.open(file_path) as source, open(target_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    
                    extracted += 1
                    
                except Exception as e:
                    st.error(f"❌ Failed to extract {file_path}: {e}")
            
            status_text.text("✅ Extraction completed!")
            st.success(f"✅ Successfully extracted {extracted} image files")
            st.rerun()
    
    except Exception as e:
        st.error(f"❌ Extraction failed: {e}")

def get_directory_stats(directory):
    """Get comprehensive directory statistics"""
    stats = {
        'total_images': 0,
        'total_size_mb': 0,
        'subfolders': 0,
        'file_types': {}
    }
    
    try:
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
        
        for root, dirs, files in os.walk(directory):
            # Count subfolders (only in immediate directory)
            if root == directory:
                stats['subfolders'] = len(dirs)
            
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in image_extensions:
                    stats['total_images'] += 1
                    
                    # File size
                    try:
                        file_size = os.path.getsize(file_path)
                        stats['total_size_mb'] += file_size / (1024 * 1024)
                    except:
                        pass
                    
                    # File type
                    if file_ext in stats['file_types']:
                        stats['file_types'][file_ext] += 1
                    else:
                        stats['file_types'][file_ext] = 1
    
    except Exception as e:
        st.error(f"Error getting directory stats: {e}")
    
    return stats

def get_subdirectories(directory):
    """Get list of subdirectories"""
    try:
        subdirs = []
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                subdirs.append(item)
        return sorted(subdirs)
    except:
        return []

def display_folder_contents(folder_path, fs_ops):
    """Display contents of a folder"""
    try:
        image_files = []
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
        
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in image_extensions:
                    image_files.append(file)
        
        if not image_files:
            st.info("📁 No images found in this folder")
            return
        
        st.markdown(f"### 📋 Folder Contents ({len(image_files)} images)")
        
        # Pagination for large folders
        items_per_page = 12
        total_pages = (len(image_files) + items_per_page - 1) // items_per_page
        
        if total_pages > 1:
            page = st.selectbox(f"Page (1-{total_pages})", range(1, total_pages + 1)) - 1
        else:
            page = 0
        
        start_idx = page * items_per_page
        end_idx = min(start_idx + items_per_page, len(image_files))
        page_files = image_files[start_idx:end_idx]
        
        # Display images in grid
        cols_per_row = 4
        rows = [page_files[i:i + cols_per_row] for i in range(0, len(page_files), cols_per_row)]
        
        for row in rows:
            cols = st.columns(len(row))
            for i, filename in enumerate(row):
                with cols[i]:
                    file_path = os.path.join(folder_path, filename)
                    try:
                        image = Image.open(file_path)
                        st.image(image, caption=filename[:20], width=150)
                        
                        # File info
                        file_size = os.path.getsize(file_path) / 1024
                        st.caption(f"{file_size:.1f} KB")
                        
                        # Action buttons
                        if st.button("🗑️", key=f"delete_{filename}", help=f"Delete {filename}"):
                            delete_file_confirm(file_path, filename)
                    
                    except Exception as e:
                        st.error(f"Error loading {filename}")
    
    except Exception as e:
        st.error(f"Error displaying folder contents: {e}")

def render_cleanup_operation(settings, fs_ops, orchestrator):
    """Render cleanup operations"""
    st.markdown("#### 🗑️ Clean Up Files")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧹 Clear Upload Directory", use_container_width=True):
            if st.button("⚠️ Confirm Clear Uploads", type="secondary"):
                clear_directory(settings.UPLOAD_DIR, "uploads")
    
    with col2:
        if st.button("🗑️ Clear Preprocessed", use_container_width=True):
            if st.button("⚠️ Confirm Clear Preprocessed", type="secondary"):
                clear_directory(settings.PREPROCESSED_DIR, "preprocessed images")
    
    with col3:
        if st.button("🧹 Clean Temp Files", use_container_width=True):
            clean_temp_files()

def clear_directory(directory, name):
    """Clear a directory"""
    try:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)
            st.success(f"✅ Cleared {name} directory")
        else:
            st.warning(f"⚠️ {name} directory doesn't exist")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Failed to clear {name}: {e}")

def clean_temp_files():
    """Clean temporary files"""
    try:
        temp_dirs = ["/tmp/streamlit_uploads", "/tmp/streamlit_search", "/tmp/streamlit_batch"]
        cleaned = 0
        
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                cleaned += 1
        
        st.success(f"✅ Cleaned {cleaned} temporary directories")
    except Exception as e:
        st.error(f"❌ Failed to clean temp files: {e}")

def delete_file_confirm(file_path, filename):
    """Confirm file deletion"""
    st.warning(f"⚠️ Are you sure you want to delete {filename}?")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("❌ Cancel"):
            st.rerun()
    with col2:
        if st.button("🗑️ Delete", type="primary"):
            try:
                os.remove(file_path)
                st.success(f"✅ Deleted {filename}")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to delete: {e}")

def render_organize_operation(settings, fs_ops):
    """Render organize by date operation"""
    st.markdown("#### 📁 Organize by Date")
    st.info("🚧 Date organization feature coming soon!")

def render_convert_operation(settings, fs_ops):
    """Render format conversion operation"""
    st.markdown("#### 🔄 Convert Formats")
    st.info("🚧 Format conversion feature coming soon!")

def render_resize_operation(settings, fs_ops):
    """Render image resize operation"""
    st.markdown("#### 📏 Resize Images")
    st.info("🚧 Image resize feature coming soon!")

def render_duplicate_operation(settings, fs_ops):
    """Render duplicate removal operation"""
    st.markdown("#### 🧹 Remove Duplicates")
    st.info("🚧 Duplicate removal feature coming soon!")

def render_archive_operation(settings, fs_ops):
    """Render archive creation operation"""
    st.markdown("#### 📦 Create Archive")
    st.info("🚧 Archive creation feature coming soon!")

def get_comprehensive_analytics(settings, fs_ops):
    """Get comprehensive storage analytics"""
    analytics = {
        'total_files': 0,
        'total_size_gb': 0,
        'avg_file_size_mb': 0,
        'largest_file_mb': 0,
        'file_types': {},
        'folder_stats': {}
    }
    
    try:
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
        total_size_bytes = 0
        file_sizes = []
        
        for root, dirs, files in os.walk(settings.UPLOAD_DIR):
            folder_size = 0
            folder_files = 0
            
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in image_extensions:
                    try:
                        file_size = os.path.getsize(file_path)
                        file_sizes.append(file_size)
                        total_size_bytes += file_size
                        folder_size += file_size
                        folder_files += 1
                        analytics['total_files'] += 1
                        
                        # File type count
                        if file_ext in analytics['file_types']:
                            analytics['file_types'][file_ext] += 1
                        else:
                            analytics['file_types'][file_ext] = 1
                    
                    except:
                        pass
            
            # Folder stats
            if folder_files > 0:
                folder_name = os.path.relpath(root, settings.UPLOAD_DIR)
                if folder_name == '.':
                    folder_name = 'Root'
                
                analytics['folder_stats'][folder_name] = {
                    'file_count': folder_files,
                    'total_size_mb': folder_size / (1024 * 1024),
                    'avg_size_mb': (folder_size / folder_files) / (1024 * 1024) if folder_files > 0 else 0,
                    'last_modified': datetime.fromtimestamp(os.path.getmtime(root)).strftime('%Y-%m-%d')
                }
        
        # Calculate final metrics
        analytics['total_size_gb'] = total_size_bytes / (1024 * 1024 * 1024)
        analytics['avg_file_size_mb'] = (total_size_bytes / analytics['total_files']) / (1024 * 1024) if analytics['total_files'] > 0 else 0
        analytics['largest_file_mb'] = max(file_sizes) / (1024 * 1024) if file_sizes else 0
    
    except Exception as e:
        st.error(f"Error getting analytics: {e}")
    
    return analytics
