import os
import streamlit as st
from PIL import Image
import sys
import os
# Add the src directory to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, src_path)

from core.search.engine import SearchEngine
from utils.filesystem.operations import FileSystemOperations
from config.settings import Settings

def render_search_page():
    """Render the face search page"""
    
    settings = Settings()
    fs_ops = FileSystemOperations()
    
    st.markdown('<h1 class="main-header">🔍 Face Search</h1>', unsafe_allow_html=True)
    
    # Initialize search engine
    search_engine = SearchEngine()
    
    # Check if search engine is ready
    if not search_engine.is_ready():
        st.error("❌ No search index found. Please build an index first using the Index Management page.")
        return
    
    # Get search engine stats
    stats = search_engine.get_search_stats()
    
    # Display search info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Indexed Faces", stats.get('total_indexed_faces', 0))
    with col2:
        st.metric("Model", stats.get('model_name', 'Unknown'))
    with col3:
        st.metric("Dimension", stats.get('embedding_dimension', 0))
    
    st.markdown("---")
    
    # Search interface
    st.markdown("## 🎯 Search Configuration")
    
    # Upload query image or select from uploads folder
    search_mode = st.radio("Search Mode:", ["Upload Image", "Select from Uploads"])
    
    query_image_path = None
    
    if search_mode == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose a query image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image to search for similar faces"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            query_image_path = os.path.join(settings.TEMP_DIR, uploaded_file.name)
            os.makedirs(settings.TEMP_DIR, exist_ok=True)
            
            with open(query_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Display uploaded image
            st.image(uploaded_file, caption="Query Image", width=200)
    
    else:  # Select from uploads
        upload_images = fs_ops.get_image_files_in_folder(settings.UPLOAD_DIR)
        
        if not upload_images:
            st.warning(f"No images found in uploads directory: {settings.UPLOAD_DIR}")
            st.info("Please add images to the uploads directory first.")
            return
        
        selected_image = st.selectbox(
            "Select image from uploads:",
            upload_images,
            help="Choose an image from the uploads directory"
        )
        
        if selected_image:
            query_image_path = os.path.join(settings.UPLOAD_DIR, selected_image)
            
            # Display selected image
            try:
                image = Image.open(query_image_path)
                st.image(image, caption=f"Selected: {selected_image}", width=200)
            except Exception as e:
                st.error(f"Error loading image: {e}")
    
    # Search parameters
    st.markdown("## ⚙️ Search Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        k = st.number_input("Number of results", min_value=1, max_value=20, value=5)
    with col2:
        threshold = st.number_input("Distance threshold", min_value=0.0, value=450.0, step=10.0)
    
    # Search button
    if st.button("🔍 Search Similar Faces", type="primary"):
        if query_image_path and os.path.exists(query_image_path):
            with st.spinner("Searching for similar faces..."):
                results = search_engine.search_similar_faces(
                    query_path=query_image_path,
                    k=k,
                    threshold=threshold,
                    enforce_detection=True
                )
            
            if results:
                display_search_results(results, stats.get('source_folder'))
            else:
                st.error("Search failed. Please check the query image and try again.")
        else:
            st.error("Please provide a query image first.")

def display_search_results(results, source_folder=None):
    """Display search results in a formatted way"""
    
    st.markdown("---")
    st.markdown("## 📋 Search Results")
    
    # Filter results by threshold
    within_threshold = [r for r in results if r['within_threshold']]
    
    if not within_threshold:
        st.warning("🚫 No matches found within the specified threshold")
        st.markdown("### All Results (Outside Threshold):")
    else:
        st.success(f"✅ Found {len(within_threshold)} matches within threshold")
    
    # Display results in a grid
    for i, result in enumerate(results):
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 2])
            
            with col1:
                # Try to display the matched image
                image_path = result.get('path')
                if image_path and os.path.exists(image_path):
                    try:
                        image = Image.open(image_path)
                        st.image(image, width=150)
                    except Exception:
                        st.write("🖼️ Image preview unavailable")
                else:
                    st.write("🖼️ Image not found")
            
            with col2:
                # Result details
                status_icon = "✅" if result['within_threshold'] else "❌"
                st.markdown(f"**{status_icon} Rank #{result['rank']}**")
                st.markdown(f"**File:** {result['filename']}")
                st.markdown(f"**Distance:** {result['distance']:.2f}")
                st.markdown(f"**Similarity:** {result['similarity_score']:.3f}")
            
            with col3:
                # Path information
                st.markdown("**Path Information:**")
                if result.get('path'):
                    st.code(result['path'], language=None)
                else:
                    st.code(result['relative_path'], language=None)
                
                # Status
                if result['within_threshold']:
                    st.markdown('<span class="status-good">✅ Within Threshold</span>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown('<span class="status-error">❌ Outside Threshold</span>', 
                              unsafe_allow_html=True)
        
        st.markdown("---")
    
    # Summary statistics
    st.markdown("### 📊 Search Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Results", len(results))
    with col2:
        st.metric("Within Threshold", len(within_threshold))
    with col3:
        avg_distance = sum(r['distance'] for r in results) / len(results) if results else 0
        st.metric("Average Distance", f"{avg_distance:.2f}")
