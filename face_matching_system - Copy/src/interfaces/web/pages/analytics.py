import os
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any
import importlib

def validate_system_dependencies() -> bool:
    """Validate that required dependencies are available"""
    required_packages = [
        'deepface', 'faiss', 'cv2', 'numpy', 'PIL', 
        'tensorflow', 'mediapipe', 'plotly'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'cv2':
                importlib.import_module('cv2')
            elif package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'faiss':
                # Try both CPU and GPU versions
                try:
                    importlib.import_module('faiss')
                except ImportError:
                    importlib.import_module('faiss_cpu')
            else:
                importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        st.warning(f"⚠️ Missing packages: {', '.join(missing_packages)}")
        return False
    
    return True

def safe_directory_check(path: str) -> Dict[str, Any]:
    """Safely check directory status with proper error handling"""
    try:
        if not path:
            return {'exists': False, 'readable': False, 'writable': False, 'error': 'Path is empty'}
        
        exists = os.path.exists(path)
        readable = os.access(path, os.R_OK) if exists else False
        writable = os.access(path, os.W_OK) if exists else False
        
        return {
            'exists': exists,
            'readable': readable,
            'writable': writable,
            'path': path,
            'error': None
        }
    except Exception as e:
        return {
            'exists': False,
            'readable': False,
            'writable': False,
            'path': path,
            'error': str(e)
        }

def render_analytics_page():
    """Render the analytics page with robust error handling"""

    st.markdown('<h1 class="main-header">📊 System Analytics</h1>', unsafe_allow_html=True)

    try:
        # Import with proper error handling
        import sys
        import os
        
        # Add the src directory to path more robustly
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(current_dir, '..', '..', '..')
        src_path = os.path.normpath(src_path)
        
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Import components with error handling
        try:
            from core.orchestrator import FaceMatchingOrchestrator
            from utils.filesystem.operations import FileSystemOperations
            from config.settings import Settings
        except ImportError as e:
            st.error(f"❌ Failed to import required modules: {e}")
            st.info("Please ensure all dependencies are installed and the project structure is correct.")
            return

        # Initialize components with error handling
        try:
            orchestrator = FaceMatchingOrchestrator()
            fs_ops = FileSystemOperations()
            settings = Settings()
        except Exception as e:
            st.error(f"❌ Failed to initialize components: {e}")
            return

        # Validate dependencies
        if not validate_system_dependencies():
            st.warning("⚠️ Some system dependencies may be missing. Analytics may be incomplete.")

        # Render different analytics sections with error boundaries
        try:
            render_system_overview(orchestrator, fs_ops, settings)
        except Exception as e:
            st.error(f"❌ Error in system overview: {e}")
            
        try:
            render_index_analytics(orchestrator)
        except Exception as e:
            st.error(f"❌ Error in index analytics: {e}")
            
        try:
            render_file_statistics(fs_ops, settings)
        except Exception as e:
            st.error(f"❌ Error in file statistics: {e}")
            
        try:
            render_performance_metrics(orchestrator)
        except Exception as e:
            st.error(f"❌ Error in performance metrics: {e}")
            
    except Exception as e:
        st.error(f"❌ Critical error in analytics page: {e}")
        st.info("Please check the system logs for more details.")

def render_system_overview(orchestrator, fs_ops, settings):
    """Render system overview analytics with robust error handling"""

    st.markdown("## 🖥️ System Overview")

    try:
        # Get system status with error handling
        system_status = orchestrator.get_system_status()
        
        if 'error' in system_status:
            st.error(f"❌ System status error: {system_status['error']}")
            return

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            try:
                index_ready = system_status.get('index', {}).get('ready', False)
                status_text = "✅ Ready" if index_ready else "❌ Not Ready"
                st.metric("Index Status", status_text)
            except Exception as e:
                st.metric("Index Status", f"❌ Error: {str(e)[:20]}...")

        with col2:
            try:
                total_faces = system_status.get('index', {}).get('stats', {}).get('total_faces', 0)
                st.metric("Indexed Faces", total_faces)
            except Exception as e:
                st.metric("Indexed Faces", "❌ Error")

        with col3:
            try:
                source_folder = st.session_state.get('custom_folder_path', '')
                source_images = 0
                if source_folder:
                    dir_status = safe_directory_check(source_folder)
                    if dir_status['exists'] and dir_status['readable']:
                        source_images = fs_ops.count_images_in_folder(source_folder)
                    elif dir_status['error']:
                        st.warning(f"Source folder error: {dir_status['error']}")
                st.metric("Source Images", source_images)
            except Exception as e:
                st.metric("Source Images", "❌ Error")

        with col4:
            try:
                upload_dir_status = safe_directory_check(settings.UPLOAD_DIR)
                if upload_dir_status['exists'] and upload_dir_status['readable']:
                    uploaded_images = fs_ops.count_images_in_folder(settings.UPLOAD_DIR)
                else:
                    uploaded_images = 0
                    if upload_dir_status['error']:
                        st.warning(f"Upload directory error: {upload_dir_status['error']}")
                st.metric("Uploaded Images", uploaded_images)
            except Exception as e:
                st.metric("Uploaded Images", "❌ Error")
                
    except Exception as e:
        st.error(f"❌ Error rendering system overview: {e}")

def render_index_analytics(orchestrator):
    """Render index-specific analytics"""

    st.markdown("## 📈 Index Analytics")

    index_stats = orchestrator.get_index_stats()

    if 'error' in index_stats:
        st.warning(f"❌ {index_stats['error']}")
        return

    # Index composition
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Index Composition")

        # Create sample distribution data based on available stats
        if index_stats.get('total_faces', 0) > 0:
            total_faces = index_stats.get('total_faces', 0)

            # Create a simple visualization of index size vs dimension
            dimension_data = {
                'Metric': ['Indexed Faces', 'Total Capacity (est.)', 'Embedding Dimension'],
                'Value': [
                    total_faces,
                    total_faces * 2,  # Estimated capacity
                    index_stats.get('embedding_dimension', 0)
                ]
            }

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=dimension_data['Metric'],
                y=dimension_data['Value'],
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
            ))
            fig.update_layout(
                title="Index Metrics",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🔧 Technical Details")

        # Detailed index information
        details = {
            "Index Type": "FAISS IndexFlatL2",
            "Distance Metric": "L2 (Euclidean)",
            "Storage Format": "Binary",
            "Memory Usage": "In-Memory + Disk",
            "Search Complexity": "O(n)",
            "Insert Complexity": "O(1)"
        }

        for key, value in details.items():
            st.write(f"**{key}:** {value}")

    # Timeline if creation date is available
    if index_stats.get('created_at'):
        st.markdown("### 📅 Index Timeline")

        try:
            created_date = datetime.fromisoformat(index_stats['created_at'].replace('Z', '+00:00'))
            current_date = datetime.now()
            age_days = (current_date - created_date).days

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Index Age", f"{age_days} days")
            with col2:
                st.metric("Created", created_date.strftime("%Y-%m-%d"))
            with col3:
                st.metric("Last Modified", "Today")  # Simplified

        except Exception:
            st.write("Unable to parse creation date")

def render_performance_metrics(orchestrator):
    """Render performance metrics with detailed analysis"""

    st.markdown("## ⚡ Performance Metrics")

    try:
        stats = orchestrator.get_index_stats()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 Model Performance")

            # Calculate accuracy metrics
            total_faces = stats.get('total_faces', 0)
            successful_embeddings = stats.get('successful_embeddings', total_faces)
            filtered_embeddings = stats.get('filtered_embeddings', 0)

            if total_faces > 0:
                success_rate = (successful_embeddings / (successful_embeddings + filtered_embeddings)) * 100
                quality_rate = (successful_embeddings / total_faces) * 100 if total_faces > 0 else 0
            else:
                success_rate = 0
                quality_rate = 0

            st.metric("Embedding Success Rate", f"{success_rate:.1f}%")
            st.metric("Quality Pass Rate", f"{quality_rate:.1f}%")
            st.metric("Total Processed", f"{total_faces:,}")
            st.metric("Successfully Indexed", f"{successful_embeddings:,}")
            st.metric("Quality Filtered", f"{filtered_embeddings:,}")

        with col2:
            st.markdown("### 🔧 Technical Specs")

            performance_metrics = {
                "Embedding Model": stats.get('model_used', 'Unknown'),
                "Detector Backend": stats.get('detector_used', 'Unknown'), 
                "Embedding Dimension": f"{stats.get('embedding_dimension', 0)}D",
                "Preprocessing": "✅ Enabled" if stats.get('preprocessing_used') else "❌ Disabled",
                "Index Type": "FAISS IndexFlatL2",
                "Distance Metric": "Euclidean (L2)"
            }

            for metric, value in performance_metrics.items():
                st.write(f"**{metric}:** {value}")

        # Performance analysis
        st.markdown("### 📊 Quality Analysis")

        if filtered_embeddings > 0:
            st.warning(f"⚠️ {filtered_embeddings} images were filtered out due to quality issues")
            filter_reasons = [
                "Face detection failed",
                "Image quality too low", 
                "Multiple faces detected",
                "Face too small or blurry"
            ]
            st.write("**Common filtering reasons:**")
            for reason in filter_reasons:
                st.write(f"• {reason}")
        else:
            st.success("✅ All processed images passed quality checks")

    except Exception as e:
        st.error(f"Unable to load performance metrics: {e}")

def render_file_statistics(fs_ops, settings):
    """Render file system statistics"""

    st.markdown("## 📁 File Statistics")

    # Directory analysis
    directories = {
        "Source": st.session_state.get('custom_folder_path', ''),
        "Uploads": settings.UPLOAD_DIR,
        "Preprocessed": settings.PREPROCESSED_DIR,
        "Embeddings": settings.EMBED_DIR,
        "Temp": settings.TEMP_DIR
    }

    # Collect directory statistics
    dir_stats = []
    total_images = 0
    total_size = 0

    for name, path in directories.items():
        if path and os.path.exists(path):
            image_count = fs_ops.count_images_in_folder(path)
            dir_size = get_directory_size(path)

            dir_stats.append({
                'Directory': name,
                'Images': image_count,
                'Size (MB)': dir_size,
                'Path': path
            })

            total_images += image_count
            total_size += dir_size
        else:
            dir_stats.append({
                'Directory': name,
                'Images': 0,
                'Size (MB)': 0,
                'Path': path or 'Not set'
            })

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Images", total_images)
    with col2:
        st.metric("Total Size (MB)", f"{total_size:.2f}")
    with col3:
        st.metric("Directories", len([d for d in dir_stats if d['Images'] > 0]))

    # Directory breakdown table
    st.markdown("### Directory Breakdown")
    st.table(dir_stats)

def get_directory_size(path: str) -> float:
    """Calculate directory size in MB with memory management and better estimation"""
    try:
        if not path or not os.path.exists(path):
            return 0.0

        total_size = 0
        file_count = 0
        max_files = 5000  # Reduced limit for better performance

        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if file_count >= max_files:
                    # Better estimation for remaining files
                    avg_size = total_size / file_count if file_count > 0 else 1024  # Default 1KB
                    remaining_files = sum(len(files) for _, _, files in os.walk(path)) - file_count
                    estimated_remaining = remaining_files * avg_size
                    total_size += estimated_remaining
                    break

                try:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath) and os.path.isfile(filepath):
                        file_size = os.path.getsize(filepath)
                        total_size += file_size
                        file_count += 1
                except (OSError, IOError):
                    # Skip files that can't be accessed
                    continue

        return total_size / (1024 * 1024)  # Convert to MB
    except Exception as e:
        return 0.0

def cleanup_temp_resources():
    """Clean up temporary resources and memory"""
    try:
        import gc
        gc.collect()  # Force garbage collection
        
        # Clear any cached Streamlit elements
        if hasattr(st, 'cache_data'):
            st.cache_data.clear()
        if hasattr(st, 'cache_resource'):
            st.cache_resource.clear()
            
    except Exception:
        pass  # Silently handle cleanup errors