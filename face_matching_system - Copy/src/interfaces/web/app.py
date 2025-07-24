
import os
import sys

# Add src to path for imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import streamlit as st

# Validate dependencies first
try:
    from utils.dependency_manager import dependency_manager

    # Check critical dependencies
    is_valid, missing_deps = dependency_manager.validate_runtime_environment()

    if not is_valid:
        st.error(f"❌ Missing critical dependencies: {', '.join(missing_deps)}")
        st.info("Please install the following packages:")
        commands = dependency_manager.suggest_installation_commands()
        for dep in missing_deps:
            if dep in commands:
                st.code(commands[dep])
        st.stop()

    from config.settings import Settings
    from utils.logging.logger import setup_logging

    # Setup logging for web app
    setup_logging()

    # Initialize settings
    settings = Settings()

except ImportError as e:
    st.error(f"❌ Failed to import required modules: {e}")
    st.info("Please ensure all dependencies are installed and the project structure is correct.")
    st.stop()
except Exception as e:
    st.error(f"❌ System initialization error: {e}")
    st.stop()

# Page configuration with Netflix-style branding
st.set_page_config(
    page_title="FaceMatch",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎯",
    menu_items={
        'About': " Face Recognition System "
    }
)

# Netflix-inspired dark theme CSS
st.markdown("""
<style>
    /* Import Netflix-style fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global theme variables */
    :root {
        --netflix-red: #E50914;
        --netflix-dark: #141414;
        --netflix-black: #000000;
        --netflix-gray: #333333;
        --netflix-light-gray: #757575;
        --netflix-white: #ffffff;
        --accent-blue: #0ea5e9;
        --accent-green: #22c55e;
        --accent-yellow: #fbbf24;
        --accent-purple: #8b5cf6;
    }
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, var(--netflix-black) 0%, var(--netflix-dark) 100%);
        color: var(--netflix-white);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(90deg, var(--netflix-red) 0%, #dc2626 100%);
        color: var(--netflix-white);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(229, 9, 20, 0.3);
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, var(--accent-blue) 0%, var(--accent-purple) 100%);
        color: var(--netflix-white);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.2);
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(145deg, var(--netflix-gray) 0%, #2d2d2d 100%);
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        border: 1px solid #404040;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.4);
        border-color: var(--accent-blue);
    }
    
    /* Status indicators */
    .status-good { 
        color: var(--accent-green); 
        font-weight: 600;
        text-shadow: 0 0 8px rgba(34, 197, 94, 0.5);
    }
    .status-warning { 
        color: var(--accent-yellow); 
        font-weight: 600;
        text-shadow: 0 0 8px rgba(251, 191, 36, 0.5);
    }
    .status-error { 
        color: var(--netflix-red); 
        font-weight: 600;
        text-shadow: 0 0 8px rgba(229, 9, 20, 0.5);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--netflix-black) 0%, var(--netflix-gray) 100%);
        border-right: 2px solid var(--netflix-red);
    }
    
    .sidebar .sidebar-content {
        background: transparent;
        color: var(--netflix-white);
    }
    
    /* Navigation styling */
    .nav-button {
        background: linear-gradient(90deg, var(--netflix-gray) 0%, #404040 100%);
        color: var(--netflix-white);
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border: none;
        border-radius: 12px;
        font-weight: 500;
        transition: all 0.3s ease;
        cursor: pointer;
        border: 1px solid transparent;
    }
    
    .nav-button:hover {
        background: linear-gradient(90deg, var(--netflix-red) 0%, #dc2626 100%);
        transform: translateX(4px);
        border-color: var(--netflix-red);
        box-shadow: 0 4px 16px rgba(229, 9, 20, 0.3);
    }
    
    .nav-button.active {
        background: linear-gradient(90deg, var(--accent-blue) 0%, var(--accent-purple) 100%);
        border-color: var(--accent-blue);
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, var(--netflix-red) 0%, #dc2626 100%);
        color: var(--netflix-white);
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(229, 9, 20, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(229, 9, 20, 0.4);
        background: linear-gradient(90deg, #dc2626 0%, var(--netflix-red) 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Secondary button styling */
    .secondary-btn {
        background: linear-gradient(90deg, var(--accent-blue) 0%, #0284c7 100%) !important;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.3) !important;
    }
    
    .secondary-btn:hover {
        background: linear-gradient(90deg, #0284c7 0%, var(--accent-blue) 100%) !important;
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.4) !important;
    }
    
    /* Success button styling */
    .success-btn {
        background: linear-gradient(90deg, var(--accent-green) 0%, #16a34a 100%) !important;
        box-shadow: 0 4px 16px rgba(34, 197, 94, 0.3) !important;
    }
    
    /* Warning button styling */
    .warning-btn {
        background: linear-gradient(90deg, var(--accent-yellow) 0%, #f59e0b 100%) !important;
        color: var(--netflix-black) !important;
        box-shadow: 0 4px 16px rgba(251, 191, 36, 0.3) !important;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: var(--netflix-gray);
        color: var(--netflix-white);
        border: 2px solid #404040;
        border-radius: 12px;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-blue);
        box-shadow: 0 0 16px rgba(14, 165, 233, 0.3);
    }
    
    /* Select box styling */
    .stSelectbox > div > div > select {
        background: var(--netflix-gray);
        color: var(--netflix-white);
        border: 2px solid #404040;
        border-radius: 12px;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: var(--netflix-gray);
        border: 2px dashed var(--accent-blue);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: var(--netflix-red);
        background: #2d2d2d;
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background: var(--netflix-gray);
        border-radius: 10px;
        overflow: hidden;
    }
    
    .stProgress .st-bp {
        background: linear-gradient(90deg, var(--netflix-red) 0%, var(--accent-blue) 100%);
        border-radius: 10px;
    }
    
    /* Metrics styling */
    .metric-container {
        background: linear-gradient(145deg, var(--netflix-gray) 0%, #2d2d2d 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #404040;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: scale(1.02);
        border-color: var(--accent-blue);
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-blue);
        text-shadow: 0 0 8px rgba(14, 165, 233, 0.5);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--netflix-light-gray);
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    /* Folder path styling */
    .folder-path {
        background: linear-gradient(90deg, var(--netflix-gray) 0%, #2d2d2d 100%);
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid var(--accent-blue);
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    
    /* Success alert */
    .stAlert[data-baseweb="notification"] {
        background: linear-gradient(90deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%);
        border-left: 4px solid var(--accent-green);
    }
    
    /* Error alert */
    .stAlert[data-baseweb="notification"][kind="error"] {
        background: linear-gradient(90deg, rgba(229, 9, 20, 0.1) 0%, rgba(229, 9, 20, 0.05) 100%);
        border-left: 4px solid var(--netflix-red);
    }
    
    /* Warning alert */
    .stAlert[data-baseweb="notification"][kind="warning"] {
        background: linear-gradient(90deg, rgba(251, 191, 36, 0.1) 0%, rgba(251, 191, 36, 0.05) 100%);
        border-left: 4px solid var(--accent-yellow);
    }
    
    /* Info alert */
    .stAlert[data-baseweb="notification"][kind="info"] {
        background: linear-gradient(90deg, rgba(14, 165, 233, 0.1) 0%, rgba(14, 165, 233, 0.05) 100%);
        border-left: 4px solid var(--accent-blue);
    }
    
    /* Sidebar title styling */
    .sidebar-title {
        color: var(--netflix-red);
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem;
        margin-bottom: 1rem;
        text-shadow: 0 0 8px rgba(229, 9, 20, 0.5);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--netflix-gray);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--netflix-light-gray);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, var(--netflix-red) 0%, #dc2626 100%);
        color: var(--netflix-white);
        box-shadow: 0 4px 16px rgba(229, 9, 20, 0.3);
    }
    
    /* Image styling */
    .stImage {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    
    .stImage:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 32px rgba(0,0,0,0.4);
    }
    
    /* Code block styling */
    .stCode {
        background: var(--netflix-black);
        border: 1px solid #404040;
        border-radius: 12px;
        color: var(--accent-blue);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
            padding: 1.5rem;
        }
        
        .section-header {
            font-size: 1.1rem;
            padding: 0.75rem 1rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
    }
    
    /* Animation for loading states */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
    
    /* Hover effects for interactive elements */
    .interactive-card {
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .interactive-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 40px rgba(0,0,0,0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'custom_folder_path' not in st.session_state:
    st.session_state.custom_folder_path = ''
if 'current_page' not in st.session_state:
    st.session_state.current_page = '🔍 Face Search'

# Main header
st.markdown('<div class="main-header">🎯 FaceMatch </div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #757575; font-size: 1.1rem; margin-bottom: 2rem;"> Face Recognition System </p>', unsafe_allow_html=True)

# Sidebar navigation with Netflix styling
with st.sidebar:
    st.markdown('<div class="sidebar-title">🎯 FaceMatch </div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Navigation options
    pages = {
        "🔍 Face Search": {"icon": "🔍", "desc": "Search for similar faces"},
        "📊 Index Management": {"icon": "📊", "desc": "Build and manage face index"},
        "📁 File Management": {"icon": "📁", "desc": "Upload and organize files"},
        "📈 System Analytics": {"icon": "📈", "desc": "View system statistics"}
    }
    
    st.markdown("### Navigation")
    
    for page_name, page_info in pages.items():
        if st.button(
            f"{page_info['icon']} {page_name.split(' ', 1)[1]}",
            key=f"nav_{page_name}",
            help=page_info['desc'],
            use_container_width=True
        ):
            st.session_state.current_page = page_name
            st.rerun()
    
    # Show current page indicator
    current_page = st.session_state.current_page
    st.markdown("---")
    
    # Safety check for page existence
    if current_page in pages:
        st.markdown(f"**Current Page:** {pages[current_page]['icon']} {current_page.split(' ', 1)[1]}")
        st.markdown(f"*{pages[current_page]['desc']}*")
    else:
        # Fallback to default page if current page doesn't exist
        st.session_state.current_page = '🔍 Face Search'
        current_page = st.session_state.current_page
        st.markdown(f"**Current Page:** {pages[current_page]['icon']} {current_page.split(' ', 1)[1]}")
        st.markdown(f"*{pages[current_page]['desc']}*")
    
    # Quick status overview
    st.markdown("---")
    st.markdown("### Quick Status")
    
    try:
        from core.orchestrator import FaceMatchingOrchestrator
        orchestrator = FaceMatchingOrchestrator()
        system_status = orchestrator.get_system_status()
        index_ready = system_status.get('index', {}).get('ready', False)
        
        if index_ready:
            st.markdown('<div class="status-good">✅ System Ready</div>', unsafe_allow_html=True)
            stats = orchestrator.get_index_stats()
            st.metric("Indexed Faces", stats.get('total_faces', 0))
        else:
            st.markdown('<div class="status-warning">⚠️ No Index Found</div>', unsafe_allow_html=True)
            st.info("Build an index to start searching")
    except Exception as e:
        st.markdown('<div class="status-error">❌ System Error</div>', unsafe_allow_html=True)

# Main content area based on page selection
if st.session_state.current_page == "🔍 Face Search":
    from pages.search import render_search_page
    render_search_page()
elif st.session_state.current_page == "📊 Index Management":
    from pages.index import render_index_page
    render_index_page()
elif st.session_state.current_page == "📁 File Management":
    from pages.files import render_files_page
    render_files_page()
elif st.session_state.current_page == "📈 System Analytics":
    from pages.analytics import render_analytics_page
    render_analytics_page()

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #757575; font-size: 0.9rem; padding: 1rem 0;">FaceMatch - Face Recognition System</p>',
    unsafe_allow_html=True
)
