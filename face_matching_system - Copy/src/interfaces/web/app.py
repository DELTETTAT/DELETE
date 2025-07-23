import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
from config.settings import Settings
from utils.logging.logger import setup_logging

# Setup logging for web app
setup_logging()

# Initialize settings
settings = Settings()

# Page configuration
st.set_page_config(
    page_title="Advanced Face Matcher",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Advanced Face Matching System with comprehensive management tools"
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-good { color: #28a745; }
    .status-warning { color: #ffc107; }
    .status-error { color: #dc3545; }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .folder-path {
        background: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 3px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'custom_folder_path' not in st.session_state:
    st.session_state.custom_folder_path = ''

# Sidebar navigation
st.sidebar.title("🎯 Face Matching System")
st.sidebar.markdown("---")

# Page selection
page = st.sidebar.radio(
    "Select Page:",
    ["🔍 Face Search", "📊 Index Management", "📁 File Management", "📈 Analytics"]
)

# Main content area based on page selection
if page == "🔍 Face Search":
    from pages.search import render_search_page
    render_search_page()
elif page == "📊 Index Management":
    from pages.index import render_index_page
    render_index_page()
elif page == "📁 File Management":
    from pages.files import render_files_page
    render_files_page()
elif page == "📈 Analytics":
    from pages.analytics import render_analytics_page
    render_analytics_page()
