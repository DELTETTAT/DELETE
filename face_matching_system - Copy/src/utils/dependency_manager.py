
import logging
import importlib
import subprocess
import sys
from typing import Dict, List, Tuple, Optional

class DependencyManager:
    """
    Manages and validates system dependencies at runtime.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Core dependencies
        self.core_dependencies = {
            'numpy': 'numpy',
            'opencv': 'cv2', 
            'pillow': 'PIL',
            'tensorflow': 'tensorflow',
            'deepface': 'deepface',
            'faiss': ['faiss', 'faiss_cpu'],
            'mediapipe': 'mediapipe',
            'plotly': 'plotly',
            'streamlit': 'streamlit'
        }
        
        # Optional dependencies
        self.optional_dependencies = {
            'psutil': 'psutil',
            'tqdm': 'tqdm'
        }
        
        self.dependency_status = {}
        self.missing_dependencies = []
        
    def check_all_dependencies(self) -> Dict[str, bool]:
        """Check all dependencies and return status"""
        self.dependency_status = {}
        self.missing_dependencies = []
        
        # Check core dependencies
        for name, modules in self.core_dependencies.items():
            status = self._check_dependency(modules)
            self.dependency_status[name] = status
            if not status:
                self.missing_dependencies.append(name)
        
        # Check optional dependencies
        for name, module in self.optional_dependencies.items():
            status = self._check_dependency(module)
            self.dependency_status[name] = status
            
        return self.dependency_status
    
    def _check_dependency(self, modules) -> bool:
        """Check if a dependency is available"""
        if isinstance(modules, str):
            modules = [modules]
        elif not isinstance(modules, list):
            modules = [modules]
            
        for module in modules:
            try:
                importlib.import_module(module)
                return True
            except ImportError:
                continue
        return False
    
    def get_missing_core_dependencies(self) -> List[str]:
        """Get list of missing core dependencies"""
        if not self.dependency_status:
            self.check_all_dependencies()
        return self.missing_dependencies
    
    def validate_runtime_environment(self) -> Tuple[bool, List[str]]:
        """Validate if the runtime environment is suitable"""
        self.check_all_dependencies()
        
        critical_missing = []
        for dep in ['numpy', 'opencv', 'pillow', 'streamlit']:
            if not self.dependency_status.get(dep, False):
                critical_missing.append(dep)
        
        is_valid = len(critical_missing) == 0
        return is_valid, critical_missing
    
    def suggest_installation_commands(self) -> Dict[str, str]:
        """Suggest installation commands for missing dependencies"""
        commands = {}
        
        package_map = {
            'opencv': 'opencv-python',
            'pillow': 'Pillow',
            'faiss': 'faiss-cpu',
            'deepface': 'deepface',
            'tensorflow': 'tensorflow',
            'mediapipe': 'mediapipe',
            'plotly': 'plotly',
            'streamlit': 'streamlit',
            'numpy': 'numpy',
            'psutil': 'psutil',
            'tqdm': 'tqdm'
        }
        
        for dep in self.missing_dependencies:
            if dep in package_map:
                commands[dep] = f"pip install {package_map[dep]}"
        
        return commands
    
    def check_model_dependencies(self, model_name: str) -> bool:
        """Check if dependencies for a specific model are available"""
        model_deps = {
            'VGG-Face': ['tensorflow', 'deepface'],
            'Facenet': ['tensorflow', 'deepface'],
            'Facenet512': ['tensorflow', 'deepface'],
            'OpenFace': ['deepface'],
            'DeepFace': ['tensorflow', 'deepface'],
            'DeepID': ['tensorflow', 'deepface'],
            'ArcFace': ['tensorflow', 'deepface'],
            'Dlib': ['deepface'],
            'SFace': ['opencv', 'deepface']
        }
        
        required_deps = model_deps.get(model_name, ['deepface'])
        
        for dep in required_deps:
            if dep in self.core_dependencies:
                if not self.dependency_status.get(dep, False):
                    return False
        
        return True
    
    def check_detector_dependencies(self, detector_name: str) -> bool:
        """Check if dependencies for a specific detector are available"""
        detector_deps = {
            'opencv': ['opencv'],
            'ssd': ['tensorflow', 'deepface'],
            'dlib': ['deepface'],
            'mtcnn': ['tensorflow', 'deepface'],
            'retinaface': ['tensorflow', 'deepface'],
            'mediapipe': ['mediapipe']
        }
        
        required_deps = detector_deps.get(detector_name, [])
        
        for dep in required_deps:
            if dep in self.core_dependencies:
                if not self.dependency_status.get(dep, False):
                    return False
        
        return True
    
    def get_available_models(self) -> List[str]:
        """Get list of available models based on dependencies"""
        all_models = [
            "VGG-Face", "Facenet", "Facenet512", "OpenFace", 
            "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"
        ]
        
        available = []
        for model in all_models:
            if self.check_model_dependencies(model):
                available.append(model)
        
        return available
    
    def get_available_detectors(self) -> List[str]:
        """Get list of available detectors based on dependencies"""
        all_detectors = [
            "opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"
        ]
        
        available = []
        for detector in all_detectors:
            if self.check_detector_dependencies(detector):
                available.append(detector)
        
        return available
    
    def auto_install_missing(self, core_only: bool = True) -> Dict[str, bool]:
        """Attempt to auto-install missing dependencies (use with caution)"""
        results = {}
        
        dependencies_to_install = self.missing_dependencies if not core_only else [
            dep for dep in self.missing_dependencies 
            if dep in ['numpy', 'opencv', 'pillow', 'streamlit']
        ]
        
        commands = self.suggest_installation_commands()
        
        for dep in dependencies_to_install:
            if dep in commands:
                try:
                    self.logger.info(f"Attempting to install {dep}...")
                    result = subprocess.run(
                        commands[dep].split(), 
                        capture_output=True, 
                        text=True, 
                        timeout=300
                    )
                    results[dep] = result.returncode == 0
                    if results[dep]:
                        self.logger.info(f"Successfully installed {dep}")
                    else:
                        self.logger.error(f"Failed to install {dep}: {result.stderr}")
                except Exception as e:
                    self.logger.error(f"Error installing {dep}: {e}")
                    results[dep] = False
        
        # Re-check dependencies after installation
        self.check_all_dependencies()
        
        return results

# Global dependency manager instance
dependency_manager = DependencyManager()
