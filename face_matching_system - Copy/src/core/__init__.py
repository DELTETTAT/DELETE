"""
Core modules for face matching system.

Contains the main business logic components:
- preprocessing: Image enhancement, alignment, and quality assessment
- embedding: Face embedding extraction
- indexing: FAISS index management
- search: Similarity search engine
"""

from .preprocessing.preprocessor import ImagePreprocessor
from .embedding.extractor import EmbeddingExtractor
from .indexing.manager import IndexManager
from .search.engine import SearchEngine

__all__ = [
    'ImagePreprocessor',
    'EmbeddingExtractor', 
    'IndexManager',
    'SearchEngine'
]
