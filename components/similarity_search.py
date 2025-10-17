# components/similarity_search.py

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path

@dataclass
class SearchResult:
    """Container for search results"""
    image_path: Path
    similarity_score: float
    embedding: np.ndarray
    metadata: dict

class SimilaritySearchEngine:
    """
    Multi-modal similarity search supporting various feature extractors
    """
    def __init__(self, config: dict):
        self.config = config
        self.feature_extractors = {}
        self.index = None
        self.metadata_db = None
        
    def initialize(self):
        """Initialize all components"""
        self._load_feature_extractors()
        self._load_or_create_index()
        self._initialize_database()