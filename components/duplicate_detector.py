# components/duplicate_detector.py

from typing import List, Set, Dict, Tuple
from collections import defaultdict
import numpy as np
from pathlib import Path

class DuplicateDetector:
    """
    Multi-stage duplicate detection system
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.hash_threshold = config.get('hash_threshold', 5)
        self.embedding_threshold = config.get('embedding_threshold', 0.95)
        self.ssim_threshold = config.get('ssim_threshold', 0.90)
        
    def find_duplicates(self, image_paths: List[str]) -> Dict[str, List[str]]:
        """
        Find duplicate and near-duplicate images
        
        Returns:
            Dictionary mapping representative image to list of duplicates
        """
        # Stage 1: Fast filtering with perceptual hashing
        hash_groups = self._group_by_perceptual_hash(image_paths)
        
        # Stage 2: Refine with deep learning embeddings
        embedding_groups = self._refine_with_embeddings(hash_groups)
        
        # Stage 3: Final verification with SSIM
        final_groups = self._verify_with_ssim(embedding_groups)
        
        return final_groups