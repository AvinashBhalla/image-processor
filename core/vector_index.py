# core/vector_index.py

import faiss
import numpy as np
from typing import List, Tuple
import pickle
from pathlib import Path

class VectorIndexManager:
    """
    Manages FAISS vector index for fast similarity search
    """
    
    def __init__(self, dimension: int, index_path: str = "data/faiss.index"):
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.index = None
        self.id_to_path = {}
        
    def create_index(self, use_gpu: bool = False):
        """
        Create FAISS index optimized for the dataset size
        """
        # For small datasets (< 10k images): Flat index
        # For medium datasets (10k - 100k): IVF index
        # For large datasets (> 100k): HNSW or IVF+PQ
        
        # Use Flat index for simplicity and small datasets
        self.index = faiss.IndexFlatL2(self.dimension)
        
        if use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
            
    def train_and_add(self, embeddings: np.ndarray, 
                     image_paths: List[str]):
        """
        Train index and add vectors
        
        Args:
            embeddings: Array of shape (n_images, dimension)
            image_paths: List of image paths corresponding to embeddings
        """
        # Flat index doesn't need training
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            print("Training index...")
            self.index.train(embeddings.astype('float32'))
            
        print("Adding vectors to index...")
        self.index.add(embeddings.astype('float32'))
        
        # Store mapping from index ID to image path
        for idx, path in enumerate(image_paths):
            self.id_to_path[idx] = path
            
    def search(self, query_embedding: np.ndarray, 
              k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for k most similar images
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            
        Returns:
            List of (image_path, distance) tuples
        """
        query = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # -1 indicates no result
                # Convert L2 distance to similarity score (0-1)
                similarity = 1 / (1 + dist)
                results.append((self.id_to_path[idx], similarity))
                
        return results
    
    def save(self):
        """Save index and metadata"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(faiss.index_gpu_to_cpu(self.index) 
                         if hasattr(self.index, 'gpu') 
                         else self.index,
                         str(self.index_path))
        
        # Save ID mapping
        with open(self.index_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(self.id_to_path, f)
            
    def load(self):
        """Load existing index"""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            
            with open(self.index_path.with_suffix('.pkl'), 'rb') as f:
                self.id_to_path = pickle.load(f)
            return True
        return False