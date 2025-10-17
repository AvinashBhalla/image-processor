# core/distributed_processor.py

from multiprocessing import Pool, Queue
import dask.array as da
from dask.distributed import Client
import numpy as np

class DistributedProcessor:
    """
    Process large datasets using distributed computing
    """
    
    def __init__(self, n_workers: int = 4):
        self.n_workers = n_workers
        self.client = None
    
    def initialize_cluster(self, scheduler_address: str = None):
        """Initialize Dask cluster"""
        if scheduler_address:
            self.client = Client(scheduler_address)
        else:
            self.client = Client(n_workers=self.n_workers)
    
    def distributed_feature_extraction(self, 
                                      image_paths: List[str],
                                      extractor) -> np.ndarray:
        """Extract features using distributed workers"""
        import dask.bag as db
        
        # Create Dask bag from image paths
        bag = db.from_sequence(image_paths, partition_size=100)
        
        # Map extraction function
        features = bag.map(lambda path: self._extract_single(path, extractor))
        
        # Compute in parallel
        results = features.compute()
        
        return np.array(results)
    
    def _extract_single(self, image_path: str, extractor):
        """Extract features from single image"""
        import cv2
        img = cv2.imread(image_path)
        return extractor.extract(img)