# core/optimizations.py

import numpy as np
from functools import lru_cache
import hashlib

class PerformanceOptimizer:
    """
    Collection of optimization techniques
    """
    
    @staticmethod
    def smart_resize(image: np.ndarray, max_dimension: int = 1024) -> np.ndarray:
        """
        Intelligently resize images to reduce processing time
        """
        h, w = image.shape[:2]
        
        if max(h, w) <= max_dimension:
            return image
        
        # Calculate new dimensions
        if h > w:
            new_h = max_dimension
            new_w = int(w * (max_dimension / h))
        else:
            new_w = max_dimension
            new_h = int(h * (max_dimension / w))
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def cached_file_hash(file_path: str) -> str:
        """
        Cache file hashes to avoid reprocessing
        """
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    
    @staticmethod
    def quantize_features(features: np.ndarray, bits: int = 8) -> np.ndarray:
        """
        Quantize features to reduce memory footprint
        """
        # Normalize to [0, 1]
        features_norm = (features - features.min()) / (features.max() - features.min())
        
        # Quantize to n bits
        max_val = 2 ** bits - 1
        quantized = np.round(features_norm * max_val).astype(np.uint8)
        
        return quantized
    
    @staticmethod
    def use_pca_compression(features: np.ndarray, 
                           n_components: int = 128) -> np.ndarray:
        """
        Use PCA to reduce feature dimensionality
        """
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=n_components)
        compressed = pca.fit_transform(features)
        
        return compressed


# Monitoring and profiling

class PerformanceMonitor:
    """
    Monitor application performance
    """
    
    def __init__(self):
        self.metrics = {}
    
    def track_time(self, operation: str):
        """Decorator to track operation time"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                import time
                start = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                
                if operation not in self.metrics:
                    self.metrics[operation] = []
                self.metrics[operation].append(elapsed)
                
                return result
            return wrapper
        return decorator
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics"""
        stats = {}
        
        for operation, times in self.metrics.items():
            stats[operation] = {
                'mean': np.mean(times),
                'median': np.median(times),
                'min': np.min(times),
                'max': np.max(times),
                'total': np.sum(times),
                'count': len(times)
            }
        
        return stats
    
    def print_report(self):
        """Print performance report"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        
        for operation, metrics in stats.items():
            print(f"\n{operation}:")
            print(f"  Mean time:   {metrics['mean']:.4f}s")
            print(f"  Median time: {metrics['median']:.4f}s")
            print(f"  Min time:    {metrics['min']:.4f}s")
            print(f"  Max time:    {metrics['max']:.4f}s")
            print(f"  Total time:  {metrics['total']:.4f}s")
            print(f"  Call count:  {metrics['count']}")