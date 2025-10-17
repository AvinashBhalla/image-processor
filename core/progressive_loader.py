# core/progressive_loader.py

class ProgressiveImageLoader:
    """
    Load images progressively as needed
    """
    
    def __init__(self, image_paths: List[str], cache_size: int = 100):
        self.image_paths = image_paths
        self.cache_size = cache_size
        self.cache = {}
        self.access_count = {}
    
    def load_image(self, index: int) -> np.ndarray:
        """Load image with LRU caching"""
        path = self.image_paths[index]
        
        if path in self.cache:
            self.access_count[path] += 1
            return self.cache[path]
        
        # Load image
        img = cv2.imread(path)
        
        # Manage cache size
        if len(self.cache) >= self.cache_size:
            # Remove least recently used
            lru_path = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_path]
            del self.access_count[lru_path]
        
        # Add to cache
        self.cache[path] = img
        self.access_count[path] = 1
        
        return img