# core/memory_efficient.py

import numpy as np
from typing import Iterator, Tuple

class MemoryEfficientProcessor:
    """
    Process images with minimal memory footprint
    """
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
    
    def chunked_processing(self, 
                          image_paths: List[str],
                          process_func) -> Iterator[Tuple[int, any]]:
        """
        Process images in chunks to manage memory
        """
        for i in range(0, len(image_paths), self.chunk_size):
            chunk = image_paths[i:i + self.chunk_size]
            
            for idx, path in enumerate(chunk):
                result = process_func(path)
                yield (i + idx, result)
            
            # Force garbage collection after each chunk
            import gc
            gc.collect()
    
    def memory_mapped_index(self, 
                           features_path: str,
                           shape: Tuple[int, int]) -> np.memmap:
        """
        Create memory-mapped array for large feature matrices
        """
        mmap = np.memmap(
            features_path,
            dtype='float32',
            mode='w+',
            shape=shape
        )
        return mmap