# core/batch_processor.py

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Callable, Any
import numpy as np
from tqdm import tqdm

class BatchProcessor:
    """
    Optimized batch processing for large image datasets
    """
    
    def __init__(self, 
                 n_workers: int = None,
                 use_gpu: bool = False,
                 batch_size: int = 32):
        self.n_workers = n_workers or mp.cpu_count()
        self.use_gpu = use_gpu
        self.batch_size = batch_size
    
    def process_images_parallel(self, 
                               image_paths: List[str],
                               process_func: Callable,
                               use_threading: bool = False) -> List[Any]:
        """
        Process images in parallel using multiprocessing or threading
        
        Args:
            image_paths: List of image paths
            process_func: Function to apply to each image
            use_threading: Use threading instead of multiprocessing
                          (better for I/O-bound tasks)
        
        Returns:
            List of processing results
        """
        executor_class = ThreadPoolExecutor if use_threading else ProcessPoolExecutor
        
        with executor_class(max_workers=self.n_workers) as executor:
            results = list(tqdm(
                executor.map(process_func, image_paths),
                total=len(image_paths),
                desc="Processing images"
            ))
        
        return results
    
    def batch_extract_features(self, 
                              image_paths: List[str],
                              feature_extractor: Any) -> np.ndarray:
        """
        Extract features in batches for GPU efficiency
        """
        all_features = []
        
        for i in tqdm(range(0, len(image_paths), self.batch_size),
                     desc="Extracting features"):
            batch_paths = image_paths[i:i + self.batch_size]
            
            # Load batch of images
            batch_images = []
            valid_paths = []
            
            for path in batch_paths:
                try:
                    img = cv2.imread(path)
                    if img is not None:
                        batch_images.append(img)
                        valid_paths.append(path)
                except:
                    continue
            
            if not batch_images:
                continue
            
            # Extract features for batch
            batch_features = self._extract_batch(batch_images, feature_extractor)
            all_features.extend(batch_features)
        
        return np.array(all_features)
    
    def _extract_batch(self, images: List[np.ndarray], 
                      feature_extractor: Any) -> List[np.ndarray]:
        """Extract features from a batch of images"""
        # This can be optimized for specific extractors
        # For now, process individually
        features = []
        for img in images:
            feature = feature_extractor.extract(img)
            features.append(feature)
        
        return features


class IncrementalIndexer:
    """
    Incrementally index new images without reprocessing entire dataset
    """
    
    def __init__(self, database: ImageDatabase, 
                 index_manager: VectorIndexManager):
        self.database = database
        self.index_manager = index_manager
        self.indexed_paths = set()
        self._load_indexed_paths()
    
    def _load_indexed_paths(self):
        """Load already indexed image paths"""
        indexed_images = self.database.get_all_indexed_images()
        self.indexed_paths = {img['file_path'] for img in indexed_images}
    
    def find_new_images(self, directory: str) -> List[str]:
        """Find images that haven't been indexed yet"""
        all_images = self._scan_directory(directory)
        new_images = [img for img in all_images if img not in self.indexed_paths]
        
        print(f"Found {len(new_images)} new images out of {len(all_images)} total")
        return new_images
    
    def _scan_directory(self, directory: str) -> List[str]:
        """Scan directory for image files"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(directory).rglob(f'*{ext}'))
            image_files.extend(Path(directory).rglob(f'*{ext.upper()}'))
        
        return [str(f) for f in image_files]
    
    def incremental_update(self, new_images: List[str], 
                          feature_extractor: Any):
        """Add new images to existing index"""
        print(f"Incrementally indexing {len(new_images)} new images...")
        
        # Extract features
        batch_processor = BatchProcessor()
        features = batch_processor.batch_extract_features(
            new_images, feature_extractor
        )
        
        # Add to index
        self.index_manager.train_and_add(features, new_images)
        
        # Update database
        for img_path, feature in zip(new_images, features):
            metadata = self._extract_metadata(img_path)
            img_id = self.database.add_image(img_path, metadata)
            self.database.cache_features(img_id, 'clip', feature)
        
        # Save updated index
        self.index_manager.save()
        
        print("Incremental indexing complete!")
    
    def _extract_metadata(self, image_path: str) -> Dict:
        """Extract metadata from image file"""
        path = Path(image_path)
        metadata = {
            'file_name': path.name,
            'file_size': path.stat().st_size,
            'modified_date': datetime.fromtimestamp(path.stat().st_mtime)
        }
        
        try:
            img = cv2.imread(image_path)
            if img is not None:
                metadata['height'], metadata['width'] = img.shape[:2]
                metadata['format'] = path.suffix[1:].upper()
        except:
            pass
        
        return metadata