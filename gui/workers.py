# Background worker threads

from PyQt6.QtCore import QThread, pyqtSignal
from pathlib import Path
from datetime import datetime
from typing import Dict
import numpy as np
import cv2

class IndexingThread(QThread):
    """Background thread for indexing images"""
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    
    def __init__(self, directory: str):
        super().__init__()
        self.directory = directory
    
    def run(self):
        """Run indexing process"""
        try:
            # Initialize components
            from core.feature_extractors import MultiModalFeatureExtractor
            from core.vector_index import VectorIndexManager
            from core.database import ImageDatabase
            
            feature_extractor = MultiModalFeatureExtractor()
            database = ImageDatabase()
            
            # Scan directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(Path(self.directory).rglob(f'*{ext}'))
            
            image_paths = [str(f) for f in image_files]
            total = len(image_paths)
            
            # Extract features with progress
            all_features = []
            for idx, img_path in enumerate(image_paths):
                try:
                    # Check if already indexed
                    cached = database.get_image_by_path(img_path)
                    if cached:
                        cached_feature = database.get_cached_features(
                            cached['id'], 'clip'
                        )
                        if cached_feature is not None:
                            all_features.append(cached_feature)
                            self.progress.emit(int((idx + 1) / total * 100))
                            continue
                    
                    # Extract new features
                    features = feature_extractor.extract_features(
                        img_path, 
                        methods=['clip']
                    )
                    
                    # Store in database
                    metadata = self._get_metadata(img_path)
                    img_id = database.add_image(img_path, metadata)
                    database.cache_features(img_id, 'clip', features['clip'])
                    
                    all_features.append(features['clip'])
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                
                self.progress.emit(int((idx + 1) / total * 100))
            
            # Build index
            if all_features:
                index_manager = VectorIndexManager(
                    dimension=len(all_features[0])
                )
                index_manager.create_index()
                index_manager.train_and_add(
                    np.array(all_features),
                    image_paths[:len(all_features)]
                )
                index_manager.save()
            
            self.finished.emit()
            
        except Exception as e:
            print(f"Indexing error: {e}")
    
    def _get_metadata(self, image_path: str) -> Dict:
        """Extract image metadata"""
        path = Path(image_path)
        return {
            'file_name': path.name,
            'file_size': path.stat().st_size,
            'modified_date': datetime.fromtimestamp(path.stat().st_mtime)
        }


class SearchThread(QThread):
    """Background thread for similarity search"""
    progress = pyqtSignal(int)
    results = pyqtSignal(list)
    
    def __init__(self, query_path: str, index_dir: str, num_results: int):
        super().__init__()
        self.query_path = query_path
        self.index_dir = index_dir
        self.num_results = num_results
    
    def run(self):
        """Run search process"""
        try:
            from core.feature_extractors import MultiModalFeatureExtractor
            from core.vector_index import VectorIndexManager
            
            self.progress.emit(20)
            
            # Extract query features
            feature_extractor = MultiModalFeatureExtractor()
            img = cv2.imread(self.query_path)
            query_features = feature_extractor.extractors['clip'].extract(img)
            
            self.progress.emit(50)
            
            # Load index and search
            index_manager = VectorIndexManager(dimension=len(query_features))
            index_manager.load()
            
            self.progress.emit(70)
            
            results = index_manager.search(query_features, k=self.num_results)
            
            self.progress.emit(100)
            self.results.emit(results)
            
        except Exception as e:
            print(f"Search error: {e}")
            self.results.emit([])


class DuplicateDetectionThread(QThread):
    """Background thread for duplicate detection"""
    progress = pyqtSignal(int)
    results = pyqtSignal(dict)
    
    def __init__(self, directory: str, hash_threshold: int, 
                 include_subdirs: bool):
        super().__init__()
        self.directory = directory
        self.hash_threshold = hash_threshold
        self.include_subdirs = include_subdirs
    
    def run(self):
        """Run duplicate detection"""
        try:
            from core.fast_duplicate_detector import FastDuplicateDetector
            
            # Scan directory
            self.progress.emit(5)
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
            image_files = []
            
            if self.include_subdirs:
                for ext in image_extensions:
                    image_files.extend(Path(self.directory).rglob(f'*{ext}'))
            else:
                for ext in image_extensions:
                    image_files.extend(Path(self.directory).glob(f'*{ext}'))
            
            image_paths = [str(f) for f in image_files]
            self.progress.emit(10)
            
            # Use fast detector with progress callback
            detector = FastDuplicateDetector(hash_threshold=self.hash_threshold)
            
            # Create progress callback
            def progress_callback(progress):
                # Map 0-100 to 20-100 range
                mapped_progress = 20 + int(progress * 0.8)
                self.progress.emit(mapped_progress)
            
            # Run detection with progress updates
            duplicates = detector.find_duplicates(image_paths, progress_callback)
            
            self.results.emit(duplicates)
            
        except Exception as e:
            print(f"Duplicate detection error: {e}")
            self.results.emit({})