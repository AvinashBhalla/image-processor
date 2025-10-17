# core/high_performance_detector.py

import imagehash
from PIL import Image
import cv2
from typing import List, Dict, Set
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import os
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import torch
import gc

class HighPerformanceDuplicateDetector:
    """
    High-performance duplicate detection using all available system resources
    """
    
    def __init__(self, hash_threshold: int = 5, use_gpu: bool = True, max_workers: int = None):
        self.hash_threshold = hash_threshold
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.max_workers = max_workers or min(mp.cpu_count() * 2, 16)  # Use more workers
        
        # Set PIL limits for high memory usage
        Image.MAX_IMAGE_PIXELS = 200_000_000  # 200MP limit for high-memory systems
        
        # Increase OpenCV thread count
        cv2.setNumThreads(mp.cpu_count())
        
        print(f"High Performance Mode: {self.max_workers} workers, GPU: {self.use_gpu}")
    
    def find_duplicates(self, image_paths: List[str], progress_callback=None) -> Dict[str, List[str]]:
        """
        High-performance duplicate detection using parallel processing
        """
        print(f"High-performance processing of {len(image_paths)} images...")
        
        if progress_callback:
            progress_callback(0)
        
        # Step 1: Parallel file size grouping (5%)
        print("Step 1: Parallel file analysis...")
        size_groups = self._parallel_size_grouping(image_paths)
        print(f"Grouped into {len(size_groups)} size groups")
        
        if progress_callback:
            progress_callback(10)
        
        # Step 2: Parallel hashing within groups (10-80%)
        print("Step 2: Parallel perceptual hashing...")
        hash_groups = self._parallel_hashing(size_groups, progress_callback)
        
        if progress_callback:
            progress_callback(90)
        
        # Step 3: Convert to final format (90-100%)
        print("Step 3: Finalizing results...")
        result = self._finalize_results(hash_groups)
        
        if progress_callback:
            progress_callback(100)
        
        print(f"Found {len(result)} duplicate groups")
        return result
    
    def _parallel_size_grouping(self, image_paths: List[str]) -> List[List[str]]:
        """Parallel file size analysis"""
        size_groups = defaultdict(list)
        
        # Use ThreadPoolExecutor for I/O bound file operations
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all file size checks
            future_to_path = {
                executor.submit(self._get_file_size, path): path 
                for path in image_paths
            }
            
            # Collect results
            for future in future_to_path:
                path = future_to_path[future]
                try:
                    size = future.result()
                    if size > 0:  # Valid file
                        size_key = (size // 1024) * 1024  # Round to nearest KB
                        size_groups[size_key].append(path)
                except Exception as e:
                    print(f"Error analyzing {path}: {e}")
                    continue
        
        # Only return groups with multiple images
        return [group for group in size_groups.values() if len(group) > 1]
    
    def _get_file_size(self, path: str) -> int:
        """Get file size safely"""
        try:
            return os.path.getsize(path)
        except:
            return 0
    
    def _parallel_hashing(self, size_groups: List[List[str]], progress_callback=None) -> List[List[str]]:
        """Parallel perceptual hashing within size groups"""
        all_hash_groups = []
        total_groups = len(size_groups)
        
        # Process groups in parallel
        with ProcessPoolExecutor(max_workers=min(self.max_workers, total_groups)) as executor:
            # Submit all groups for processing
            future_to_group = {
                executor.submit(self._hash_group, group): group 
                for group in size_groups
            }
            
            # Collect results with progress updates
            for i, future in enumerate(future_to_group):
                try:
                    hash_groups = future.result()
                    all_hash_groups.extend(hash_groups)
                    
                    if progress_callback:
                        progress = 10 + int((i + 1) / total_groups * 70)
                        progress_callback(progress)
                        
                except Exception as e:
                    print(f"Error processing group: {e}")
                    continue
        
        return all_hash_groups
    
    def _hash_group(self, image_paths: List[str]) -> List[List[str]]:
        """Hash a single group of images (runs in separate process)"""
        if len(image_paths) < 2:
            return []
        
        # Compute hashes for all images in group
        image_hashes = {}
        valid_paths = []
        
        for img_path in image_paths:
            try:
                # Load and resize image for speed
                img = self._load_and_resize(img_path)
                if img is None:
                    continue
                
                # Compute perceptual hash
                phash = imagehash.phash(img, hash_size=8)
                image_hashes[img_path] = phash
                valid_paths.append(img_path)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if len(valid_paths) < 2:
            return []
        
        # Find similar images using hash distance
        groups = []
        processed = set()
        
        for img_path in valid_paths:
            if img_path in processed:
                continue
            
            group = [img_path]
            processed.add(img_path)
            current_hash = image_hashes[img_path]
            
            # Compare with remaining images
            for other_path in valid_paths:
                if other_path in processed:
                    continue
                
                other_hash = image_hashes[other_path]
                distance = current_hash - other_hash
                
                if distance <= self.hash_threshold:
                    group.append(other_path)
                    processed.add(other_path)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _load_and_resize(self, image_path: str) -> Image.Image:
        """Load and resize image for efficient processing"""
        try:
            img = Image.open(image_path)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize large images (more aggressive for speed)
            width, height = img.size
            if width * height > 500_000:  # 0.5MP limit for speed
                # Calculate new size maintaining aspect ratio
                max_dim = 800  # Smaller for speed
                if width > height:
                    new_width = max_dim
                    new_height = int(height * max_dim / width)
                else:
                    new_height = max_dim
                    new_width = int(width * max_dim / height)
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            return img
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def _finalize_results(self, hash_groups: List[List[str]]) -> Dict[str, List[str]]:
        """Convert hash groups to final result format"""
        result = {}
        for group in hash_groups:
            if len(group) > 1:
                representative = min(group)  # Use lexicographically first as representative
                result[representative] = [p for p in group if p != representative]
        
        return result
