# core/memory_optimized_detector.py

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
import psutil

class MemoryOptimizedDetector:
    """
    Memory-optimized duplicate detection that uses more RAM for speed
    """
    
    def __init__(self, hash_threshold: int = 5, memory_limit_gb: int = 8):
        self.hash_threshold = hash_threshold
        self.memory_limit_gb = memory_limit_gb
        
        # Set PIL limits for high memory usage
        Image.MAX_IMAGE_PIXELS = 500_000_000  # 500MP limit for high-memory systems
        
        # Increase OpenCV thread count
        cv2.setNumThreads(mp.cpu_count())
        
        # Calculate optimal batch size based on available memory
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        self.batch_size = min(1000, int(available_memory * 100))  # More aggressive batching
        
        print(f"Memory Optimized Mode: {self.batch_size} batch size, {available_memory:.1f}GB available")
    
    def find_duplicates(self, image_paths: List[str], progress_callback=None) -> Dict[str, List[str]]:
        """
        Memory-optimized duplicate detection
        """
        print(f"Memory-optimized processing of {len(image_paths)} images...")
        
        if progress_callback:
            progress_callback(0)
        
        # Step 1: Batch file size grouping (5%)
        print("Step 1: Batch file analysis...")
        size_groups = self._batch_size_grouping(image_paths)
        print(f"Grouped into {len(size_groups)} size groups")
        
        if progress_callback:
            progress_callback(10)
        
        # Step 2: Batch hashing (10-80%)
        print("Step 2: Batch perceptual hashing...")
        hash_groups = self._batch_hashing(size_groups, progress_callback)
        
        if progress_callback:
            progress_callback(90)
        
        # Step 3: Finalize results (90-100%)
        print("Step 3: Finalizing results...")
        result = self._finalize_results(hash_groups)
        
        if progress_callback:
            progress_callback(100)
        
        print(f"Found {len(result)} duplicate groups")
        return result
    
    def _batch_size_grouping(self, image_paths: List[str]) -> List[List[str]]:
        """Batch file size analysis for better memory usage"""
        size_groups = defaultdict(list)
        
        # Process in batches to manage memory
        for i in range(0, len(image_paths), self.batch_size):
            batch = image_paths[i:i + self.batch_size]
            
            for path in batch:
                try:
                    size = os.path.getsize(path)
                    if size > 0:
                        size_key = (size // 1024) * 1024
                        size_groups[size_key].append(path)
                except:
                    continue
            
            # Force garbage collection after each batch
            gc.collect()
        
        return [group for group in size_groups.values() if len(group) > 1]
    
    def _batch_hashing(self, size_groups: List[List[str]], progress_callback=None) -> List[List[str]]:
        """Batch hashing with memory optimization"""
        all_hash_groups = []
        total_groups = len(size_groups)
        
        # Process groups in batches
        for i in range(0, total_groups, self.batch_size // 10):  # Smaller batches for hashing
            batch_groups = size_groups[i:i + self.batch_size // 10]
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=min(8, len(batch_groups))) as executor:
                future_to_group = {
                    executor.submit(self._hash_group_optimized, group): group 
                    for group in batch_groups
                }
                
                for future in future_to_group:
                    try:
                        hash_groups = future.result()
                        all_hash_groups.extend(hash_groups)
                    except Exception as e:
                        print(f"Error processing group: {e}")
                        continue
            
            # Update progress
            if progress_callback:
                progress = 10 + int((i + len(batch_groups)) / total_groups * 70)
                progress_callback(progress)
            
            # Force garbage collection
            gc.collect()
        
        return all_hash_groups
    
    def _hash_group_optimized(self, image_paths: List[str]) -> List[List[str]]:
        """Optimized hashing for a single group"""
        if len(image_paths) < 2:
            return []
        
        # Pre-load all images in the group for faster comparison
        image_data = {}
        valid_paths = []
        
        for img_path in image_paths:
            try:
                img = self._load_and_resize_fast(img_path)
                if img is not None:
                    image_data[img_path] = img
                    valid_paths.append(img_path)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        if len(valid_paths) < 2:
            return []
        
        # Compute hashes for all images
        image_hashes = {}
        for img_path in valid_paths:
            try:
                phash = imagehash.phash(image_data[img_path], hash_size=8)
                image_hashes[img_path] = phash
            except Exception as e:
                print(f"Error hashing {img_path}: {e}")
                continue
        
        # Find similar images
        groups = []
        processed = set()
        
        for img_path in valid_paths:
            if img_path in processed:
                continue
            
            group = [img_path]
            processed.add(img_path)
            current_hash = image_hashes[img_path]
            
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
    
    def _load_and_resize_fast(self, image_path: str) -> Image.Image:
        """Fast image loading with aggressive resizing"""
        try:
            img = Image.open(image_path)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # More aggressive resizing for speed
            width, height = img.size
            if width * height > 200_000:  # 0.2MP limit for maximum speed
                max_dim = 500  # Very small for speed
                if width > height:
                    new_width = max_dim
                    new_height = int(height * max_dim / width)
                else:
                    new_height = max_dim
                    new_width = int(width * max_dim / height)
                
                img = img.resize((new_width, new_height), Image.Resampling.NEAREST)  # Fastest resampling
            
            return img
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def _finalize_results(self, hash_groups: List[List[str]]) -> Dict[str, List[str]]:
        """Convert hash groups to final result format"""
        result = {}
        for group in hash_groups:
            if len(group) > 1:
                representative = min(group)
                result[representative] = [p for p in group if p != representative]
        
        return result
