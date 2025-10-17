# core/fast_duplicate_detector.py

import imagehash
from PIL import Image
import cv2
from typing import List, Dict, Set
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import os

class FastDuplicateDetector:
    """
    Optimized duplicate detection using multiple strategies
    """
    
    def __init__(self, hash_threshold: int = 5):
        self.hash_threshold = hash_threshold
        
        # Set PIL limits to prevent memory issues
        Image.MAX_IMAGE_PIXELS = 50_000_000  # 50MP limit
        
    def find_duplicates(self, image_paths: List[str], progress_callback=None) -> Dict[str, List[str]]:
        """
        Fast duplicate detection using optimized hashing
        """
        print(f"Processing {len(image_paths)} images...")
        
        if progress_callback:
            progress_callback(0)
        
        # Step 1: Quick filtering by file size and basic hash
        size_groups = self._group_by_size(image_paths)
        print(f"Grouped into {len(size_groups)} size groups")
        
        if progress_callback:
            progress_callback(30)
        
        # Step 2: Perceptual hashing within size groups
        hash_groups = []
        total_groups = len(size_groups)
        
        for i, size_group in enumerate(size_groups):
            if len(size_group) > 1:
                groups = self._hash_within_group(size_group)
                hash_groups.extend(groups)
            
            if progress_callback:
                progress = 30 + int((i + 1) / total_groups * 60)
                progress_callback(progress)
        
        print(f"Found {len(hash_groups)} potential duplicate groups")
        
        if progress_callback:
            progress_callback(90)
        
        # Step 3: Convert to final format
        result = {}
        for i, group in enumerate(hash_groups):
            if len(group) > 1:
                representative = min(group)  # Use lexicographically first as representative
                result[representative] = [p for p in group if p != representative]
        
        if progress_callback:
            progress_callback(100)
        
        return result
    
    def _group_by_size(self, image_paths: List[str]) -> List[List[str]]:
        """Group images by file size for quick filtering"""
        size_groups = defaultdict(list)
        
        for path in image_paths:
            try:
                size = os.path.getsize(path)
                # Group by size ranges (exact duplicates will have same size)
                size_key = (size // 1024) * 1024  # Round to nearest KB
                size_groups[size_key].append(path)
            except:
                continue
        
        # Only return groups with multiple images
        return [group for group in size_groups.values() if len(group) > 1]
    
    def _hash_within_group(self, image_paths: List[str]) -> List[List[str]]:
        """Apply perceptual hashing within a size group"""
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
            
            # Resize large images
            width, height = img.size
            if width * height > 1_000_000:  # 1MP limit
                # Calculate new size maintaining aspect ratio
                max_dim = 1000
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
