# core/duplicate_detection.py

import imagehash
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from typing import List, Dict, Set
from collections import defaultdict
import numpy as np
from tqdm import tqdm

class MultiStageDuplicateDetector:
    """
    Three-stage duplicate detection pipeline
    """
    
    def __init__(self, 
                 hash_threshold: int = 5,
                 embedding_threshold: float = 0.95,
                 ssim_threshold: float = 0.90):
        self.hash_threshold = hash_threshold
        self.embedding_threshold = embedding_threshold
        self.ssim_threshold = ssim_threshold
        
        self.feature_extractor = None  # Will be initialized
        
    def stage1_perceptual_hashing(self, 
                                 image_paths: List[str]) -> Dict[str, List[str]]:
        """
        Stage 1: Fast filtering using perceptual hashing
        Groups images that are likely duplicates
        
        Time Complexity: O(n)
        """
        print("Stage 1: Perceptual hashing...")
        
        # Set PIL image size limit to prevent memory issues
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = 100_000_000  # 100MP limit
        
        hash_to_images = defaultdict(list)
        image_to_hash = {}
        
        for i, img_path in enumerate(tqdm(image_paths, desc="Computing hashes")):
            try:
                # Load image with size limit
                img = Image.open(img_path)
                
                # Resize large images to speed up processing
                if img.size[0] * img.size[1] > 2_000_000:  # 2MP limit
                    img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
                
                # Compute hashes
                phash = imagehash.phash(img, hash_size=8)  # Smaller hash for speed
                dhash = imagehash.dhash(img, hash_size=8)
                
                # Store both hashes
                image_to_hash[img_path] = (phash, dhash)
                
                # Group by exact phash match first
                hash_to_images[str(phash)].append(img_path)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Find groups with similar hashes using optimized approach
        candidate_groups = []
        processed = set()
        
        # First pass: exact hash matches
        for hash_str, paths in hash_to_images.items():
            if len(paths) > 1:
                candidate_groups.append(paths)
                processed.update(paths)
        
        # Second pass: similar hashes (much more efficient)
        hash_list = list(image_to_hash.items())
        for i, (img_path, (phash, dhash)) in enumerate(hash_list):
            if img_path in processed:
                continue
                
            group = [img_path]
            processed.add(img_path)
            
            # Only compare with unprocessed images
            for j in range(i + 1, len(hash_list)):
                other_path, (other_phash, other_dhash) = hash_list[j]
                if other_path in processed:
                    continue
                    
                # Calculate hash distance
                phash_dist = phash - other_phash
                dhash_dist = dhash - other_dhash
                
                if phash_dist <= self.hash_threshold or \
                   dhash_dist <= self.hash_threshold:
                    group.append(other_path)
                    processed.add(other_path)
            
            if len(group) > 1:
                candidate_groups.append(group)
        
        return candidate_groups
    
    def stage2_embedding_similarity(self, 
                                   candidate_groups: List[List[str]]) -> List[List[str]]:
        """
        Stage 2: Refine candidates using deep learning embeddings
        
        Time Complexity: O(k * m^2) where k is number of groups, m is avg group size
        """
        print("Stage 2: Deep embedding comparison...")
        
        from core.feature_extractors import CLIPFeatureExtractor
        
        if self.feature_extractor is None:
            self.feature_extractor = CLIPFeatureExtractor()
        
        refined_groups = []
        
        for group in tqdm(candidate_groups, desc="Embedding comparison"):
            if len(group) < 2:
                continue
            
            # Extract embeddings for all images in group
            embeddings = []
            valid_paths = []
            
            for img_path in group:
                try:
                    img = cv2.imread(img_path)
                    embedding = self.feature_extractor.extract(img)
                    embeddings.append(embedding)
                    valid_paths.append(img_path)
                except Exception as e:
                    print(f"Error extracting features from {img_path}: {e}")
                    continue
            
            if len(embeddings) < 2:
                continue
            
            # Compute pairwise cosine similarities
            embeddings = np.array(embeddings)
            similarities = self._cosine_similarity_matrix(embeddings)
            
            # Form subgroups based on embedding similarity
            subgroups = self._cluster_by_similarity(
                valid_paths, 
                similarities, 
                self.embedding_threshold
            )
            
            refined_groups.extend(subgroups)
        
        return refined_groups
    
    def stage3_ssim_verification(self, 
                                candidate_groups: List[List[str]]) -> Dict[str, List[str]]:
        """
        Stage 3: Final verification using SSIM (Structural Similarity)
        
        Time Complexity: O(k * m^2 * image_size) - most expensive but accurate
        """
        print("Stage 3: SSIM verification...")
        
        final_duplicates = {}
        
        for group in tqdm(candidate_groups, desc="SSIM verification"):
            if len(group) < 2:
                continue
            
            # Load all images
            images = []
            valid_paths = []
            
            for img_path in group:
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images.append(img)
                        valid_paths.append(img_path)
                except:
                    continue
            
            if len(images) < 2:
                continue
            
            # Compute SSIM for all pairs
            verified_duplicates = []
            representative = valid_paths[0]
            
            for i in range(1, len(images)):
                # Resize images to same size for SSIM comparison
                img1_resized = cv2.resize(images[0], (512, 512))
                img2_resized = cv2.resize(images[i], (512, 512))
                
                # Compute SSIM
                ssim_score = ssim(img1_resized, img2_resized)
                
                if ssim_score >= self.ssim_threshold:
                    verified_duplicates.append(valid_paths[i])
            
            if verified_duplicates:
                final_duplicates[representative] = verified_duplicates
        
        return final_duplicates
    
    def _cosine_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix"""
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        
        # Compute similarity matrix
        similarity_matrix = np.dot(normalized, normalized.T)
        
        return similarity_matrix
    
    def _cluster_by_similarity(self, 
                              paths: List[str], 
                              similarity_matrix: np.ndarray,
                              threshold: float) -> List[List[str]]:
        """Group images based on similarity threshold"""
        n = len(paths)
        visited = set()
        clusters = []
        
        for i in range(n):
            if i in visited:
                continue
            
            cluster = [paths[i]]
            visited.add(i)
            
            for j in range(i + 1, n):
                if j in visited:
                    continue
                
                if similarity_matrix[i, j] >= threshold:
                    cluster.append(paths[j])
                    visited.add(j)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def detect_variations(self, image1: np.ndarray, 
                         image2: np.ndarray) -> Dict[str, any]:
        """
        Detect what type of variation exists between two images
        """
        variations = {
            'size_diff': False,
            'crop_detected': False,
            'brightness_diff': 0.0,
            'compression_diff': False,
            'format_diff': False
        }
        
        # Size difference
        if image1.shape != image2.shape:
            variations['size_diff'] = True
            
            # Check if one is cropped version of other
            if self._is_cropped(image1, image2):
                variations['crop_detected'] = True
        
        # Brightness difference
        mean1 = np.mean(image1)
        mean2 = np.mean(image2)
        variations['brightness_diff'] = abs(mean1 - mean2)
        
        return variations
    
    def _is_cropped(self, img1: np.ndarray, img2: np.ndarray) -> bool:
        """
        Detect if one image is a cropped version of another
        """
        # Resize smaller image to match larger one's aspect ratio
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Use template matching to find if smaller image is in larger
        if h1 * w1 > h2 * w2:
            larger, smaller = img1, img2
        else:
            larger, smaller = img2, img1
        
        # Convert to grayscale if needed
        if len(larger.shape) == 3:
            larger_gray = cv2.cvtColor(larger, cv2.COLOR_BGR2GRAY)
            smaller_gray = cv2.cvtColor(smaller, cv2.COLOR_BGR2GRAY)
        else:
            larger_gray, smaller_gray = larger, smaller
        
        # Resize for faster matching
        scale = 0.5
        larger_resized = cv2.resize(larger_gray, None, fx=scale, fy=scale)
        smaller_resized = cv2.resize(smaller_gray, None, fx=scale, fy=scale)
        
        # Template matching
        if smaller_resized.shape[0] <= larger_resized.shape[0] and \
           smaller_resized.shape[1] <= larger_resized.shape[1]:
            result = cv2.matchTemplate(larger_resized, smaller_resized, 
                                      cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            return max_val > 0.85  # High correlation indicates crop
        
        return False


class DuplicateReportGenerator:
    """
    Generate reports for duplicate detection results
    """
    
    def __init__(self):
        self.report_data = {}
    
    def generate_report(self, 
                       duplicates: Dict[str, List[str]], 
                       output_path: str = "duplicate_report.html"):
        """
        Generate HTML report with duplicate groups
        """
        html_content = self._create_html_template()
        
        total_duplicates = sum(len(dups) for dups in duplicates.values())
        space_savings = self._calculate_space_savings(duplicates)
        
        # Add statistics
        stats_html = f"""
        <div class="statistics">
            <h2>Duplicate Detection Summary</h2>
            <p><strong>Total duplicate groups:</strong> {len(duplicates)}</p>
            <p><strong>Total duplicate files:</strong> {total_duplicates}</p>
            <p><strong>Potential space savings:</strong> {space_savings / (1024**2):.2f} MB</p>
        </div>
        """
        
        # Add duplicate groups
        groups_html = "<div class='duplicate-groups'>"
        
        for idx, (representative, duplicates_list) in enumerate(duplicates.items()):
            groups_html += self._create_group_html(idx, representative, duplicates_list)
        
        groups_html += "</div>"
        
        # Combine and save
        final_html = html_content.replace("{{STATS}}", stats_html)
        final_html = final_html.replace("{{GROUPS}}", groups_html)
        
        with open(output_path, 'w') as f:
            f.write(final_html)
        
        print(f"Report generated: {output_path}")
    
    def _calculate_space_savings(self, duplicates: Dict[str, List[str]]) -> int:
        """Calculate potential space savings from removing duplicates"""
        total_size = 0
        
        for representative, dups in duplicates.items():
            for dup_path in dups:
                try:
                    total_size += Path(dup_path).stat().st_size
                except:
                    continue
        
        return total_size
    
    def _create_group_html(self, idx: int, 
                          representative: str, 
                          duplicates: List[str]) -> str:
        """Create HTML for a duplicate group"""
        group_html = f"""
        <div class="duplicate-group">
            <h3>Group {idx + 1}</h3>
            <div class="representative">
                <h4>Keep (Representative)</h4>
                <img src="file://{representative}" />
                <p>{Path(representative).name}</p>
                <p class="file-info">Size: {Path(representative).stat().st_size / 1024:.2f} KB</p>
            </div>
            <div class="duplicates-list">
                <h4>Duplicates ({len(duplicates)}) - Consider Deleting</h4>
        """
        
        for dup_path in duplicates:
            try:
                size = Path(dup_path).stat().st_size / 1024
                group_html += f"""
                <div class="duplicate-item">
                    <img src="file://{dup_path}" />
                    <p>{Path(dup_path).name}</p>
                    <p class="file-info">Size: {size:.2f} KB</p>
                    <button onclick="deleteFile('{dup_path}')">Delete</button>
                </div>
                """
            except:
                continue
        
        group_html += """
            </div>
        </div>
        """
        
        return group_html
    
    def _create_html_template(self) -> str:
        """HTML template for report"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Duplicate Detection Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .statistics { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .duplicate-group { border: 1px solid #ccc; margin: 20px 0; padding: 15px; }
                .representative { background: #e8f5e9; padding: 10px; }
                .duplicates-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; margin-top: 10px; }
                .duplicate-item { border: 1px solid #ddd; padding: 10px; text-align: center; }
                img { max-width: 100%; height: auto; max-height: 200px; object-fit: contain; }
                button { background: #f44336; color: white; border: none; padding: 5px 10px; cursor: pointer; }
                button:hover { background: #d32f2f; }
                .file-info { font-size: 0.9em; color: #666; }
            </style>
        </head>
        <body>
            <h1>Image Duplicate Detection Report</h1>
            {{STATS}}
            {{GROUPS}}
        </body>
        </html>
        """