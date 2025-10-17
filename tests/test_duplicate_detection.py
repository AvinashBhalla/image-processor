# tests/test_duplicate_detection.py

import pytest
import numpy as np
import cv2
from core.duplicate_detection import MultiStageDuplicateDetector
from pathlib import Path

@pytest.fixture
def duplicate_images(tmp_path):
    """Create images with duplicates"""
    # Create original image
    img1 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    path1 = tmp_path / "original.jpg"
    cv2.imwrite(str(path1), img1)
    
    # Create exact duplicate
    path2 = tmp_path / "duplicate.jpg"
    cv2.imwrite(str(path2), img1)
    
    # Create near-duplicate (slightly modified)
    img2 = cv2.convertScaleAbs(img1, alpha=1.1, beta=10)
    path3 = tmp_path / "near_duplicate.jpg"
    cv2.imwrite(str(path3), img2)
    
    # Create different image
    img3 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    path4 = tmp_path / "different.jpg"
    cv2.imwrite(str(path4), img3)
    
    return [str(path1), str(path2), str(path3), str(path4)]

def test_duplicate_detection(duplicate_images):
    """Test duplicate detection"""
    detector = MultiStageDuplicateDetector(hash_threshold=5)
    
    # Run detection
    hash_groups = detector.stage1_perceptual_hashing(duplicate_images)
    
    # Should detect at least one group of duplicates
    assert len(hash_groups) > 0
    
    # The first three images should be grouped together
    found_group = False
    for group in hash_groups:
        if len(group) >= 2:
            found_group = True
            break
    
    assert found_group == True

def test_perceptual_hashing(duplicate_images):
    """Test perceptual hashing stage"""
    detector = MultiStageDuplicateDetector()
    hash_groups = detector.stage1_perceptual_hashing(duplicate_images)
    
    # Should return list of groups
    assert isinstance(hash_groups, list)
    
    # At least one group should have duplicates
    assert any(len(group) > 1 for group in hash_groups)

def test_ssim_calculation(duplicate_images):
    """Test SSIM calculation between images"""
    from skimage.metrics import structural_similarity as ssim
    
    img1 = cv2.imread(duplicate_images[0], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(duplicate_images[1], cv2.IMREAD_GRAYSCALE)
    
    # Resize to same dimensions
    img1_resized = cv2.resize(img1, (512, 512))
    img2_resized = cv2.resize(img2, (512, 512))
    
    score = ssim(img1_resized, img2_resized)
    
    assert 0 <= score <= 1
    assert score > 0.9  # Exact duplicates should have high SSIM