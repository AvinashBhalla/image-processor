# tests/test_similarity_search.py

import pytest
import numpy as np
import cv2
from core.feature_extractors import CLIPFeatureExtractor, MultiModalFeatureExtractor
from core.vector_index import VectorIndexManager
import tempfile
from pathlib import Path

@pytest.fixture
def sample_images(tmp_path):
    """Create sample test images"""
    images = []
    for i in range(5):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img_path = tmp_path / f"test_image_{i}.jpg"
        cv2.imwrite(str(img_path), img)
        images.append(str(img_path))
    return images

@pytest.fixture
def feature_extractor():
    """Initialize feature extractor"""
    return CLIPFeatureExtractor(device='cpu')

def test_feature_extraction(feature_extractor, sample_images):
    """Test that feature extraction works"""
    img = cv2.imread(sample_images[0])
    features = feature_extractor.extract(img)
    
    assert features is not None
    assert isinstance(features, np.ndarray)
    assert len(features) > 0

def test_feature_dimension(feature_extractor, sample_images):
    """Test that feature dimensions are consistent"""
    features_list = []
    
    for img_path in sample_images:
        img = cv2.imread(img_path)
        features = feature_extractor.extract(img)
        features_list.append(features)
    
    # All features should have same dimension
    dimensions = [len(f) for f in features_list]
    assert len(set(dimensions)) == 1

def test_vector_index_creation():
    """Test FAISS index creation"""
    index = VectorIndexManager(dimension=512)
    index.create_index()
    
    assert index.index is not None

def test_vector_index_search(sample_images, feature_extractor, tmp_path):
    """Test similarity search"""
    # Extract features
    features = []
    for img_path in sample_images:
        img = cv2.imread(img_path)
        feat = feature_extractor.extract(img)
        features.append(feat)
    
    # Create and populate index
    index = VectorIndexManager(dimension=len(features[0]), 
                               index_path=str(tmp_path / "test.index"))
    index.create_index()
    index.train_and_add(np.array(features), sample_images)
    
    # Search
    query_features = features[0]
    results = index.search(query_features, k=3)
    
    assert len(results) > 0
    assert results[0][0] == sample_images[0]  # Should find itself

def test_blur_detection():
    """Test blur detection functionality"""
    from core.feature_extractors import MultiModalFeatureExtractor
    
    extractor = MultiModalFeatureExtractor()
    
    # Create blurry image
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    blurry = cv2.GaussianBlur(img, (15, 15), 0)
    
    is_blurry = extractor._detect_blur(blurry)
    assert is_blurry == True