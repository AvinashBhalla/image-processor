import torch
import torch.nn as nn
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np
from typing import List

class MultiModalFeatureExtractor:
    """
    Combines multiple feature extraction methods for robust matching
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.extractors = {}
        self._initialize_extractors()
        
    def _initialize_extractors(self):
        """Initialize multiple feature extractors"""
        
        # 1. CLIP - Best for semantic similarity and robustness to blur
        self.extractors['clip'] = CLIPFeatureExtractor(self.device)
        
        # 2. EfficientNet - Good balance of speed and accuracy
        self.extractors['efficientnet'] = EfficientNetExtractor(self.device)
        
        # 3. Perceptual Hash - Fast for exact/near-exact matches
        self.extractors['phash'] = PerceptualHashExtractor()
        
        # 4. SIFT/ORB - Robust to partial images
        self.extractors['keypoints'] = KeypointExtractor()
        
    def extract_features(self, image_path: str, 
                        methods: List[str] = ['clip']) -> dict:
        """
        Extract features using specified methods
        
        Args:
            image_path: Path to image
            methods: List of extraction methods to use
            
        Returns:
            Dictionary of feature vectors
        """
        features = {}
        image = self._load_and_preprocess(image_path)
        
        for method in methods:
            if method in self.extractors:
                features[method] = self.extractors[method].extract(image)
                
        return features
    
    def _load_and_preprocess(self, image_path: str) -> np.ndarray:
        """Load and preprocess image with enhancement for blurry images"""
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Enhancement for blurry images
        if self._detect_blur(img):
            img = self._enhance_blurry_image(img)
            
        return img
    
    def _detect_blur(self, image: np.ndarray, threshold: float = 100.0) -> bool:
        """Detect if image is blurry using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold
    
    def _enhance_blurry_image(self, image: np.ndarray) -> np.ndarray:
        """Apply enhancement techniques for blurry images"""
        # Unsharp masking
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        # Contrast enhancement
        lab = cv2.cvtColor(unsharp, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced


class CLIPFeatureExtractor:
    """
    CLIP-based feature extraction - excellent for blurry and partial images
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(device)
        self.model.eval()
        
    @torch.no_grad()
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract CLIP embeddings"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        image_features = self.model.get_image_features(**inputs)
        
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().flatten()


class EfficientNetExtractor:
    """
    EfficientNet-based extraction - good performance/speed tradeoff
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        import timm
        self.model = timm.create_model('efficientnet_b0', 
                                       pretrained=True, 
                                       num_classes=0)  # Remove classification head
        self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    @torch.no_grad()
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract EfficientNet features"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        features = self.model(image_tensor)
        return features.cpu().numpy().flatten()


class KeypointExtractor:
    """
    SIFT/ORB keypoint extraction - robust to partial images
    """
    
    def __init__(self):
        # Use SIFT if available, otherwise ORB
        try:
            self.detector = cv2.SIFT_create(nfeatures=500)
        except:
            self.detector = cv2.ORB_create(nfeatures=500)
            
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract and aggregate keypoint descriptors"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(128)  # Return zero vector if no keypoints
        
        # Aggregate descriptors using mean and std
        mean_desc = np.mean(descriptors, axis=0)
        std_desc = np.std(descriptors, axis=0)
        
        return np.concatenate([mean_desc, std_desc])


class PerceptualHashExtractor:
    """
    Fast perceptual hashing for quick filtering
    """
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract perceptual hash"""
        import imagehash
        from PIL import Image
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Combine multiple hashes for robustness
        phash = imagehash.phash(pil_image)
        dhash = imagehash.dhash(pil_image)
        whash = imagehash.whash(pil_image)
        
        # Convert to numpy array
        combined = np.concatenate([
            np.array(phash.hash.flatten(), dtype=float),
            np.array(dhash.hash.flatten(), dtype=float),
            np.array(whash.hash.flatten(), dtype=float)
        ])
        
        return combined