"""
Image utility functions
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple

def get_image_info(image_path: str) -> dict:
    """Get image metadata"""
    path = Path(image_path)
    img = cv2.imread(image_path)
    
    if img is None:
        return None
    
    return {
        'path': str(path),
        'name': path.name,
        'size': path.stat().st_size,
        'width': img.shape[1],
        'height': img.shape[0],
        'channels': img.shape[2] if len(img.shape) == 3 else 1,
        'format': path.suffix[1:].upper()
    }

def resize_maintain_aspect(image: np.ndarray, 
                          target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image maintaining aspect ratio"""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)