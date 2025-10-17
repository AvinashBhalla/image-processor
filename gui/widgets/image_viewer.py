"""
Simple image viewer widget
"""

from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

class ImageViewer(QLabel):
    """Custom image viewer with zoom and pan"""
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False)
        
    def load_image(self, image_path: str):
        """Load and display image"""
        pixmap = QPixmap(image_path)
        self.setPixmap(pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))