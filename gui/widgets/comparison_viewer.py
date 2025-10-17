# gui/widgets/comparison_viewer.py

from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel,
                             QSlider, QPushButton, QScrollArea)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np

class ImageComparisonViewer(QWidget):
    """
    Side-by-side image comparison with difference visualization
    """
    
    def __init__(self):
        super().__init__()
        self.image1 = None
        self.image2 = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)
        
        # Image display area
        images_layout = QHBoxLayout()
        
        # Left image
        left_container = QVBoxLayout()
        self.left_label = QLabel("Image 1")
        self.left_image = QLabel()
        self.left_image.setFixedSize(400, 400)
        self.left_image.setScaledContents(True)
        self.left_info = QLabel("No image loaded")
        left_container.addWidget(self.left_label)
        left_container.addWidget(self.left_image)
        left_container.addWidget(self.left_info)
        
        # Right image
        right_container = QVBoxLayout()
        self.right_label = QLabel("Image 2")
        self.right_image = QLabel()
        self.right_image.setFixedSize(400, 400)
        self.right_image.setScaledContents(True)
        self.right_info = QLabel("No image loaded")
        right_container.addWidget(self.right_label)
        right_container.addWidget(self.right_image)
        right_container.addWidget(self.right_info)
        
        # Difference visualization
        diff_container = QVBoxLayout()
        self.diff_label = QLabel("Difference Heatmap")
        self.diff_image = QLabel()
        self.diff_image.setFixedSize(400, 400)
        self.diff_image.setScaledContents(True)
        diff_container.addWidget(self.diff_label)
        diff_container.addWidget(self.diff_image)
        
        images_layout.addLayout(left_container)
        images_layout.addLayout(right_container)
        images_layout.addLayout(diff_container)
        
        layout.addLayout(images_layout)
        
        # Blend slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Blend:"))
        self.blend_slider = QSlider(Qt.Orientation.Horizontal)
        self.blend_slider.setRange(0, 100)
        self.blend_slider.setValue(50)
        self.blend_slider.valueChanged.connect(self.update_blend)
        slider_layout.addWidget(self.blend_slider)
        
        layout.addLayout(slider_layout)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        self.ssim_btn = QPushButton("Calculate SSIM")
        self.ssim_btn.clicked.connect(self.calculate_ssim)
        
        self.histogram_btn = QPushButton("Compare Histograms")
        self.histogram_btn.clicked.connect(self.compare_histograms)
        
        buttons_layout.addWidget(self.ssim_btn)
        buttons_layout.addWidget(self.histogram_btn)
        buttons_layout.addStretch()
        
        layout.addLayout(buttons_layout)
        
        # Results display
        self.results_label = QLabel("Comparison results will appear here")
        self.results_label.setWordWrap(True)
        layout.addWidget(self.results_label)
    
    def load_images(self, path1: str, path2: str):
        """Load two images for comparison"""
        self.image1 = cv2.imread(path1)
        self.image2 = cv2.imread(path2)
        
        # Display images
        self.display_image(self.image1, self.left_image)
        self.display_image(self.image2, self.right_image)
        
        # Update info
        self.left_info.setText(
            f"Size: {self.image1.shape[1]}x{self.image1.shape[0]}\n"
            f"File: {Path(path1).name}"
        )
        self.right_info.setText(
            f"Size: {self.image2.shape[1]}x{self.image2.shape[0]}\n"
            f"File: {Path(path2).name}"
        )
        
        # Generate difference heatmap
        self.generate_difference_map()
    
    def display_image(self, cv_image: np.ndarray, label: QLabel):
        """Display OpenCV image in QLabel"""
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, 
                         QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qt_image))
    
    def generate_difference_map(self):
        """Generate and display difference heatmap"""
        if self.image1 is None or self.image2 is None:
            return
        
        # Resize to same size
        h = min(self.image1.shape[0], self.image2.shape[0])
        w = min(self.image1.shape[1], self.image2.shape[1])
        
        img1_resized = cv2.resize(self.image1, (w, h))
        img2_resized = cv2.resize(self.image2, (w, h))
        
        # Calculate absolute difference
        diff = cv2.absdiff(img1_resized, img2_resized)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Apply colormap for visualization
        heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
        
        self.display_image(heatmap, self.diff_image)
    
    def update_blend(self):
        """Update blended view based on slider"""
        if self.image1 is None or self.image2 is None:
            return
        
        alpha = self.blend_slider.value() / 100.0
        
        # Resize to same size
        h = min(self.image1.shape[0], self.image2.shape[0])
        w = min(self.image1.shape[1], self.image2.shape[1])
        
        img1_resized = cv2.resize(self.image1, (w, h))
        img2_resized = cv2.resize(self.image2, (w, h))
        
        # Blend images
        blended = cv2.addWeighted(img1_resized, alpha, img2_resized, 1 - alpha, 0)
        
        self.display_image(blended, self.diff_image)
    
    def calculate_ssim(self):
        """Calculate and display SSIM score"""
        if self.image1 is None or self.image2 is None:
            return
        
        from skimage.metrics import structural_similarity as ssim
        
        # Convert to grayscale and resize
        gray1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
        
        h = min(gray1.shape[0], gray2.shape[0])
        w = min(gray1.shape[1], gray2.shape[1])
        
        gray1_resized = cv2.resize(gray1, (w, h))
        gray2_resized = cv2.resize(gray2, (w, h))
        
        # Calculate SSIM
        ssim_score = ssim(gray1_resized, gray2_resized)
        
        self.results_label.setText(
            f"Structural Similarity Index (SSIM): {ssim_score:.4f}\n"
            f"Interpretation: {'Very similar' if ssim_score > 0.95 else 'Similar' if ssim_score > 0.8 else 'Different'}"
        )
    
    def compare_histograms(self):
        """Compare color histograms"""
        if self.image1 is None or self.image2 is None:
            return
        
        # Calculate histograms
        hist1 = cv2.calcHist([self.image1], [0, 1, 2], None, 
                            [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([self.image2], [0, 1, 2], None,
                            [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        # Normalize
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        # Compare using different methods
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        chi_square = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
        
        self.results_label.setText(
            f"Histogram Comparison:\n"
            f"Correlation: {correlation:.4f}\n"
            f"Chi-Square: {chi_square:.4f}\n"
            f"Intersection: {intersection:.4f}"
        )