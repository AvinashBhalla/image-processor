"""
Visual duplicate group viewer with image previews and management
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, 
                             QLabel, QPushButton, QGroupBox, QGridLayout,
                             QMessageBox, QCheckBox, QFrame, QSplitter)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import cv2
from pathlib import Path
from typing import List, Dict, Tuple
import os

class DuplicateGroupViewer(QWidget):
    """
    Visual viewer for duplicate image groups with management capabilities
    """
    
    # Signals
    image_selected = pyqtSignal(str)  # Emitted when best image is selected
    images_deleted = pyqtSignal(list)  # Emitted when images are deleted
    group_processed = pyqtSignal(int)  # Emitted when a group is processed
    
    def __init__(self):
        super().__init__()
        self.duplicate_groups = {}
        self.current_group_index = 0
        self.selected_images = set()  # Images selected for deletion
        self.best_images = {}  # Best image for each group
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Header with controls
        header_layout = QHBoxLayout()
        
        self.group_label = QLabel("Group 1 of 0")
        self.group_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        self.prev_button = QPushButton("← Previous")
        self.prev_button.clicked.connect(self.previous_group)
        
        self.next_button = QPushButton("Next →")
        self.next_button.clicked.connect(self.next_group)
        
        self.select_best_button = QPushButton("Select Best Image")
        self.select_best_button.clicked.connect(self.select_best_image)
        
        self.delete_selected_button = QPushButton("Delete Selected")
        self.delete_selected_button.clicked.connect(self.delete_selected_images)
        
        self.keep_best_only_button = QPushButton("Keep Best Only")
        self.keep_best_only_button.clicked.connect(self.keep_best_only)
        
        header_layout.addWidget(self.group_label)
        header_layout.addStretch()
        header_layout.addWidget(self.prev_button)
        header_layout.addWidget(self.next_button)
        header_layout.addStretch()
        header_layout.addWidget(self.select_best_button)
        header_layout.addWidget(self.delete_selected_button)
        header_layout.addWidget(self.keep_best_only_button)
        
        layout.addLayout(header_layout)
        
        # Main content area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(400)
        
        self.content_widget = QWidget()
        self.content_layout = QGridLayout(self.content_widget)
        
        self.scroll_area.setWidget(self.content_widget)
        layout.addWidget(self.scroll_area)
        
        # Status bar
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
    
    def load_duplicate_groups(self, duplicate_groups: Dict[str, List[str]]):
        """Load duplicate groups for display"""
        self.duplicate_groups = duplicate_groups
        self.current_group_index = 0
        self.selected_images.clear()
        self.best_images.clear()
        
        if not duplicate_groups:
            self.status_label.setText("No duplicate groups found")
            return
        
        self.update_group_display()
        self.status_label.setText(f"Loaded {len(duplicate_groups)} duplicate groups")
    
    def update_group_display(self):
        """Update the display for the current group"""
        if not self.duplicate_groups:
            return
        
        # Clear previous content
        for i in reversed(range(self.content_layout.count())):
            self.content_layout.itemAt(i).widget().setParent(None)
        
        # Get current group
        group_keys = list(self.duplicate_groups.keys())
        if self.current_group_index >= len(group_keys):
            return
        
        representative = group_keys[self.current_group_index]
        duplicates = self.duplicate_groups[representative]
        all_images = [representative] + duplicates
        
        # Update group label
        self.group_label.setText(f"Group {self.current_group_index + 1} of {len(group_keys)}")
        
        # Update navigation buttons
        self.prev_button.setEnabled(self.current_group_index > 0)
        self.next_button.setEnabled(self.current_group_index < len(group_keys) - 1)
        
        # Display images in grid
        cols = 4  # 4 images per row
        for i, image_path in enumerate(all_images):
            row = i // cols
            col = i % cols
            
            # Create image widget
            image_widget = self.create_image_widget(image_path, i == 0)  # First is representative
            self.content_layout.addWidget(image_widget, row, col)
    
    def create_image_widget(self, image_path: str, is_representative: bool = False) -> QWidget:
        """Create a widget for displaying an image with controls"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Shape.Box)
        widget.setLineWidth(2)
        
        if is_representative:
            widget.setStyleSheet("QFrame { border: 3px solid green; }")
        
        layout = QVBoxLayout(widget)
        
        # Image preview
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setMinimumSize(150, 150)
        image_label.setMaximumSize(200, 200)
        image_label.setScaledContents(True)
        
        # Load and display image
        pixmap = self.load_image_thumbnail(image_path)
        if pixmap:
            image_label.setPixmap(pixmap)
        else:
            image_label.setText("Failed to load")
        
        layout.addWidget(image_label)
        
        # File info
        file_name = Path(image_path).name
        if len(file_name) > 20:
            file_name = file_name[:17] + "..."
        
        info_label = QLabel(file_name)
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        if is_representative:
            rep_label = QLabel("REPRESENTATIVE")
            rep_label.setStyleSheet("color: green; font-weight: bold;")
            controls_layout.addWidget(rep_label)
        else:
            # Checkbox for selection
            select_checkbox = QCheckBox("Select")
            select_checkbox.setProperty("image_path", image_path)
            select_checkbox.toggled.connect(self.on_image_selected)
            controls_layout.addWidget(select_checkbox)
            
            # Best image button
            best_button = QPushButton("Best")
            best_button.setProperty("image_path", image_path)
            best_button.clicked.connect(self.mark_as_best)
            controls_layout.addWidget(best_button)
        
        layout.addLayout(controls_layout)
        
        return widget
    
    def load_image_thumbnail(self, image_path: str) -> QPixmap:
        """Load and create thumbnail for image"""
        try:
            # Load image with OpenCV
            img = cv2.imread(image_path)
            if img is None:
                return QPixmap()
            
            # Resize to thumbnail size
            height, width = img.shape[:2]
            max_size = 200
            
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            
            img_resized = cv2.resize(img, (new_width, new_height))
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage then QPixmap
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            return QPixmap.fromImage(qt_image)
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return QPixmap()
    
    def on_image_selected(self, checked: bool):
        """Handle image selection checkbox"""
        checkbox = self.sender()
        image_path = checkbox.property("image_path")
        
        if checked:
            self.selected_images.add(image_path)
        else:
            self.selected_images.discard(image_path)
        
        self.update_status()
    
    def mark_as_best(self):
        """Mark an image as the best in the group"""
        button = self.sender()
        image_path = button.property("image_path")
        
        # Get current group
        group_keys = list(self.duplicate_groups.keys())
        representative = group_keys[self.current_group_index]
        
        self.best_images[representative] = image_path
        self.status_label.setText(f"Marked {Path(image_path).name} as best image")
    
    def select_best_image(self):
        """Automatically select the best image based on quality metrics"""
        group_keys = list(self.duplicate_groups.keys())
        representative = group_keys[self.current_group_index]
        duplicates = self.duplicate_groups[representative]
        all_images = [representative] + duplicates
        
        best_image = self.find_best_image(all_images)
        if best_image:
            self.best_images[representative] = best_image
            self.status_label.setText(f"Auto-selected {Path(best_image).name} as best image")
    
    def find_best_image(self, image_paths: List[str]) -> str:
        """Find the best image based on quality metrics"""
        best_score = -1
        best_image = None
        
        for image_path in image_paths:
            try:
                # Load image
                img = cv2.imread(image_path)
                if img is None:
                    continue
                
                # Calculate quality score
                score = self.calculate_image_quality(img)
                
                if score > best_score:
                    best_score = score
                    best_image = image_path
                    
            except Exception as e:
                print(f"Error analyzing {image_path}: {e}")
                continue
        
        return best_image
    
    def calculate_image_quality(self, img) -> float:
        """Calculate image quality score"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (sharpness)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate image size (larger is generally better)
            size_score = img.shape[0] * img.shape[1]
            
            # Calculate contrast
            contrast = gray.std()
            
            # Combined score
            score = laplacian_var * 0.5 + (size_score / 1000000) * 0.3 + contrast * 0.2
            
            return score
            
        except Exception:
            return 0
    
    def delete_selected_images(self):
        """Delete selected images"""
        if not self.selected_images:
            QMessageBox.warning(self, "No Selection", "Please select images to delete")
            return
        
        # Confirm deletion
        reply = QMessageBox.question(
            self, "Confirm Deletion",
            f"Are you sure you want to delete {len(self.selected_images)} images?\n\nThis action cannot be undone!",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            deleted_files = []
            for image_path in self.selected_images:
                try:
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        deleted_files.append(image_path)
                except Exception as e:
                    print(f"Error deleting {image_path}: {e}")
            
            # Emit signal
            self.images_deleted.emit(deleted_files)
            
            # Update display
            self.selected_images.clear()
            self.update_group_display()
            self.status_label.setText(f"Deleted {len(deleted_files)} images")
    
    def keep_best_only(self):
        """Keep only the best image from the current group"""
        group_keys = list(self.duplicate_groups.keys())
        representative = group_keys[self.current_group_index]
        duplicates = self.duplicate_groups[representative]
        
        # Get best image
        best_image = self.best_images.get(representative)
        if not best_image:
            QMessageBox.warning(self, "No Best Image", "Please select a best image first")
            return
        
        # Confirm action
        reply = QMessageBox.question(
            self, "Keep Best Only",
            f"Keep only {Path(best_image).name} and delete all other images in this group?\n\nThis action cannot be undone!",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            all_images = [representative] + duplicates
            images_to_delete = [img for img in all_images if img != best_image]
            
            deleted_files = []
            for image_path in images_to_delete:
                try:
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        deleted_files.append(image_path)
                except Exception as e:
                    print(f"Error deleting {image_path}: {e}")
            
            # Emit signal
            self.images_deleted.emit(deleted_files)
            
            # Remove group from display
            del self.duplicate_groups[representative]
            if self.current_group_index >= len(self.duplicate_groups):
                self.current_group_index = max(0, len(self.duplicate_groups) - 1)
            
            # Update display
            self.update_group_display()
            self.status_label.setText(f"Kept best image, deleted {len(deleted_files)} others")
    
    def previous_group(self):
        """Navigate to previous group"""
        if self.current_group_index > 0:
            self.current_group_index -= 1
            self.update_group_display()
    
    def next_group(self):
        """Navigate to next group"""
        group_keys = list(self.duplicate_groups.keys())
        if self.current_group_index < len(group_keys) - 1:
            self.current_group_index += 1
            self.update_group_display()
    
    def update_status(self):
        """Update status label"""
        selected_count = len(self.selected_images)
        if selected_count > 0:
            self.status_label.setText(f"{selected_count} images selected for deletion")
        else:
            self.status_label.setText("Ready")
