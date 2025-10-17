"""
Visual similarity search results viewer with context menus
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, 
                             QLabel, QPushButton, QGroupBox, QGridLayout,
                             QMessageBox, QCheckBox, QFrame, QMenu, QSlider,
                             QSpinBox, QComboBox, QProgressBar)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QPixmap, QImage, QAction, QFont
import cv2
from pathlib import Path
from typing import List, Tuple, Dict
import os
import subprocess
import platform

class SimilaritySearchViewer(QWidget):
    """
    Visual viewer for similarity search results with context menus
    """
    
    # Signals
    image_selected = pyqtSignal(str)  # Emitted when image is selected
    images_deleted = pyqtSignal(list)  # Emitted when images are deleted
    search_requested = pyqtSignal(str, int)  # Emitted when new search is requested
    
    def __init__(self):
        super().__init__()
        self.search_results = []
        self.current_query_path = ""
        self.selected_images = set()
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Header with search controls
        header_layout = QHBoxLayout()
        
        self.query_label = QLabel("Query Image: None selected")
        self.query_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        self.select_query_btn = QPushButton("Select Query Image")
        self.select_query_btn.clicked.connect(self.select_query_image)
        
        self.num_results_spin = QSpinBox()
        self.num_results_spin.setRange(5, 50)
        self.num_results_spin.setValue(10)
        self.num_results_spin.setPrefix("Results: ")
        
        self.similarity_threshold = QSlider(Qt.Orientation.Horizontal)
        self.similarity_threshold.setRange(0, 100)
        self.similarity_threshold.setValue(70)
        self.similarity_threshold.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.similarity_threshold.valueChanged.connect(self.on_threshold_changed)
        
        self.threshold_label = QLabel("Similarity: 70%")
        
        self.search_btn = QPushButton("Search Similar")
        self.search_btn.clicked.connect(self.perform_search)
        self.search_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        
        header_layout.addWidget(self.query_label)
        header_layout.addWidget(self.select_query_btn)
        header_layout.addStretch()
        header_layout.addWidget(self.num_results_spin)
        header_layout.addWidget(QLabel("Min Similarity:"))
        header_layout.addWidget(self.similarity_threshold)
        header_layout.addWidget(self.threshold_label)
        header_layout.addWidget(self.search_btn)
        
        layout.addLayout(header_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Results area
        results_label = QLabel("Similarity Search Results:")
        results_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
        layout.addWidget(results_label)
        
        # Scroll area for results
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(400)
        
        self.content_widget = QWidget()
        self.content_layout = QGridLayout(self.content_widget)
        
        self.scroll_area.setWidget(self.content_widget)
        layout.addWidget(self.scroll_area)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_images)
        
        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self.deselect_all_images)
        
        self.delete_selected_btn = QPushButton("Delete Selected")
        self.delete_selected_btn.clicked.connect(self.delete_selected_images)
        self.delete_selected_btn.setStyleSheet("background-color: #f44336; color: white;")
        
        self.export_results_btn = QPushButton("Export Results")
        self.export_results_btn.clicked.connect(self.export_results)
        
        action_layout.addWidget(self.select_all_btn)
        action_layout.addWidget(self.deselect_all_btn)
        action_layout.addWidget(self.delete_selected_btn)
        action_layout.addWidget(self.export_results_btn)
        action_layout.addStretch()
        
        layout.addLayout(action_layout)
        
        # Status bar
        self.status_label = QLabel("Ready - Select a query image to start")
        layout.addWidget(self.status_label)
    
    def select_query_image(self):
        """Select query image for similarity search"""
        from PyQt6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Query Image", "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp)"
        )
        
        if file_path:
            self.current_query_path = file_path
            self.query_label.setText(f"Query Image: {Path(file_path).name}")
            self.status_label.setText("Query image selected - Click 'Search Similar' to find matches")
    
    def on_threshold_changed(self, value):
        """Handle similarity threshold change"""
        self.threshold_label.setText(f"Similarity: {value}%")
    
    def perform_search(self):
        """Perform similarity search"""
        if not self.current_query_path:
            QMessageBox.warning(self, "No Query Image", "Please select a query image first")
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Emit search signal
        num_results = self.num_results_spin.value()
        self.search_requested.emit(self.current_query_path, num_results)
    
    def display_search_results(self, results: List[Tuple[str, float]]):
        """Display similarity search results"""
        self.search_results = results
        self.selected_images.clear()
        
        # Clear previous content
        for i in reversed(range(self.content_layout.count())):
            self.content_layout.itemAt(i).widget().setParent(None)
        
        # Filter results by similarity threshold
        threshold = self.similarity_threshold.value() / 100.0
        filtered_results = [(path, score) for path, score in results if score >= threshold]
        
        if not filtered_results:
            self.status_label.setText("No similar images found above the threshold")
            return
        
        # Display results in grid
        cols = 4  # 4 images per row
        for i, (image_path, similarity) in enumerate(filtered_results):
            row = i // cols
            col = i % cols
            
            # Create image widget
            image_widget = self.create_image_widget(image_path, similarity, i == 0)
            self.content_layout.addWidget(image_widget, row, col)
        
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Found {len(filtered_results)} similar images")
    
    def create_image_widget(self, image_path: str, similarity: float, is_query: bool = False) -> QWidget:
        """Create a widget for displaying an image with controls"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Shape.Box)
        widget.setLineWidth(2)
        
        if is_query:
            widget.setStyleSheet("QFrame { border: 3px solid blue; }")
        else:
            widget.setStyleSheet("QFrame { border: 1px solid gray; }")
        
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
        
        # File info and similarity score
        file_name = Path(image_path).name
        if len(file_name) > 20:
            file_name = file_name[:17] + "..."
        
        info_text = f"{file_name}\nSimilarity: {similarity:.1%}"
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(info_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        if is_query:
            query_label = QLabel("QUERY")
            query_label.setStyleSheet("color: blue; font-weight: bold;")
            controls_layout.addWidget(query_label)
        else:
            # Checkbox for selection
            select_checkbox = QCheckBox("Select")
            select_checkbox.setProperty("image_path", image_path)
            select_checkbox.toggled.connect(self.on_image_selected)
            controls_layout.addWidget(select_checkbox)
            
            # Context menu button
            menu_btn = QPushButton("⋮")
            menu_btn.setProperty("image_path", image_path)
            menu_btn.clicked.connect(self.show_context_menu)
            menu_btn.setMaximumWidth(30)
            controls_layout.addWidget(menu_btn)
        
        layout.addLayout(controls_layout)
        
        return widget
    
    def load_image_thumbnail(self, image_path: str) -> QPixmap:
        """Load and create thumbnail for image"""
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                return QPixmap()
            
            # Load image with OpenCV
            img = cv2.imread(image_path)
            if img is None or img.size == 0:
                print(f"Could not load image: {image_path}")
                return QPixmap()
            
            # Check if image has valid dimensions
            if len(img.shape) < 2 or img.shape[0] == 0 or img.shape[1] == 0:
                print(f"Invalid image dimensions: {image_path}")
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
            
            # Ensure valid dimensions
            new_width = max(1, new_width)
            new_height = max(1, new_height)
            
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
    
    def show_context_menu(self):
        """Show context menu for image actions"""
        button = self.sender()
        image_path = button.property("image_path")
        
        menu = QMenu(self)
        
        # View in Explorer
        view_action = QAction("View in Explorer", self)
        view_action.triggered.connect(lambda: self.view_in_explorer(image_path))
        menu.addAction(view_action)
        
        # Open with default app
        open_action = QAction("Open with Default App", self)
        open_action.triggered.connect(lambda: self.open_with_default(image_path))
        menu.addAction(open_action)
        
        menu.addSeparator()
        
        # Delete file
        delete_action = QAction("Delete File", self)
        delete_action.triggered.connect(lambda: self.delete_single_file(image_path))
        menu.addAction(delete_action)
        
        # Show menu at cursor position
        menu.exec(button.mapToGlobal(button.rect().bottomLeft()))
    
    def view_in_explorer(self, image_path: str):
        """Open file location in explorer"""
        try:
            if platform.system() == "Windows":
                subprocess.run(["explorer", "/select,", image_path])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", "-R", image_path])
            else:  # Linux
                subprocess.run(["xdg-open", os.path.dirname(image_path)])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open explorer: {e}")
    
    def open_with_default(self, image_path: str):
        """Open image with default application"""
        try:
            if platform.system() == "Windows":
                os.startfile(image_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", image_path])
            else:  # Linux
                subprocess.run(["xdg-open", image_path])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open image: {e}")
    
    def delete_single_file(self, image_path: str):
        """Delete a single file"""
        reply = QMessageBox.question(
            self, "Confirm Deletion",
            f"Are you sure you want to delete {Path(image_path).name}?\n\nThis action cannot be undone!",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    self.images_deleted.emit([image_path])
                    self.status_label.setText(f"Deleted {Path(image_path).name}")
                    
                    # Refresh results
                    self.refresh_results()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not delete file: {e}")
    
    def on_image_selected(self, checked: bool):
        """Handle image selection checkbox"""
        checkbox = self.sender()
        image_path = checkbox.property("image_path")
        
        if checked:
            self.selected_images.add(image_path)
        else:
            self.selected_images.discard(image_path)
        
        self.update_status()
    
    def select_all_images(self):
        """Select all images in results"""
        for i in range(self.content_layout.count()):
            widget = self.content_layout.itemAt(i).widget()
            if isinstance(widget, QFrame):
                checkbox = widget.findChild(QCheckBox)
                if checkbox:
                    checkbox.setChecked(True)
    
    def deselect_all_images(self):
        """Deselect all images"""
        for i in range(self.content_layout.count()):
            widget = self.content_layout.itemAt(i).widget()
            if isinstance(widget, QFrame):
                checkbox = widget.findChild(QCheckBox)
                if checkbox:
                    checkbox.setChecked(False)
    
    def delete_selected_images(self):
        """Delete selected images"""
        if not self.selected_images:
            QMessageBox.warning(self, "No Selection", "Please select images to delete")
            return
        
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
            self.refresh_results()
            self.status_label.setText(f"Deleted {len(deleted_files)} images")
    
    def export_results(self):
        """Export search results to a text file"""
        if not self.search_results:
            QMessageBox.warning(self, "No Results", "No search results to export")
            return
        
        from PyQt6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "similarity_results.txt",
            "Text Files (*.txt)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("Similarity Search Results\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Query Image: {self.current_query_path}\n")
                    f.write(f"Number of Results: {len(self.search_results)}\n\n")
                    
                    for i, (path, similarity) in enumerate(self.search_results, 1):
                        f.write(f"{i}. {Path(path).name} - Similarity: {similarity:.1%}\n")
                        f.write(f"   Path: {path}\n\n")
                
                QMessageBox.information(self, "Export Complete", f"Results exported to {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", f"Could not export results: {e}")
    
    def refresh_results(self):
        """Refresh the results display"""
        if self.search_results:
            self.display_search_results(self.search_results)
    
    def update_status(self):
        """Update status label"""
        selected_count = len(self.selected_images)
        if selected_count > 0:
            self.status_label.setText(f"{selected_count} images selected for deletion")
        else:
            self.status_label.setText("Ready")
