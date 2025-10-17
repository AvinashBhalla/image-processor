# gui/main_window.py

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QProgressBar,
                             QTabWidget, QListWidget, QGridLayout, QScrollArea,
                             QMessageBox, QLineEdit, QSpinBox, QCheckBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap, QImage
import cv2
from typing import List, Tuple, Dict
from pathlib import Path
from gui.workers import IndexingThread, SearchThread, DuplicateDetectionThread

class ImageProcessorGUI(QMainWindow):
    """
    Main GUI window for image processing application
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Image Processor")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize backend components
        self.feature_extractor = None
        self.similarity_engine = None
        self.duplicate_detector = None
        
        self.init_ui()
        self.init_backend()
    
    def init_ui(self):
        """Initialize user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        tabs = QTabWidget()
        tabs.addTab(self.create_similarity_search_tab(), "Similarity Search")
        tabs.addTab(self.create_duplicate_detection_tab(), "Duplicate Detection")
        tabs.addTab(self.create_settings_tab(), "Settings")
        
        layout.addWidget(tabs)
    
    def create_similarity_search_tab(self) -> QWidget:
        """Create similarity search interface"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Query image selection
        query_layout = QHBoxLayout()
        self.query_image_label = QLabel("No image selected")
        self.query_image_label.setFixedSize(300, 300)
        self.query_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.query_image_label.setStyleSheet("border: 2px dashed #ccc;")
        
        query_controls = QVBoxLayout()
        select_query_btn = QPushButton("Select Query Image")
        select_query_btn.clicked.connect(self.select_query_image)
        
        self.index_dir_label = QLabel("Index directory: Not set")
        select_index_btn = QPushButton("Select Image Directory to Search")
        select_index_btn.clicked.connect(self.select_index_directory)
        
        self.num_results_spin = QSpinBox()
        self.num_results_spin.setRange(1, 100)
        self.num_results_spin.setValue(10)
        self.num_results_spin.setPrefix("Results: ")
        
        search_btn = QPushButton("Search Similar Images")
        search_btn.clicked.connect(self.perform_similarity_search)
        search_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        
        query_controls.addWidget(select_query_btn)
        query_controls.addWidget(self.index_dir_label)
        query_controls.addWidget(select_index_btn)
        query_controls.addWidget(self.num_results_spin)
        query_controls.addWidget(search_btn)
        query_controls.addStretch()
        
        query_layout.addWidget(self.query_image_label)
        query_layout.addLayout(query_controls)
        
        layout.addLayout(query_layout)
        
        # Progress bar
        self.search_progress = QProgressBar()
        layout.addWidget(self.search_progress)
        
        # Results area
        results_label = QLabel("Search Results:")
        results_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(results_label)
        
        self.results_scroll = QScrollArea()
        self.results_widget = QWidget()
        self.results_layout = QGridLayout(self.results_widget)
        self.results_scroll.setWidget(self.results_widget)
        self.results_scroll.setWidgetResizable(True)
        
        layout.addWidget(self.results_scroll)
        
        return tab
    
    def create_duplicate_detection_tab(self) -> QWidget:
        """Create duplicate detection interface"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.scan_dir_label = QLabel("Scan directory: Not set")
        select_scan_btn = QPushButton("Select Directory to Scan")
        select_scan_btn.clicked.connect(self.select_scan_directory)
        
        scan_btn = QPushButton("Start Duplicate Scan")
        scan_btn.clicked.connect(self.perform_duplicate_scan)
        scan_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 10px;")
        
        controls_layout.addWidget(self.scan_dir_label)
        controls_layout.addWidget(select_scan_btn)
        controls_layout.addWidget(scan_btn)
        
        layout.addLayout(controls_layout)
        
        # Options
        options_layout = QHBoxLayout()
        self.hash_threshold_spin = QSpinBox()
        self.hash_threshold_spin.setRange(0, 20)
        self.hash_threshold_spin.setValue(5)
        self.hash_threshold_spin.setPrefix("Hash threshold: ")
        
        self.include_subdirs_check = QCheckBox("Include subdirectories")
        self.include_subdirs_check.setChecked(True)
        
        options_layout.addWidget(self.hash_threshold_spin)
        options_layout.addWidget(self.include_subdirs_check)
        options_layout.addStretch()
        
        layout.addLayout(options_layout)
        
        # Progress
        self.duplicate_progress = QProgressBar()
        layout.addWidget(self.duplicate_progress)
        
        # Results
        results_label = QLabel("Duplicate Groups:")
        results_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(results_label)
        
        self.duplicate_results_list = QListWidget()
        layout.addWidget(self.duplicate_results_list)
        
        # Action buttons
        action_layout = QHBoxLayout()
        generate_report_btn = QPushButton("Generate HTML Report")
        generate_report_btn.clicked.connect(self.generate_duplicate_report)
        
        delete_selected_btn = QPushButton("Delete Selected Duplicates")
        delete_selected_btn.clicked.connect(self.delete_selected_duplicates)
        delete_selected_btn.setStyleSheet("background-color: #f44336; color: white;")
        
        action_layout.addWidget(generate_report_btn)
        action_layout.addWidget(delete_selected_btn)
        action_layout.addStretch()
        
        layout.addLayout(action_layout)
        
        return tab
    
    def create_settings_tab(self) -> QWidget:
        """Create settings interface"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Model selection
        model_label = QLabel("Feature Extraction Model:")
        model_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(model_label)
        
        # GPU settings
        self.use_gpu_check = QCheckBox("Use GPU acceleration (if available)")
        import torch
        self.use_gpu_check.setChecked(torch.cuda.is_available())
        self.use_gpu_check.setEnabled(torch.cuda.is_available())
        layout.addWidget(self.use_gpu_check)
        
        # Cache settings
        cache_label = QLabel("Cache Settings:")
        cache_label.setStyleSheet("font-weight: bold; margin-top: 20px;")
        layout.addWidget(cache_label)
        
        clear_cache_btn = QPushButton("Clear Feature Cache")
        clear_cache_btn.clicked.connect(self.clear_cache)
        layout.addWidget(clear_cache_btn)
        
        rebuild_index_btn = QPushButton("Rebuild Search Index")
        rebuild_index_btn.clicked.connect(self.rebuild_index)
        layout.addWidget(rebuild_index_btn)
        
        layout.addStretch()
        
        return tab
    
    def init_backend(self):
        """Initialize backend processing components"""
        # This will be done lazily when needed
        pass
    
    def select_query_image(self):
        """Open file dialog to select query image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Query Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)"
        )
        
        if file_path:
            self.query_image_path = file_path
            self.display_image(file_path, self.query_image_label)
    
    def select_index_directory(self):
        """Select directory containing images to search"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Image Directory"
        )
        
        if dir_path:
            self.index_directory = dir_path
            self.index_dir_label.setText(f"Index directory: {dir_path}")
            
            # Start indexing in background
            self.start_indexing(dir_path)
    
    def select_scan_directory(self):
        """Select directory to scan for duplicates"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Scan"
        )
        
        if dir_path:
            self.scan_directory = dir_path
            self.scan_dir_label.setText(f"Scan directory: {dir_path}")
    
    def display_image(self, image_path: str, label: QLabel):
        """Display image in label widget"""
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)
    
    def start_indexing(self, directory: str):
        """Start indexing images in background thread"""
        self.indexing_thread = IndexingThread(directory)
        self.indexing_thread.progress.connect(self.search_progress.setValue)
        self.indexing_thread.finished.connect(self.indexing_finished)
        self.indexing_thread.start()
    
    def indexing_finished(self):
        """Called when indexing is complete"""
        QMessageBox.information(self, "Indexing Complete", 
                              "Image directory has been indexed successfully!")
    
    def perform_similarity_search(self):
        """Perform similarity search"""
        if not hasattr(self, 'query_image_path'):
            QMessageBox.warning(self, "No Query Image", 
                              "Please select a query image first!")
            return
        
        if not hasattr(self, 'index_directory'):
            QMessageBox.warning(self, "No Index Directory", 
                              "Please select an image directory to search!")
            return
        
        # Start search in background
        num_results = self.num_results_spin.value()
        self.search_thread = SearchThread(
            self.query_image_path,
            self.index_directory,
            num_results
        )
        self.search_thread.progress.connect(self.search_progress.setValue)
        self.search_thread.results.connect(self.display_search_results)
        self.search_thread.start()
    
    def display_search_results(self, results: List[Tuple[str, float]]):
        """Display search results in grid"""
        # Clear previous results
        for i in reversed(range(self.results_layout.count())): 
            self.results_layout.itemAt(i).widget().setParent(None)
        
        # Display new results
        cols = 4
        for idx, (image_path, score) in enumerate(results):
            row = idx // cols
            col = idx % cols
            
            # Create result widget
            result_widget = QWidget()
            result_layout = QVBoxLayout(result_widget)
            
            # Image
            image_label = QLabel()
            image_label.setFixedSize(200, 200)
            image_label.setScaledContents(False)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setStyleSheet("border: 1px solid #ddd;")
            self.display_image(image_path, image_label)
            
            # Info
            info_label = QLabel(f"Similarity: {score:.2%}\n{Path(image_path).name}")
            info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            info_label.setWordWrap(True)
            
            # Button to open image
            open_btn = QPushButton("Open")
            open_btn.clicked.connect(lambda checked, path=image_path: self.open_image_location(path))
            
            result_layout.addWidget(image_label)
            result_layout.addWidget(info_label)
            result_layout.addWidget(open_btn)
            
            self.results_layout.addWidget(result_widget, row, col)
    
    def perform_duplicate_scan(self):
        """Perform duplicate detection scan"""
        if not hasattr(self, 'scan_directory'):
            QMessageBox.warning(self, "No Directory", 
                              "Please select a directory to scan!")
            return
        
        # Start duplicate detection in background
        hash_threshold = self.hash_threshold_spin.value()
        include_subdirs = self.include_subdirs_check.isChecked()
        
        self.duplicate_thread = DuplicateDetectionThread(
            self.scan_directory,
            hash_threshold,
            include_subdirs
        )
        self.duplicate_thread.progress.connect(self.duplicate_progress.setValue)
        self.duplicate_thread.results.connect(self.display_duplicate_results)
        self.duplicate_thread.start()
    
    def display_duplicate_results(self, duplicates: Dict[str, List[str]]):
        """Display duplicate detection results"""
        self.duplicate_results_list.clear()
        self.duplicate_data = duplicates
        
        total_duplicates = sum(len(dups) for dups in duplicates.values())
        
        for idx, (representative, dups) in enumerate(duplicates.items()):
            item_text = f"Group {idx + 1}: {len(dups)} duplicates of {Path(representative).name}"
            self.duplicate_results_list.addItem(item_text)
        
        # Show summary
        QMessageBox.information(
            self,
            "Scan Complete",
            f"Found {len(duplicates)} duplicate groups with {total_duplicates} total duplicates!"
        )
    
    def generate_duplicate_report(self):
        """Generate HTML report for duplicates"""
        if not hasattr(self, 'duplicate_data'):
            QMessageBox.warning(self, "No Results", 
                              "Please run a duplicate scan first!")
            return
        
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Report",
            "duplicate_report.html",
            "HTML Files (*.html)"
        )
        
        if save_path:
            report_gen = DuplicateReportGenerator()
            report_gen.generate_report(self.duplicate_data, save_path)
            QMessageBox.information(self, "Success", 
                                  f"Report saved to {save_path}")
    
    def delete_selected_duplicates(self):
        """Delete selected duplicate files"""
        # Implementation for safe deletion with confirmation
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            "Are you sure you want to delete selected duplicates?\nThis cannot be undone!",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Implement deletion logic
            pass
    
    def open_image_location(self, image_path: str):
        """Open image location in file explorer"""
        import subprocess
        import platform
        
        if platform.system() == "Windows":
            subprocess.Popen(f'explorer /select,"{image_path}"')
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", "-R", image_path])
        else:  # Linux
            subprocess.Popen(["xdg-open", str(Path(image_path).parent)])
    
    def clear_cache(self):
        """Clear feature cache"""
        reply = QMessageBox.question(
            self,
            "Clear Cache",
            "This will delete all cached features. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Clear cache logic
            db = ImageDatabase()
            # Delete cache files
            Path("data/faiss.index").unlink(missing_ok=True)
            Path("data/images.db").unlink(missing_ok=True)
            QMessageBox.information(self, "Success", "Cache cleared!")
    
    def rebuild_index(self):
        """Rebuild search index"""
        if hasattr(self, 'index_directory'):
            self.start_indexing(self.index_directory)
        else:
            QMessageBox.warning(self, "No Directory", 
                              "Please select an index directory first!")