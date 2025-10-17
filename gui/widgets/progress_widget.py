"""
Custom progress widget with status
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QProgressBar, QLabel
from PyQt6.QtCore import Qt

class ProgressWidget(QWidget):
    """Progress bar with status text"""
    
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        
    def set_status(self, text: str):
        """Update status text"""
        self.status_label.setText(text)
        
    def set_progress(self, value: int):
        """Update progress value (0-100)"""
        self.progress_bar.setValue(value)