import sys
import logging
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from gui.main_window import ImageProcessorGUI
from config import SystemConfig

def setup_logging(config: SystemConfig):
    """Setup application logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "app.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def initialize_directories(config: SystemConfig):
    """Create necessary directories"""
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(config.database_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.index_path).parent.mkdir(parents=True, exist_ok=True)

def main():
    """Main application entry point"""
    # Load configuration
    config = SystemConfig.load()
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("Starting Image Processor Application")
    
    # Initialize directories
    initialize_directories(config)
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Advanced Image Processor")
    app.setOrganizationName("ImageProcessing")
    
    # Create and show main window
    window = ImageProcessorGUI()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()