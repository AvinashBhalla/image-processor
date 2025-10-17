# security/input_validation.py

from pathlib import Path
from typing import List
import magic
import os

class SecurityValidator:
    """
    Validate inputs for security
    """
    
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    
    @staticmethod
    def validate_image_path(path: str) -> bool:
        """
        Validate image path for security
        """
        try:
            # Check path traversal
            path_obj = Path(path).resolve()
            
            # Ensure file exists and is a file (not directory)
            if not path_obj.is_file():
                return False
            
            # Check extension
            if path_obj.suffix.lower() not in SecurityValidator.ALLOWED_EXTENSIONS:
                return False
            
            # Check file size
            if path_obj.stat().st_size > SecurityValidator.MAX_FILE_SIZE:
                return False
            
            # Verify actual file type (not just extension)
            mime = magic.from_file(str(path_obj), mime=True)
            if not mime.startswith('image/'):
                return False
            
            return True
            
        except Exception as e:
            print(f"Validation error: {e}")
            return False
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent injection attacks
        """
        # Remove path separators
        filename = filename.replace('/', '_').replace('\\', '_')
        
        # Remove special characters
        import re
        filename = re.sub(r'[^\w\s.-]', '', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:250] + ext
        
        return filename
    
    @staticmethod
    def validate_directory(directory: str, allow_system_dirs: bool = False) -> bool:
        """
        Validate directory path
        """
        try:
            dir_path = Path(directory).resolve()
            
            # Check if directory exists
            if not dir_path.is_dir():
                return False
            
            # Prevent access to system directories
            if not allow_system_dirs:
                system_dirs = {
                    Path('/etc'), Path('/sys'), Path('/proc'),
                    Path('C:\\Windows'), Path('C:\\Program Files')
                }
                
                for sys_dir in system_dirs:
                    if sys_dir.exists() and dir_path.is_relative_to(sys_dir):
                        return False
            
            # Check permissions
            if not os.access(dir_path, os.R_OK):
                return False
            
            return True
            
        except Exception as e:
            print(f"Directory validation error: {e}")
            return False