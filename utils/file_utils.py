"""
File operation utilities
"""

from pathlib import Path
from typing import List

def get_image_files(directory: str, recursive: bool = True) -> List[str]:
    """Get all image files in directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    path = Path(directory)
    image_files = []
    
    if recursive:
        for ext in image_extensions:
            image_files.extend(path.rglob(f'*{ext}'))
            image_files.extend(path.rglob(f'*{ext.upper()}'))
    else:
        for ext in image_extensions:
            image_files.extend(path.glob(f'*{ext}'))
            image_files.extend(path.glob(f'*{ext.upper()}'))
    
    return [str(f) for f in image_files]

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"