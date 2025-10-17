# components/smart_duplicate_manager.py

from pathlib import Path
from typing import List, Dict
import shutil
from datetime import datetime

class SmartDuplicateManager:
    """
    Intelligent duplicate management with safety features
    """
    
    def __init__(self, trash_dir: str = "data/trash"):
        self.trash_dir = Path(trash_dir)
        self.trash_dir.mkdir(parents=True, exist_ok=True)
        self.operation_log = []
    
    def select_best_representative(self, 
                                   image_group: List[str]) -> str:
        """
        Select the best image from a duplicate group based on:
        - File size (larger usually means less compression)
        - Resolution
        - File format (prefer lossless)
        - Creation date (prefer original)
        """
        scores = {}
        
        for img_path in image_group:
            path = Path(img_path)
            score = 0
            
            # File size score (normalized)
            size = path.stat().st_size
            score += size / (1024 ** 2)  # MB
            
            # Resolution score
            img = cv2.imread(img_path)
            if img is not None:
                pixels = img.shape[0] * img.shape[1]
                score += pixels / 1000000  # Megapixels
            
            # Format score (prefer lossless)
            if path.suffix.lower() in ['.png', '.tiff', '.bmp']:
                score += 10
            elif path.suffix.lower() in ['.jpg', '.jpeg']:
                score += 5
            
            # Older files (likely originals) score higher
            age_days = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
            score += age_days / 365  # Years old
            
            scores[img_path] = score
        
        # Return image with highest score
        return max(scores, key=scores.get)
    
    def safe_delete(self, 
                   file_path: str, 
                   move_to_trash: bool = True) -> bool:
        """
        Safely delete file (move to trash by default)
        """
        try:
            source = Path(file_path)
            
            if not source.exists():
                print(f"File not found: {file_path}")
                return False
            
            if move_to_trash:
                # Move to trash with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                trash_path = self.trash_dir / f"{timestamp}_{source.name}"
                shutil.move(str(source), str(trash_path))
                
                # Log operation
                self.operation_log.append({
                    'operation': 'move_to_trash',
                    'source': str(source),
                    'destination': str(trash_path),
                    'timestamp': timestamp
                })
            else:
                # Permanent deletion
                source.unlink()
                
                self.operation_log.append({
                    'operation': 'delete',
                    'source': str(source),
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                })
            
            return True
            
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
            return False
    
    def auto_clean_duplicates(self, 
                             duplicates: Dict[str, List[str]],
                             dry_run: bool = True) -> Dict:
        """
        Automatically clean duplicates with intelligent selection
        """
        results = {
            'processed_groups': 0,
            'files_deleted': 0,
            'space_saved': 0,
            'errors': []
        }
        
        for representative, dups in duplicates.items():
            # Add representative to group for comparison
            full_group = [representative] + dups
            
            # Select best image
            best = self.select_best_representative(full_group)
            
            # Delete others
            to_delete = [img for img in full_group if img != best]
            
            for img_path in to_delete:
                if not dry_run:
                    size = Path(img_path).stat().st_size
                    if self.safe_delete(img_path):
                        results['files_deleted'] += 1
                        results['space_saved'] += size
                    else:
                        results['errors'].append(img_path)
                else:
                    # Dry run - just calculate
                    size = Path(img_path).stat().st_size
                    results['files_deleted'] += 1
                    results['space_saved'] += size
            
            results['processed_groups'] += 1
        
        return results
    
    def restore_from_trash(self, file_name: str, 
                          restore_path: str = None) -> bool:
        """
        Restore file from trash
        """
        # Find file in trash
        trash_files = list(self.trash_dir.glob(f"*_{file_name}"))
        
        if not trash_files:
            print(f"File not found in trash: {file_name}")
            return False
        
        # Use most recent if multiple matches
        trash_file = max(trash_files, key=lambda p: p.stat().st_mtime)
        
        if restore_path is None:
            # Restore to original location from log
            for log_entry in reversed(self.operation_log):
                if log_entry.get('destination') == str(trash_file):
                    restore_path = log_entry['source']
                    break
        
        if restore_path is None:
            print("Cannot determine original location")
            return False
        
        # Restore file
        try:
            shutil.move(str(trash_file), restore_path)
            print(f"Restored: {restore_path}")
            return True
        except Exception as e:
            print(f"Error restoring file: {e}")
            return False
    
    def empty_trash(self, older_than_days: int = 30):
        """
        Empty trash (permanently delete old files)
        """
        cutoff_date = datetime.now().timestamp() - (older_than_days * 86400)
        deleted_count = 0
        
        for file_path in self.trash_dir.iterdir():
            if file_path.stat().st_mtime < cutoff_date:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        
        print(f"Emptied trash: {deleted_count} files deleted")