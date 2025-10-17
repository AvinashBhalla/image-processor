# core/database.py

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict
import json
from datetime import datetime

class ImageDatabase:
    """
    SQLite database for image metadata and caching
    """
    
    def __init__(self, db_path: str = "data/images.db"):
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database schema"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        cursor = self.conn.cursor()
        
        # Main images table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_name TEXT NOT NULL,
                file_size INTEGER,
                width INTEGER,
                height INTEGER,
                format TEXT,
                created_date TIMESTAMP,
                modified_date TIMESTAMP,
                indexed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                perceptual_hash TEXT,
                is_blurry BOOLEAN,
                blur_score FLOAT
            )
        """)
        
        # Features cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                feature_type TEXT NOT NULL,
                feature_vector BLOB NOT NULL,
                extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images(id),
                UNIQUE(image_id, feature_type)
            )
        """)
        
        # Duplicate groups table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS duplicate_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                representative_image_id INTEGER NOT NULL,
                duplicate_image_id INTEGER NOT NULL,
                similarity_score FLOAT NOT NULL,
                detection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (representative_image_id) REFERENCES images(id),
                FOREIGN KEY (duplicate_image_id) REFERENCES images(id)
            )
        """)
        
        # Indexing for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_path ON images(file_path)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_perceptual_hash ON images(perceptual_hash)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_image ON features(image_id)
        """)
        
        self.conn.commit()
    
    def add_image(self, image_path: str, metadata: Dict) -> int:
        """Add image to database"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO images 
            (file_path, file_name, file_size, width, height, format, 
             created_date, modified_date, perceptual_hash, is_blurry, blur_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            image_path,
            metadata.get('file_name'),
            metadata.get('file_size'),
            metadata.get('width'),
            metadata.get('height'),
            metadata.get('format'),
            metadata.get('created_date'),
            metadata.get('modified_date'),
            metadata.get('perceptual_hash'),
            metadata.get('is_blurry'),
            metadata.get('blur_score')
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def cache_features(self, image_id: int, feature_type: str, 
                      feature_vector: np.ndarray):
        """Cache extracted features"""
        cursor = self.conn.cursor()
        
        # Serialize numpy array
        feature_blob = feature_vector.tobytes()
        
        cursor.execute("""
            INSERT OR REPLACE INTO features 
            (image_id, feature_type, feature_vector)
            VALUES (?, ?, ?)
        """, (image_id, feature_type, feature_blob))
        
        self.conn.commit()
    
    def get_cached_features(self, image_id: int, 
                          feature_type: str) -> Optional[np.ndarray]:
        """Retrieve cached features"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT feature_vector FROM features
            WHERE image_id = ? AND feature_type = ?
        """, (image_id, feature_type))
        
        result = cursor.fetchone()
        
        if result:
            # Deserialize numpy array
            return np.frombuffer(result[0], dtype=np.float32)
        
        return None
    
    def get_image_by_path(self, file_path: str) -> Optional[Dict]:
        """Get image metadata by file path"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM images WHERE file_path = ?
        """, (file_path,))
        
        result = cursor.fetchone()
        
        if result:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, result))
        
        return None
    
    def get_all_indexed_images(self) -> List[Dict]:
        """Get all indexed images"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT * FROM images")
        
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        return [dict(zip(columns, row)) for row in results]
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()