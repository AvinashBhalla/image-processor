# scripts/maintenance.py

import argparse
from pathlib import Path
import shutil
from datetime import datetime, timedelta

def clean_cache(days: int = 30):
    """Clean old cache files"""
    cache_dir = Path("data/cache")
    cutoff = datetime.now() - timedelta(days=days)
    
    removed = 0
    for file in cache_dir.rglob("*"):
        if file.is_file():
            if datetime.fromtimestamp(file.stat().st_mtime) < cutoff:
                file.unlink()
                removed += 1
    
    print(f"Removed {removed} old cache files")

def compact_database():
    """Compact SQLite database"""
    import sqlite3
    
    db_path = "data/images.db"
    backup_path = f"{db_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Backup
    shutil.copy(db_path, backup_path)
    print(f"Backup created: {backup_path}")
    
    # Compact
    conn = sqlite3.connect(db_path)
    conn.execute("VACUUM")
    conn.close()
    
    print("Database compacted")

def verify_index():
    """Verify index integrity"""
    from core.vector_index import VectorIndexManager
    
    index = VectorIndexManager(dimension=512)
    if index.load():
        print(f"Index loaded successfully: {len(index.id_to_path)} images")
    else:
        print("Index not found or corrupted")

def generate_report():
    """Generate system health report"""
    from core.database import ImageDatabase
    import psutil
    
    db = ImageDatabase()
    images = db.get_all_indexed_images()
    
    # System stats
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('.')
    
    report = f"""
    ╔════════════════════════════════════════╗
    ║   Image Processor - System Report     ║
    ╚════════════════════════════════════════╝
    
    📊 Database Statistics:
       • Total indexed images: {len(images)}
       • Database size: {Path('data/images.db').stat().st_size / (1024**2):.2f} MB
    
    💾 System Resources:
       • Memory usage: {memory.percent}%
       • Available memory: {memory.available / (1024**3):.2f} GB
       • Disk usage: {disk.percent}%
       • Available disk: {disk.free / (1024**3):.2f} GB
    
    📁 Cache Status:
       • Cache size: {sum(f.stat().st_size for f in Path('data/cache').rglob('*') if f.is_file()) / (1024**2):.2f} MB
    
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    print(report)
    
    # Save to file
    with open(f"reports/health_report_{datetime.now().strftime('%Y%m%d')}.txt", 'w') as f:
        f.write(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maintenance utilities")
    parser.add_argument('action', choices=['clean', 'compact', 'verify', 'report'])
    parser.add_argument('--days', type=int, default=30, help='Days for cache cleanup')
    
    args = parser.parse_args()
    
    if args.action == 'clean':
        clean_cache(args.days)
    elif args.action == 'compact':
        compact_database()
    elif args.action == 'verify':
        verify_index()
    elif args.action == 'report':
        generate_report()