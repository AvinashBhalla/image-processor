# cli.py

import argparse
import sys
from pathlib import Path
from core.feature_extractors import MultiModalFeatureExtractor
from core.vector_index import VectorIndexManager
from core.duplicate_detection import MultiStageDuplicateDetector
from utils.report_generator import DuplicateReportGenerator
import json

def similarity_search_command(args):
    """Execute similarity search from command line"""
    print(f"Searching for images similar to: {args.query}")
    
    # Initialize
    feature_extractor = MultiModalFeatureExtractor()
    
    # Extract query features
    import cv2
    query_img = cv2.imread(args.query)
    query_features = feature_extractor.extractors['clip'].extract(query_img)
    
    # Load index
    index_manager = VectorIndexManager(dimension=len(query_features))
    if not index_manager.load():
        print("Error: No index found. Please index images first.")
        return
    
    # Search
    results = index_manager.search(query_features, k=args.top_k)
    
    # Output results
    print(f"\nTop {len(results)} similar images:")
    for i, (path, score) in enumerate(results, 1):
        print(f"{i}. {path} (similarity: {score:.4f})")
    
    # Save results to JSON if requested
    if args.output:
        output_data = [
            {"path": path, "similarity": float(score)}
            for path, score in results
        ]
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

def index_command(args):
    """Index images from directory"""
    print(f"Indexing images from: {args.directory}")
    
    from core.batch_processor import IncrementalIndexer
    from core.database import ImageDatabase
    
    database = ImageDatabase()
    feature_extractor = MultiModalFeatureExtractor()
    index_manager = VectorIndexManager(dimension=512)  # CLIP dimension
    
    indexer = IncrementalIndexer(database, index_manager)
    new_images = indexer.find_new_images(args.directory)
    
    if not new_images:
        print("No new images to index.")
        return
    
    print(f"Found {len(new_images)} new images. Starting indexing...")
    indexer.incremental_update(new_images, feature_extractor.extractors['clip'])
    print("Indexing complete!")

def duplicate_command(args):
    """Detect duplicate images"""
    print(f"Scanning for duplicates in: {args.directory}")
    
    detector = MultiStageDuplicateDetector(
        hash_threshold=args.hash_threshold
    )
    
    # Scan directory
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
        image_files.extend(Path(args.directory).rglob(f'*{ext}'))
    
    image_paths = [str(f) for f in image_files]
    print(f"Found {len(image_paths)} images")
    
    # Detect duplicates
    hash_groups = detector.stage1_perceptual_hashing(image_paths)
    embedding_groups = detector.stage2_embedding_similarity(hash_groups)
    duplicates = detector.stage3_ssim_verification(embedding_groups)
    
    # Report results
    total_dups = sum(len(dups) for dups in duplicates.values())
    print(f"\nFound {len(duplicates)} duplicate groups with {total_dups} total duplicates")
    
    # Generate report
    if args.report:
        report_gen = DuplicateReportGenerator()
        report_gen.generate_report(duplicates, args.report)
        print(f"Report saved to: {args.report}")
    else:
        # Print to console
        for i, (rep, dups) in enumerate(duplicates.items(), 1):
            print(f"\nGroup {i}:")
            print(f"  Representative: {rep}")
            print(f"  Duplicates ({len(dups)}):")
            for dup in dups:
                print(f"    - {dup}")

def main_cli():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Advanced Image Processor - Command Line Interface"
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Similarity search command
    search_parser = subparsers.add_parser('search', help='Search for similar images')
    search_parser.add_argument('query', help='Path to query image')
    search_parser.add_argument('-k', '--top-k', type=int, default=10,
                              help='Number of results to return')
    search_parser.add_argument('-o', '--output', help='Output JSON file for results')
    search_parser.set_defaults(func=similarity_search_command)
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index images from directory')
    index_parser.add_argument('directory', help='Directory containing images')
    index_parser.set_defaults(func=index_command)
    
    # Duplicate detection command
    duplicate_parser = subparsers.add_parser('duplicates', 
                                            help='Detect duplicate images')
    duplicate_parser.add_argument('directory', help='Directory to scan')
    duplicate_parser.add_argument('-t', '--hash-threshold', type=int, default=5,
                                 help='Hash similarity threshold')
    duplicate_parser.add_argument('-r', '--report', help='Output HTML report path')
    duplicate_parser.set_defaults(func=duplicate_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute command
    args.func(args)

if __name__ == "__main__":
    main_cli()