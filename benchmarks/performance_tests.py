# benchmarks/performance_tests.py

import time
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
from core.feature_extractors import MultiModalFeatureExtractor
from core.vector_index import VectorIndexManager
from core.duplicate_detection import MultiStageDuplicateDetector

class PerformanceBenchmark:
    """
    Benchmark suite for performance testing
    """
    
    def __init__(self):
        self.results = {}
    
    def benchmark_feature_extraction(self, 
                                    image_paths: List[str],
                                    methods: List[str] = ['clip', 'efficientnet']) -> Dict:
        """Benchmark feature extraction speed"""
        extractor = MultiModalFeatureExtractor()
        results = {}
        
        for method in methods:
            times = []
            
            for img_path in image_paths:
                img = cv2.imread(img_path)
                
                start = time.time()
                features = extractor.extractors[method].extract(img)
                elapsed = time.time() - start
                
                times.append(elapsed)
            
            results[method] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'total_time': np.sum(times),
                'images_per_second': len(image_paths) / np.sum(times)
            }
        
        self.results['feature_extraction'] = results
        return results
    
    def benchmark_index_search(self, 
                              index: VectorIndexManager,
                              query_features: List[np.ndarray],
                              k_values: List[int] = [1, 5, 10, 50]) -> Dict:
        """Benchmark search speed for different k values"""
        results = {}
        
        for k in k_values:
            times = []
            
            for query in query_features:
                start = time.time()
                _ = index.search(query, k=k)
                elapsed = time.time() - start
                times.append(elapsed)
            
            results[f'k={k}'] = {
                'mean_time': np.mean(times),
                'queries_per_second': len(query_features) / np.sum(times)
            }
        
        self.results['index_search'] = results
        return results
    
    def benchmark_duplicate_detection(self, 
                                     image_paths: List[str]) -> Dict:
        """Benchmark duplicate detection pipeline"""
        detector = MultiStageDuplicateDetector()
        
        # Stage 1: Perceptual hashing
        start = time.time()
        hash_groups = detector.stage1_perceptual_hashing(image_paths)
        stage1_time = time.time() - start
        
        # Stage 2: Embedding similarity
        start = time.time()
        embedding_groups = detector.stage2_embedding_similarity(hash_groups)
        stage2_time = time.time() - start
        
        # Stage 3: SSIM verification
        start = time.time()
        final_groups = detector.stage3_ssim_verification(embedding_groups)
        stage3_time = time.time() - start
        
        results = {
            'stage1_time': stage1_time,
            'stage2_time': stage2_time,
            'stage3_time': stage3_time,
            'total_time': stage1_time + stage2_time + stage3_time,
            'images_per_second': len(image_paths) / (stage1_time + stage2_time + stage3_time)
        }
        
        self.results['duplicate_detection'] = results
        return results
    
    def benchmark_memory_usage(self, operation_func, *args) -> Dict:
        """Benchmark memory usage of operation"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure before
        mem_before = process.memory_info().rss / (1024 ** 2)  # MB
        
        # Run operation
        start = time.time()
        result = operation_func(*args)
        elapsed = time.time() - start
        
        # Measure after
        mem_after = process.memory_info().rss / (1024 ** 2)  # MB
        
        return {
            'memory_before_mb': mem_before,
            'memory_after_mb': mem_after,
            'memory_increase_mb': mem_after - mem_before,
            'execution_time': elapsed
        }
    
    def generate_report(self, output_path: str = "benchmark_report.html"):
        """Generate HTML report with benchmark results"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Benchmark Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                .benchmark-section { margin: 30px 0; padding: 20px; background: #f5f5f5; border-radius: 5px; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #4CAF50; color: white; }
                .metric { font-weight: bold; color: #2196F3; }
            </style>
        </head>
        <body>
            <h1>Performance Benchmark Report</h1>
            <p>Generated: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
        """
        
        # Add results for each benchmark
        for benchmark_name, results in self.results.items():
            html += f"""
            <div class="benchmark-section">
                <h2>{benchmark_name.replace('_', ' ').title()}</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
            """
            
            for key, value in results.items():
                if isinstance(value, dict):
                    html += f"<tr><td colspan='2'><strong>{key}</strong></td></tr>"
                    for sub_key, sub_value in value.items():
                        html += f"<tr><td>&nbsp;&nbsp;{sub_key}</td><td class='metric'>{sub_value:.4f}</td></tr>"
                else:
                    html += f"<tr><td>{key}</td><td class='metric'>{value:.4f}</td></tr>"
            
            html += "</table></div>"
        
        html += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        print(f"Benchmark report saved to: {output_path}")
    
    def plot_results(self, output_dir: str = "benchmark_plots"):
        """Generate visualization plots for benchmark results"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Plot feature extraction times
        if 'feature_extraction' in self.results:
            methods = list(self.results['feature_extraction'].keys())
            times = [self.results['feature_extraction'][m]['mean_time'] 
                    for m in methods]
            
            plt.figure(figsize=(10, 6))
            plt.bar(methods, times, color=['#4CAF50', '#2196F3', '#FF9800'])
            plt.xlabel('Method')
            plt.ylabel('Mean Time (seconds)')
            plt.title('Feature Extraction Performance')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(Path(output_dir) / 'feature_extraction.png', dpi=150)
            plt.close()
        
        # Plot search performance
        if 'index_search' in self.results:
            k_values = list(self.results['index_search'].keys())
            qps = [self.results['index_search'][k]['queries_per_second'] 
                   for k in k_values]
            
            plt.figure(figsize=(10, 6))
            plt.plot(k_values, qps, marker='o', linewidth=2, markersize=8)
            plt.xlabel('K Value')
            plt.ylabel('Queries per Second')
            plt.title('Search Performance vs K')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(Path(output_dir) / 'search_performance.png', dpi=150)
            plt.close()
        
        print(f"Plots saved to: {output_dir}")


# Example benchmark execution
def run_benchmarks():
    """Run complete benchmark suite"""
    print("Starting performance benchmarks...")
    
    benchmark = PerformanceBenchmark()
    
    # Create sample images for testing
    sample_dir = Path("benchmark_data")
    sample_dir.mkdir(exist_ok=True)
    
    sample_images = []
    for i in range(100):
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img_path = sample_dir / f"sample_{i}.jpg"
        cv2.imwrite(str(img_path), img)
        sample_images.append(str(img_path))
    
    # Run benchmarks
    print("\n1. Benchmarking feature extraction...")
    fe_results = benchmark.benchmark_feature_extraction(
        sample_images[:20],
        methods=['clip']
    )
    print(f"   CLIP: {fe_results['clip']['images_per_second']:.2f} images/sec")
    
    print("\n2. Benchmarking duplicate detection...")
    dd_results = benchmark.benchmark_duplicate_detection(sample_images[:50])
    print(f"   Total time: {dd_results['total_time']:.2f} seconds")
    print(f"   Throughput: {dd_results['images_per_second']:.2f} images/sec")
    
    # Generate reports
    print("\n3. Generating reports...")
    benchmark.generate_report()
    benchmark.plot_results()
    
    print("\nBenchmarks complete!")

if __name__ == "__main__":
    run_benchmarks()