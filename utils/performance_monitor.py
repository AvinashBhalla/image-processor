# utils/performance_monitor.py

import psutil
import time
import threading
from typing import Callable

class PerformanceMonitor:
    """
    Monitor system resources during processing
    """
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.monitoring = False
        self.thread = None
        self.callback = None
        
    def start_monitoring(self, callback: Callable[[dict], None] = None):
        """Start monitoring system resources"""
        self.callback = callback
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.thread:
            self.thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_available_gb = memory.available / (1024**3)
                
                # Get GPU metrics if available
                gpu_info = self._get_gpu_info()
                
                metrics = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'memory_available_gb': memory_available_gb,
                    'gpu_info': gpu_info,
                    'timestamp': time.time()
                }
                
                if self.callback:
                    self.callback(metrics)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Performance monitoring error: {e}")
                break
    
    def _get_gpu_info(self) -> dict:
        """Get GPU information if available"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_used = torch.cuda.memory_allocated(0)
                gpu_memory_free = gpu_memory - gpu_memory_used
                
                return {
                    'available': True,
                    'memory_total_gb': gpu_memory / (1024**3),
                    'memory_used_gb': gpu_memory_used / (1024**3),
                    'memory_free_gb': gpu_memory_free / (1024**3),
                    'memory_percent': (gpu_memory_used / gpu_memory) * 100
                }
        except:
            pass
        
        return {'available': False}
    
    @staticmethod
    def get_system_info() -> dict:
        """Get current system information"""
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        info = {
            'cpu_count': cpu_count,
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'memory_percent': memory.percent
        }
        
        # Add GPU info
        try:
            import torch
            if torch.cuda.is_available():
                info['gpu_available'] = True
                info['gpu_count'] = torch.cuda.device_count()
                info['gpu_name'] = torch.cuda.get_device_name(0)
            else:
                info['gpu_available'] = False
        except:
            info['gpu_available'] = False
        
        return info
