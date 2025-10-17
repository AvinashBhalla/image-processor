# utils/logging_config.py

import logging
import logging.handlers
from pathlib import Path
import json
from datetime import datetime

class CustomLogger:
    """
    Enhanced logging with structured output
    """
    
    def __init__(self, name: str, log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with multiple handlers"""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler (rotating)
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}.log",
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # JSON handler for structured logs
        json_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_structured.json",
            maxBytes=10*1024*1024,
            backupCount=5
        )
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(JSONFormatter())
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.addHandler(json_handler)
        
        return logger
    
    def log_operation(self, operation: str, **kwargs):
        """Log structured operation data"""
        data = {
            'operation': operation,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        self.logger.info(json.dumps(data))


class JSONFormatter(logging.Formatter):
    """Format logs as JSON"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


# Performance monitoring
class PerformanceLogger:
    """
    Log performance metrics
    """
    
    def __init__(self):
        self.metrics = []
    
    def log_metric(self, operation: str, duration: float, **metadata):
        """Log a performance metric"""
        metric = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'duration_seconds': duration,
            **metadata
        }
        self.metrics.append(metric)
    
    def save_metrics(self, output_path: str):
        """Save metrics to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_statistics(self, operation: str = None) -> dict:
        """Get statistics for operations"""
        import numpy as np
        
        if operation:
            durations = [m['duration_seconds'] for m in self.metrics 
                        if m['operation'] == operation]
        else:
            durations = [m['duration_seconds'] for m in self.metrics]
        
        if not durations:
            return {}
        
        return {
            'count': len(durations),
            'mean': np.mean(durations),
            'median': np.median(durations),
            'min': np.min(durations),
            'max': np.max(durations),
            'std': np.std(durations),
            'total': np.sum(durations)
        }