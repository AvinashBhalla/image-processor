from dataclasses import dataclass, field
from typing import List
import yaml
from pathlib import Path

@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction"""
    model_name: str = "openai/clip-vit-base-patch32"
    use_gpu: bool = True
    batch_size: int = 32
    cache_features: bool = True
    max_image_dimension: int = 1024


@dataclass
class SimilaritySearchConfig:
    """Configuration for similarity search"""
    index_type: str = "IVF"  # Options: Flat, IVF, HNSW
    nlist: int = 100  # Number of clusters for IVF
    nprobe: int = 10  # Number of clusters to search
    similarity_threshold: float = 0.7
    max_results: int = 100


@dataclass
class DuplicateDetectionConfig:
    """Configuration for duplicate detection"""
    hash_threshold: int = 5
    embedding_threshold: float = 0.95
    ssim_threshold: float = 0.90
    enable_stage1: bool = True  # Perceptual hashing
    enable_stage2: bool = True  # Embeddings
    enable_stage3: bool = True  # SSIM


@dataclass
class SystemConfig:
    """System-wide configuration"""
    n_workers: int = 4
    cache_dir: str = "data/cache"
    database_path: str = "data/images.db"
    index_path: str = "data/faiss.index"
    log_level: str = "INFO"
    memory_limit_gb: float = 4.0
    
    # Feature extraction
    feature_extraction: FeatureExtractionConfig = field(
        default_factory=FeatureExtractionConfig
    )
    
    # Similarity search
    similarity_search: SimilaritySearchConfig = field(
        default_factory=SimilaritySearchConfig
    )
    
    # Duplicate detection
    duplicate_detection: DuplicateDetectionConfig = field(
        default_factory=DuplicateDetectionConfig
    )
    
    def save(self, path: str = "config.yaml"):
        """Save configuration to YAML file"""
        config_dict = {
            'n_workers': self.n_workers,
            'cache_dir': self.cache_dir,
            'database_path': self.database_path,
            'index_path': self.index_path,
            'log_level': self.log_level,
            'memory_limit_gb': self.memory_limit_gb,
            'feature_extraction': {
                'model_name': self.feature_extraction.model_name,
                'use_gpu': self.feature_extraction.use_gpu,
                'batch_size': self.feature_extraction.batch_size,
                'cache_features': self.feature_extraction.cache_features,
                'max_image_dimension': self.feature_extraction.max_image_dimension
            },
            'similarity_search': {
                'index_type': self.similarity_search.index_type,
                'nlist': self.similarity_search.nlist,
                'nprobe': self.similarity_search.nprobe,
                'similarity_threshold': self.similarity_search.similarity_threshold,
                'max_results': self.similarity_search.max_results
            },
            'duplicate_detection': {
                'hash_threshold': self.duplicate_detection.hash_threshold,
                'embedding_threshold': self.duplicate_detection.embedding_threshold,
                'ssim_threshold': self.duplicate_detection.ssim_threshold,
                'enable_stage1': self.duplicate_detection.enable_stage1,
                'enable_stage2': self.duplicate_detection.enable_stage2,
                'enable_stage3': self.duplicate_detection.enable_stage3
            }
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @classmethod
    def load(cls, path: str = "config.yaml") -> 'SystemConfig':
        """Load configuration from YAML file"""
        if not Path(path).exists():
            return cls()  # Return default config
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        
        # Load system settings
        config.n_workers = config_dict.get('n_workers', config.n_workers)
        config.cache_dir = config_dict.get('cache_dir', config.cache_dir)
        config.database_path = config_dict.get('database_path', config.database_path)
        config.index_path = config_dict.get('index_path', config.index_path)
        config.log_level = config_dict.get('log_level', config.log_level)
        config.memory_limit_gb = config_dict.get('memory_limit_gb', config.memory_limit_gb)
        
        # Load feature extraction settings
        if 'feature_extraction' in config_dict:
            fe = config_dict['feature_extraction']
            config.feature_extraction = FeatureExtractionConfig(
                model_name=fe.get('model_name', config.feature_extraction.model_name),
                use_gpu=fe.get('use_gpu', config.feature_extraction.use_gpu),
                batch_size=fe.get('batch_size', config.feature_extraction.batch_size),
                cache_features=fe.get('cache_features', config.feature_extraction.cache_features),
                max_image_dimension=fe.get('max_image_dimension', config.feature_extraction.max_image_dimension)
            )
        
        # Load similarity search settings
        if 'similarity_search' in config_dict:
            ss = config_dict['similarity_search']
            config.similarity_search = SimilaritySearchConfig(
                index_type=ss.get('index_type', config.similarity_search.index_type),
                nlist=ss.get('nlist', config.similarity_search.nlist),
                nprobe=ss.get('nprobe', config.similarity_search.nprobe),
                similarity_threshold=ss.get('similarity_threshold', config.similarity_search.similarity_threshold),
                max_results=ss.get('max_results', config.similarity_search.max_results)
            )
        
        # Load duplicate detection settings
        if 'duplicate_detection' in config_dict:
            dd = config_dict['duplicate_detection']
            config.duplicate_detection = DuplicateDetectionConfig(
                hash_threshold=dd.get('hash_threshold', config.duplicate_detection.hash_threshold),
                embedding_threshold=dd.get('embedding_threshold', config.duplicate_detection.embedding_threshold),
                ssim_threshold=dd.get('ssim_threshold', config.duplicate_detection.ssim_threshold),
                enable_stage1=dd.get('enable_stage1', config.duplicate_detection.enable_stage1),
                enable_stage2=dd.get('enable_stage2', config.duplicate_detection.enable_stage2),
                enable_stage3=dd.get('enable_stage3', config.duplicate_detection.enable_stage3)
            )
        
        return config