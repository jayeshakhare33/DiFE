"""
Storage Backend Interfaces
Abstract base classes and implementations for CSV, Redis, and Parquet
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import logging
import os
import pickle

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for storage backends"""
    
    @abstractmethod
    def save_features(self, features: pd.DataFrame, key: str, **kwargs):
        """Save features to storage"""
        pass
    
    @abstractmethod
    def load_features(self, key: str, **kwargs) -> pd.DataFrame:
        """Load features from storage"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in storage"""
        pass
    
    @abstractmethod
    def delete(self, key: str):
        """Delete key from storage"""
        pass


class CSVBackend(StorageBackend):
    """CSV-based storage backend"""
    
    def __init__(self, base_dir: str = './data/features'):
        """
        Initialize CSV backend
        
        Args:
            base_dir: Base directory for CSV files
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        logger.info(f"Initialized CSVBackend with base_dir: {base_dir}")
    
    def save_features(self, features: pd.DataFrame, key: str, **kwargs):
        """Save features to CSV"""
        file_path = os.path.join(self.base_dir, f'{key}.csv')
        features.to_csv(file_path, index=True, **kwargs)
        logger.info(f"Saved features to {file_path}")
    
    def load_features(self, key: str, **kwargs) -> pd.DataFrame:
        """Load features from CSV"""
        file_path = os.path.join(self.base_dir, f'{key}.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Features file not found: {file_path}")
        
        features = pd.read_csv(file_path, index_col=0, **kwargs)
        logger.info(f"Loaded features from {file_path}")
        return features
    
    def exists(self, key: str) -> bool:
        """Check if CSV file exists"""
        file_path = os.path.join(self.base_dir, f'{key}.csv')
        return os.path.exists(file_path)
    
    def delete(self, key: str):
        """Delete CSV file"""
        file_path = os.path.join(self.base_dir, f'{key}.csv')
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted {file_path}")


class RedisBackend(StorageBackend):
    """Redis-based storage backend"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, 
                 password: Optional[str] = None):
        """
        Initialize Redis backend
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
        """
        try:
            import redis
            self.redis_client = redis.Redis(
                host=host, port=port, db=db, password=password, decode_responses=False
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Initialized RedisBackend with host: {host}, port: {port}, db: {db}")
        except ImportError:
            raise ImportError("redis package not installed. Install with: pip install redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def save_features(self, features: pd.DataFrame, key: str, **kwargs):
        """Save features to Redis"""
        # Serialize DataFrame to pickle
        serialized = pickle.dumps(features)
        
        # Store in Redis
        self.redis_client.set(f'features:{key}', serialized)
        logger.info(f"Saved features to Redis key: features:{key}")
    
    def load_features(self, key: str, **kwargs) -> pd.DataFrame:
        """Load features from Redis"""
        serialized = self.redis_client.get(f'features:{key}')
        if serialized is None:
            raise KeyError(f"Features not found in Redis: features:{key}")
        
        features = pickle.loads(serialized)
        logger.info(f"Loaded features from Redis key: features:{key}")
        return features
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        return self.redis_client.exists(f'features:{key}') > 0
    
    def delete(self, key: str):
        """Delete key from Redis"""
        self.redis_client.delete(f'features:{key}')
        logger.info(f"Deleted Redis key: features:{key}")


class ParquetBackend(StorageBackend):
    """Parquet-based storage backend"""
    
    def __init__(self, base_dir: str = './data/features'):
        """
        Initialize Parquet backend
        
        Args:
            base_dir: Base directory for Parquet files
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        logger.info(f"Initialized ParquetBackend with base_dir: {base_dir}")
    
    def save_features(self, features: pd.DataFrame, key: str, **kwargs):
        """Save features to Parquet"""
        file_path = os.path.join(self.base_dir, f'{key}.parquet')
        features.to_parquet(file_path, index=True, **kwargs)
        logger.info(f"Saved features to {file_path}")
    
    def load_features(self, key: str, **kwargs) -> pd.DataFrame:
        """Load features from Parquet"""
        file_path = os.path.join(self.base_dir, f'{key}.parquet')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Features file not found: {file_path}")
        
        features = pd.read_parquet(file_path, **kwargs)
        logger.info(f"Loaded features from {file_path}")
        return features
    
    def exists(self, key: str) -> bool:
        """Check if Parquet file exists"""
        file_path = os.path.join(self.base_dir, f'{key}.parquet')
        return os.path.exists(file_path)
    
    def delete(self, key: str):
        """Delete Parquet file"""
        file_path = os.path.join(self.base_dir, f'{key}.parquet')
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted {file_path}")











