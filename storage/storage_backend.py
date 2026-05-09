"""
Storage Backend Interfaces
Abstract base classes and implementations for CSV and Parquet
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import logging
import os

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














