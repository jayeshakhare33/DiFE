"""
Feature Store
Manages feature storage with multiple backends
"""

import pandas as pd
from typing import Optional, Dict
import logging
from .storage_backend import StorageBackend, CSVBackend, RedisBackend, ParquetBackend

logger = logging.getLogger(__name__)


class FeatureStore:
    """Manages feature storage with multiple backends"""
    
    def __init__(self, backend_type: str = 'csv', **backend_kwargs):
        """
        Initialize feature store
        
        Args:
            backend_type: Storage backend type ('csv', 'redis', or 'parquet')
            **backend_kwargs: Backend-specific keyword arguments
        """
        self.backend_type = backend_type
        
        if backend_type == 'csv':
            self.backend: StorageBackend = CSVBackend(**backend_kwargs)
        elif backend_type == 'redis':
            self.backend: StorageBackend = RedisBackend(**backend_kwargs)
        elif backend_type == 'parquet':
            self.backend: StorageBackend = ParquetBackend(**backend_kwargs)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
        
        logger.info(f"Initialized FeatureStore with backend: {backend_type}")
    
    def save_features(self, features: pd.DataFrame, key: str, **kwargs):
        """
        Save features to storage
        
        Args:
            features: DataFrame with features
            key: Storage key
            **kwargs: Additional arguments for backend
        """
        self.backend.save_features(features, key, **kwargs)
    
    def load_features(self, key: str, **kwargs) -> pd.DataFrame:
        """
        Load features from storage
        
        Args:
            key: Storage key
            **kwargs: Additional arguments for backend
            
        Returns:
            DataFrame with features
        """
        return self.backend.load_features(key, **kwargs)
    
    def exists(self, key: str) -> bool:
        """
        Check if features exist in storage
        
        Args:
            key: Storage key
            
        Returns:
            True if features exist, False otherwise
        """
        return self.backend.exists(key)
    
    def delete_features(self, key: str):
        """
        Delete features from storage
        
        Args:
            key: Storage key
        """
        self.backend.delete(key)
    
    def list_keys(self) -> list:
        """
        List all feature keys in storage
        
        Returns:
            List of keys
        """
        # This is backend-specific
        if hasattr(self.backend, 'list_keys'):
            return self.backend.list_keys()
        else:
            logger.warning("list_keys not implemented for this backend")
            return []














