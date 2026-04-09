"""
Distributed Feature Engineering Pipeline

This module provides utilities for running feature engineering in a distributed manner,
supporting both local multiprocessing and distributed computing frameworks.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from gnn.feature_engineering import (
    FeatureRegistry, DistributedFeatureEngine, 
    initialize_default_registry, FeatureExtractor
)

logger = logging.getLogger(__name__)


class DistributedFeaturePipeline:
    """
    Main pipeline for distributed feature engineering
    """
    
    def __init__(self, 
                 registry: Optional[FeatureRegistry] = None,
                 n_workers: Optional[int] = None,
                 output_dir: str = "./features",
                 cache_dir: str = "./feature_cache"):
        """
        Initialize the distributed feature pipeline
        
        Args:
            registry: Feature registry (default: initialize with default extractors)
            n_workers: Number of parallel workers (default: CPU count)
            output_dir: Directory for output features
            cache_dir: Directory for caching intermediate results
        """
        self.registry = registry or initialize_default_registry()
        self.n_workers = n_workers or mp.cpu_count()
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize feature engine
        self.engine = DistributedFeatureEngine(self.registry, self.n_workers)
        
        logger.info(f"Initialized DistributedFeaturePipeline with {self.n_workers} workers")
    
    def process_transaction_data(self,
                                transaction_df: pd.DataFrame,
                                graph_data: Optional[Any] = None,
                                extractor_names: Optional[List[str]] = None,
                                use_cache: bool = True) -> np.ndarray:
        """
        Process transaction data and extract features in parallel
        
        Args:
            transaction_df: DataFrame with transaction data
            graph_data: HeteroData graph object (optional)
            extractor_names: List of extractor names to use (None = all)
            use_cache: Whether to use cached features if available
        
        Returns:
            Combined feature matrix
        """
        # Prepare data dictionary
        data = {
            'transaction_df': transaction_df,
            'graph': graph_data,
            'node_features': (
                graph_data['target'].x.cpu().numpy()
                if graph_data is not None
                and 'target' in graph_data.node_types
                and hasattr(graph_data['target'], 'x')
                and graph_data['target'].x is not None
                else None
            ),
            'node_type': 'target'
        }
        
        # Check cache
        cache_key = self._get_cache_key(transaction_df, extractor_names)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
        
        if use_cache and os.path.exists(cache_path):
            logger.info(f"Loading cached features from {cache_path}")
            return np.load(cache_path)
        
        # Extract features in parallel
        logger.info(f"Extracting features using {len(extractor_names or self.registry.list_extractors())} extractors")
        feature_dict = self.engine.extract_features_parallel(
            data, 
            extractor_names=extractor_names,
            output_dir=self.output_dir
        )
        
        # Combine features
        combined_features = self.engine.combine_features(feature_dict, target_length=len(transaction_df))
        
        # Cache results
        if use_cache:
            np.save(cache_path, combined_features)
            logger.info(f"Cached features to {cache_path}")
        
        return combined_features
    
    def process_batch(self,
                     batch_data: List[Dict[str, Any]],
                     extractor_names: Optional[List[str]] = None) -> List[np.ndarray]:
        """
        Process multiple batches in parallel
        
        Args:
            batch_data: List of data dictionaries, one per batch
            extractor_names: List of extractor names to use
        
        Returns:
            List of feature arrays, one per batch
        """
        logger.info(f"Processing {len(batch_data)} batches in parallel")
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(self.process_transaction_data, 
                              batch['transaction_df'],
                              batch.get('graph'),
                              extractor_names,
                              use_cache=False)
                for batch in batch_data
            ]
            
            results = []
            for future in as_completed(futures):
                try:
                    features = future.result()
                    results.append(features)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    results.append(np.array([]))
        
        return results
    
    def _get_cache_key(self, df: pd.DataFrame, extractor_names: Optional[List[str]]) -> str:
        """Generate cache key from data and extractor names"""
        import hashlib
        
        # Create hash from DataFrame shape and column names
        df_hash = hashlib.md5(
            f"{df.shape}_{list(df.columns)[:10]}".encode()
        ).hexdigest()[:8]
        
        ext_hash = hashlib.md5(
            str(sorted(extractor_names or [])).encode()
        ).hexdigest()[:8]
        
        return f"{df_hash}_{ext_hash}"
    
    def save_features(self, features: np.ndarray, filename: str):
        """Save features to disk"""
        output_path = os.path.join(self.output_dir, filename)
        np.save(output_path, features)
        logger.info(f"Saved features to {output_path}")
    
    def load_features(self, filename: str) -> np.ndarray:
        """Load features from disk"""
        input_path = os.path.join(self.output_dir, filename)
        features = np.load(input_path)
        logger.info(f"Loaded features from {input_path}, shape: {features.shape}")
        return features


def create_feature_config(config_path: str, 
                          extractor_names: List[str],
                          feature_types: Optional[List[str]] = None):
    """
    Create a configuration file for feature engineering
    
    Args:
        config_path: Path to save configuration
        extractor_names: List of extractor names to use
        feature_types: Optional list of feature types to filter
    """
    config = {
        'extractor_names': extractor_names,
        'feature_types': feature_types,
        'n_workers': mp.cpu_count(),
        'output_dir': './features',
        'cache_dir': './feature_cache'
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved feature configuration to {config_path}")


def load_feature_config(config_path: str) -> Dict[str, Any]:
    """Load feature configuration from file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

