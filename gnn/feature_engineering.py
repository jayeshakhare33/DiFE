"""
Distributed Feature Engineering Framework for Fraud Detection

This module provides a decentralized, parallelizable feature engineering system
that can be distributed across multiple workers/nodes.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Any, Optional, Tuple
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor(ABC):
    """Base class for all feature extractors"""
    
    def __init__(self, name: str, feature_type: str):
        self.name = name
        self.feature_type = feature_type  # 'graph', 'temporal', 'statistical', 'risk'
    
    @abstractmethod
    def extract(self, data: Dict[str, Any], **kwargs) -> np.ndarray:
        """Extract features from data"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return list of feature names"""
        pass
    
    def can_parallelize(self) -> bool:
        """Whether this extractor can be parallelized"""
        return True


class GraphNeighborAggregator(FeatureExtractor):
    """Extract aggregated features from graph neighbors"""
    
    def __init__(self, aggregation_functions: List[str] = ['mean', 'std', 'max', 'min', 'sum', 'count']):
        super().__init__("graph_neighbor_aggregator", "graph")
        self.agg_funcs = aggregation_functions
    
    def extract(self, data: Dict[str, Any], **kwargs) -> np.ndarray:
        """
        Extract neighbor aggregation features
        
        Args:
            data: Dictionary containing 'graph' (HeteroData), 'node_features', 'node_type'
        """
        from torch_geometric.data import HeteroData
        import torch as th
        
        graph = data.get('graph')
        node_features = data.get('node_features')
        node_type = data.get('node_type', 'target')

        if graph is None:
            return np.array([])

        if node_features is None and hasattr(graph[node_type], 'x') and graph[node_type].x is not None:
            node_features = graph[node_type].x.cpu().numpy()
        if node_features is None:
            return np.array([])
        
        features_list = []
        
        # For each edge type connected to target nodes
        for edge_type in graph.edge_types:
            src_type, rel, dst_type = edge_type
            
            if dst_type == node_type:
                # Get edges pointing to target nodes
                edge_index = graph[edge_type].edge_index
                if edge_index.numel() == 0:
                    continue
                
                # Get source node features if available
                if hasattr(graph[src_type], 'x') and graph[src_type].x is not None:
                    src_features = graph[src_type].x.cpu().numpy()
                else:
                    # Use dummy features if not available
                    src_features = np.ones((edge_index[0].max().item() + 1, node_features.shape[1]))
                
                # Aggregate features for each target node
                target_node_ids = edge_index[1].cpu().numpy()
                source_node_ids = edge_index[0].cpu().numpy()
                
                # Group by target node
                unique_targets = np.unique(target_node_ids)
                agg_features = []
                
                for target_id in unique_targets:
                    mask = target_node_ids == target_id
                    neighbor_features = src_features[source_node_ids[mask]]
                    
                    if len(neighbor_features) == 0:
                        agg_features.append(np.zeros(len(self.agg_funcs) * node_features.shape[1]))
                        continue
                    
                    # Apply aggregation functions
                    target_agg = []
                    for func_name in self.agg_funcs:
                        if func_name == 'mean':
                            agg = np.mean(neighbor_features, axis=0)
                        elif func_name == 'std':
                            agg = np.std(neighbor_features, axis=0)
                        elif func_name == 'max':
                            agg = np.max(neighbor_features, axis=0)
                        elif func_name == 'min':
                            agg = np.min(neighbor_features, axis=0)
                        elif func_name == 'sum':
                            agg = np.sum(neighbor_features, axis=0)
                        elif func_name == 'count':
                            agg = np.array([len(neighbor_features)] * node_features.shape[1])
                        else:
                            agg = np.zeros(node_features.shape[1])
                        target_agg.append(agg)
                    
                    agg_features.append(np.concatenate(target_agg))
                
                # Map back to original order
                feature_matrix = np.zeros((len(node_features), len(agg_features[0])))
                for i, target_id in enumerate(unique_targets):
                    if target_id < len(feature_matrix):
                        feature_matrix[target_id] = agg_features[i]
                
                features_list.append(feature_matrix)
        
        if features_list:
            return np.hstack(features_list)
        return np.array([])
    
    def get_feature_names(self) -> List[str]:
        names = []
        for func in self.agg_funcs:
            names.extend([f"neighbor_{func}_{i}" for i in range(100)])  # Placeholder, actual size varies
        return names


class TemporalFeatureExtractor(FeatureExtractor):
    """Extract temporal features from transaction data"""
    
    def __init__(self):
        super().__init__("temporal_features", "temporal")
    
    def extract(self, data: Dict[str, Any], **kwargs) -> np.ndarray:
        """
        Extract temporal features
        
        Args:
            data: Dictionary containing 'transaction_df' with TransactionDT column
        """
        df = data.get('transaction_df')
        if df is None or 'TransactionDT' not in df.columns:
            return np.array([])
        
        features = []
        
        # Time-based features
        if 'TransactionDT' in df.columns:
            dt = df['TransactionDT'].values
            
            # Hour of day (cyclical encoding)
            hour_sin = np.sin(2 * np.pi * (dt % 86400) / 86400)
            hour_cos = np.cos(2 * np.pi * (dt % 86400) / 86400)
            
            # Day of week (cyclical encoding)
            day_sin = np.sin(2 * np.pi * (dt // 86400 % 7) / 7)
            day_cos = np.cos(2 * np.pi * (dt // 86400 % 7) / 7)
            
            # Time since last transaction (if available)
            time_diff = np.diff(np.concatenate([[0], dt]))
            time_diff = np.concatenate([[0], time_diff])
            
            # Log transform
            time_diff_log = np.log1p(np.abs(time_diff))
            
            features.extend([hour_sin, hour_cos, day_sin, day_cos, time_diff, time_diff_log])
        
        if features:
            return np.column_stack(features)
        return np.array([])
    
    def get_feature_names(self) -> List[str]:
        return ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'time_diff', 'time_diff_log']


class StatisticalFeatureExtractor(FeatureExtractor):
    """Extract statistical features"""
    
    def __init__(self):
        super().__init__("statistical_features", "statistical")
    
    def extract(self, data: Dict[str, Any], **kwargs) -> np.ndarray:
        """
        Extract statistical features
        
        Args:
            data: Dictionary containing 'transaction_df' with numerical columns
        """
        df = data.get('transaction_df')
        if df is None:
            return np.array([])
        
        features = []
        numerical_cols = list(df.select_dtypes(include=[np.number]).columns[:30])
        if not numerical_cols:
            return np.array([])

        # Keep statistical features row-aligned: one row per sample.
        for col in numerical_cols:
            values = df[col].astype(float).fillna(0.0).values
            col_std = np.std(values)
            if col_std < 1e-8:
                col_std = 1.0
            zscore = (values - np.mean(values)) / col_std
            abs_dev = np.abs(values - np.median(values))
            features.append(zscore)
            features.append(abs_dev)

        if features:
            return np.column_stack(features)
        return np.array([])
    
    def get_feature_names(self) -> List[str]:
        return [f"stat_{stat}_{i}" for stat in ['mean', 'std', 'min', 'max', 'p25', 'p50', 'p75', 'skew', 'kurt'] 
                for i in range(50)]


class RiskScoreExtractor(FeatureExtractor):
    """Extract risk scoring features based on patterns"""
    
    def __init__(self):
        super().__init__("risk_score_features", "risk")
    
    def extract(self, data: Dict[str, Any], **kwargs) -> np.ndarray:
        """
        Extract risk scoring features
        
        Args:
            data: Dictionary containing transaction and graph data
        """
        df = data.get('transaction_df')
        graph = data.get('graph')
        
        if df is None:
            return np.array([])
        
        features = []
        n_samples = len(df)
        
        # Risk feature 1: Transaction amount anomalies
        if 'TransactionAmt' in df.columns:
            amt = df['TransactionAmt'].fillna(0).values
            amt_log = np.log1p(amt)
            amt_zscore = (amt - np.mean(amt)) / (np.std(amt) + 1e-8)
            features.append(amt_log)
            features.append(amt_zscore)
        
        # Risk feature 2: Card velocity (if card info available)
        if 'card1' in df.columns:
            card_counts = df['card1'].value_counts().to_dict()
            card_velocity = df['card1'].map(card_counts).fillna(0).values
            features.append(np.log1p(card_velocity))
        
        # Risk feature 3: Address frequency
        if 'addr1' in df.columns:
            addr_counts = df['addr1'].value_counts().to_dict()
            addr_freq = df['addr1'].map(addr_counts).fillna(0).values
            features.append(np.log1p(addr_freq))
        
        # Risk feature 4: Email domain risk
        if 'P_emaildomain' in df.columns:
            email_counts = df['P_emaildomain'].value_counts().to_dict()
            email_freq = df['P_emaildomain'].map(email_counts).fillna(0).values
            features.append(np.log1p(email_freq))
        
        # Risk feature 5: Product code distribution
        if 'ProductCD' in df.columns:
            product_counts = df['ProductCD'].value_counts().to_dict()
            product_freq = df['ProductCD'].map(product_counts).fillna(0).values
            features.append(np.log1p(product_freq))
        
        if features:
            return np.column_stack(features)
        return np.array([])
    
    def get_feature_names(self) -> List[str]:
        return ['amt_log', 'amt_zscore', 'card_velocity', 'addr_freq', 'email_freq', 'product_freq']


class FeatureRegistry:
    """Registry for managing feature extractors"""
    
    def __init__(self):
        self.extractors: Dict[str, FeatureExtractor] = {}
    
    def register(self, extractor: FeatureExtractor):
        """Register a feature extractor"""
        self.extractors[extractor.name] = extractor
        logger.info(f"Registered feature extractor: {extractor.name}")
    
    def get_extractor(self, name: str) -> Optional[FeatureExtractor]:
        """Get a feature extractor by name"""
        return self.extractors.get(name)
    
    def list_extractors(self) -> List[str]:
        """List all registered extractor names"""
        return list(self.extractors.keys())
    
    def get_extractors_by_type(self, feature_type: str) -> List[FeatureExtractor]:
        """Get all extractors of a specific type"""
        return [ext for ext in self.extractors.values() if ext.feature_type == feature_type]


class DistributedFeatureEngine:
    """Distributed feature engineering engine"""
    
    def __init__(self, registry: FeatureRegistry, n_workers: Optional[int] = None):
        self.registry = registry
        self.n_workers = n_workers or mp.cpu_count()
        logger.info(f"Initialized DistributedFeatureEngine with {self.n_workers} workers")
    
    def extract_features_parallel(self, data: Dict[str, Any], 
                                 extractor_names: Optional[List[str]] = None,
                                 output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Extract features in parallel across multiple workers
        
        Args:
            data: Input data dictionary
            extractor_names: List of extractor names to use (None = all)
            output_dir: Directory to save intermediate results
        
        Returns:
            Dictionary mapping extractor names to feature arrays
        """
        if extractor_names is None:
            extractor_names = self.registry.list_extractors()
        
        # Filter to only parallelizable extractors
        extractors = [self.registry.get_extractor(name) for name in extractor_names 
                      if self.registry.get_extractor(name) is not None]
        extractors = [ext for ext in extractors if ext.can_parallelize()]
        
        results = {}
        
        # Use ProcessPoolExecutor for CPU-bound tasks
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit tasks
            future_to_extractor = {
                executor.submit(self._extract_features_worker, ext, data, output_dir): ext.name
                for ext in extractors
            }
            
            # Collect results
            for future in as_completed(future_to_extractor):
                extractor_name = future_to_extractor[future]
                try:
                    features = future.result()
                    results[extractor_name] = features
                    logger.info(f"Completed feature extraction: {extractor_name}, shape: {features.shape}")
                except Exception as e:
                    logger.error(f"Error extracting features from {extractor_name}: {e}")
                    results[extractor_name] = np.array([])
        
        return results
    
    @staticmethod
    def _extract_features_worker(extractor: FeatureExtractor, data: Dict[str, Any], 
                                 output_dir: Optional[str]) -> np.ndarray:
        """Worker function for parallel feature extraction"""
        try:
            features = extractor.extract(data)
            
            # Save intermediate results if output_dir provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{extractor.name}.npy")
                np.save(output_path, features)
            
            return features
        except Exception as e:
            logger.error(f"Error in worker for {extractor.name}: {e}")
            return np.array([])
    
    def combine_features(self, feature_dict: Dict[str, np.ndarray], 
                        target_length: Optional[int] = None) -> np.ndarray:
        """
        Combine features from multiple extractors
        
        Args:
            feature_dict: Dictionary of extractor names to feature arrays
            target_length: Expected length of feature vectors (for padding/trimming)
        
        Returns:
            Combined feature matrix
        """
        valid_features = [v for v in feature_dict.values() if isinstance(v, np.ndarray) and v.size > 0]
        if not valid_features:
            return np.array([])

        if target_length is None:
            row_lengths = [(v.shape[0] if v.ndim > 1 else len(v)) for v in valid_features]
            target_length = min(row_lengths) if row_lengths else 0
            if target_length == 0:
                return np.array([])

        aligned_features = []
        for features in valid_features:
            if features.ndim == 1:
                features = features.reshape(-1, 1)
            elif features.ndim > 2:
                features = features.reshape(features.shape[0], -1)

            if features.shape[0] < target_length:
                padding = np.zeros((target_length - features.shape[0], features.shape[1]))
                features = np.vstack([features, padding])
            elif features.shape[0] > target_length:
                features = features[:target_length]

            aligned_features.append(features)

        return np.hstack(aligned_features) if aligned_features else np.array([])


def initialize_default_registry() -> FeatureRegistry:
    """Initialize registry with default feature extractors"""
    registry = FeatureRegistry()
    
    # Register default extractors
    registry.register(GraphNeighborAggregator())
    registry.register(TemporalFeatureExtractor())
    registry.register(StatisticalFeatureExtractor())
    registry.register(RiskScoreExtractor())
    
    return registry

