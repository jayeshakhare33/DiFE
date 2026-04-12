"""
Inference Service
Handles model inference and feature retrieval
"""

import torch
import dgl
import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, Optional, List
import logging

from gnn_training.gnn_model import HeteroRGCN
from storage.feature_store import FeatureStore

logger = logging.getLogger(__name__)


class InferenceService:
    """Service for model inference and feature retrieval"""
    
    def __init__(self, model_path: str, metadata_path: str, feature_store: FeatureStore,
                 device: str = 'cpu'):
        """
        Initialize inference service
        
        Args:
            model_path: Path to trained model
            metadata_path: Path to model metadata
            feature_store: Feature store instance
            device: Device to run inference on
        """
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.feature_store = feature_store
        self.device = torch.device(device)
        
        # Load model metadata
        self.metadata = self._load_metadata()
        
        # Initialize model
        self.model = self._load_model()
        
        logger.info(f"Initialized InferenceService with model: {model_path}")
    
    def _load_metadata(self) -> Dict:
        """Load model metadata"""
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        logger.info("Loaded model metadata")
        return metadata
    
    def _load_model(self) -> HeteroRGCN:
        """Load trained model"""
        model = HeteroRGCN(
            ntype_dict=self.metadata['ntype_cnt'],
            etypes=self.metadata['etypes'],
            in_size=self.metadata.get('in_size', 360),
            hidden_size=self.metadata.get('hidden_size', 16),
            out_size=self.metadata.get('out_size', 2),
            n_layers=self.metadata.get('n_layers', 3),
            embedding_size=self.metadata.get('embedding_size', 360)
        )
        
        # Load state dict
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to(self.device)
        
        logger.info("Loaded trained model")
        return model
    
    def predict(self, node_ids: List[str], graph: Optional[dgl.DGLHeteroGraph] = None) -> Dict:
        """
        Predict fraud for given node IDs
        
        Args:
            node_ids: List of node IDs to predict
            graph: Optional graph (if None, will use stored graph)
            
        Returns:
            Dictionary with predictions
        """
        logger.info(f"Predicting for {len(node_ids)} nodes")
        
        # Get features for nodes
        features = self._get_features(node_ids)
        
        # If graph not provided, create a simple graph
        if graph is None:
            graph = self._create_simple_graph(node_ids)
        
        # Normalize features
        if 'feat_mean' in self.metadata and 'feat_std' in self.metadata:
            mean = torch.tensor(self.metadata['feat_mean'], device=self.device)
            std = torch.tensor(self.metadata['feat_std'], device=self.device)
            features = (features - mean) / std
        
        # Run inference
        with torch.no_grad():
            output = self.model(graph, features)
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
        
        # Format results
        results = {
            'node_ids': node_ids,
            'predictions': preds.cpu().numpy().tolist(),
            'probabilities': probs.cpu().numpy().tolist(),
            'fraud_scores': probs[:, 1].cpu().numpy().tolist()
        }
        
        logger.info(f"Predicted {sum(results['predictions'])} fraudulent nodes")
        return results
    
    def get_features(self, node_ids: List[str], feature_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get features for given node IDs
        
        Args:
            node_ids: List of node IDs
            feature_types: Optional list of feature types to retrieve
            
        Returns:
            DataFrame with features
        """
        logger.info(f"Retrieving features for {len(node_ids)} nodes")
        
        # Try to load from feature store
        features = None
        for key in ['graph_features', 'embeddings', 'degree_features', 'centrality_features']:
            if self.feature_store.exists(key):
                try:
                    stored_features = self.feature_store.load_features(key)
                    # Filter by node IDs
                    if 'node_id' in stored_features.columns:
                        features = stored_features[stored_features['node_id'].isin(node_ids)]
                    elif stored_features.index.name == 'node_id':
                        features = stored_features.loc[node_ids]
                    break
                except Exception as e:
                    logger.warning(f"Failed to load features from {key}: {e}")
        
        if features is None or len(features) == 0:
            # Fallback: return empty DataFrame
            logger.warning("No features found in feature store")
            features = pd.DataFrame(index=node_ids)
        
        return features
    
    def _get_features(self, node_ids: List[str]) -> torch.Tensor:
        """Get features tensor for nodes"""
        # Try to get from feature store
        features_df = self.get_features(node_ids)
        
        if len(features_df) == 0:
            # Use default features (zeros)
            logger.warning("No features found, using default features")
            n_features = self.metadata.get('in_size', 360)
            features = torch.zeros(len(node_ids), n_features, device=self.device)
        else:
            # Convert to tensor
            feature_values = features_df.values
            features = torch.tensor(feature_values, dtype=torch.float32, device=self.device)
        
        return features
    
    def _create_simple_graph(self, node_ids: List[str]) -> dgl.DGLHeteroGraph:
        """Create a simple graph for inference"""
        # Create a minimal graph with just target nodes
        n_nodes = len(node_ids)
        edge_list = [(i, i) for i in range(n_nodes)]  # Self-loops
        
        # Create graph
        g = dgl.heterograph({
            ('target', 'self_relation', 'target'): edge_list
        })
        
        return g

