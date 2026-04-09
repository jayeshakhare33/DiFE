"""
Advanced Feature Extractors for Improved Fraud Detection Accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from gnn.feature_engineering import FeatureExtractor
import logging

logger = logging.getLogger(__name__)


class GraphCentralityExtractor(FeatureExtractor):
    """Extract graph centrality features"""
    
    def __init__(self):
        super().__init__("graph_centrality", "graph")
    
    def extract(self, data: Dict[str, Any], **kwargs) -> np.ndarray:
        """Extract centrality measures from graph"""
        try:
            import networkx as nx
            from torch_geometric.data import HeteroData
            
            graph = data.get('graph')
            if graph is None:
                return np.array([])
            
            # Convert to NetworkX for centrality calculations
            # This is a simplified version - in practice, you'd want to compute
            # centrality on the full heterogeneous graph
            
            features = []
            node_type = data.get('node_type', 'target')
            
            # For each target node, compute neighbor counts (simplified centrality)
            if hasattr(graph[node_type], 'x') and graph[node_type].x is not None:
                n_nodes = graph[node_type].x.shape[0]
                
                # Degree centrality (number of connections)
                degree_centrality = np.zeros(n_nodes)
                
                for edge_type in graph.edge_types:
                    src_type, rel, dst_type = edge_type
                    if dst_type == node_type:
                        edge_index = graph[edge_type].edge_index
                        if edge_index.numel() > 0:
                            target_nodes = edge_index[1].cpu().numpy()
                            unique, counts = np.unique(target_nodes, return_counts=True)
                            for node_id, count in zip(unique, counts):
                                if node_id < n_nodes:
                                    degree_centrality[node_id] += count
                
                features.append(degree_centrality)
                
                # Normalized degree centrality
                if degree_centrality.max() > 0:
                    normalized_degree = degree_centrality / (degree_centrality.max() + 1e-8)
                    features.append(normalized_degree)
                
                if features:
                    return np.column_stack(features)
        except Exception as e:
            logger.error(f"Error computing centrality features: {e}")
        
        return np.array([])
    
    def get_feature_names(self) -> List[str]:
        return ['degree_centrality', 'normalized_degree_centrality']


class PatternMatchingExtractor(FeatureExtractor):
    """Extract pattern-based features for fraud detection"""
    
    def __init__(self):
        super().__init__("pattern_matching", "risk")
    
    def extract(self, data: Dict[str, Any], **kwargs) -> np.ndarray:
        """Extract pattern matching features"""
        df = data.get('transaction_df')
        if df is None:
            return np.array([])
        
        features = []
        n_samples = len(df)
        
        # Pattern 1: Rapid successive transactions (velocity)
        if 'TransactionDT' in df.columns and 'card1' in df.columns:
            df_sorted = df.sort_values('TransactionDT')
            time_diffs = df_sorted.groupby('card1')['TransactionDT'].diff().fillna(0)
            rapid_transactions = (time_diffs < 3600).astype(int).values  # Within 1 hour
            features.append(rapid_transactions)
        
        # Pattern 2: Unusual amount patterns
        if 'TransactionAmt' in df.columns:
            amt = df['TransactionAmt'].fillna(0).values
            # Check for round numbers (potential fraud indicator)
            round_amounts = (amt % 100 == 0).astype(int)
            # Check for very small amounts
            small_amounts = (amt < 1).astype(int)
            features.append(round_amounts)
            features.append(small_amounts)
        
        # Pattern 3: Multiple cards from same address
        if 'card1' in df.columns and 'addr1' in df.columns:
            card_addr_counts = df.groupby('addr1')['card1'].nunique()
            addr_card_diversity = df['addr1'].map(card_addr_counts).fillna(0).values
            features.append(np.log1p(addr_card_diversity))
        
        # Pattern 4: Device sharing patterns
        if 'DeviceInfo' in df.columns and 'card1' in df.columns:
            device_card_counts = df.groupby('DeviceInfo')['card1'].nunique()
            device_sharing = df['DeviceInfo'].map(device_card_counts).fillna(0).values
            features.append(np.log1p(device_sharing))
        
        if features:
            return np.column_stack(features)
        return np.array([])
    
    def get_feature_names(self) -> List[str]:
        return ['rapid_transactions', 'round_amounts', 'small_amounts', 
                'addr_card_diversity', 'device_sharing']


class EmbeddingBasedExtractor(FeatureExtractor):
    """Extract features based on learned embeddings from graph"""
    
    def __init__(self, embedding_dim: int = 16):
        super().__init__("embedding_based", "graph")
        self.embedding_dim = embedding_dim
    
    def extract(self, data: Dict[str, Any], **kwargs) -> np.ndarray:
        """
        Extract features based on graph embeddings
        
        This would typically use pre-trained embeddings or compute them on-the-fly
        """
        graph = data.get('graph')
        if graph is None:
            return np.array([])
        
        # Placeholder: In practice, you'd use pre-computed node embeddings
        # For now, return empty array - this would be populated after initial training
        return np.array([])
    
    def get_feature_names(self) -> List[str]:
        return [f"embedding_{i}" for i in range(self.embedding_dim)]


class CrossFeatureExtractor(FeatureExtractor):
    """Extract cross-feature interactions"""
    
    def __init__(self):
        super().__init__("cross_features", "statistical")
    
    def extract(self, data: Dict[str, Any], **kwargs) -> np.ndarray:
        """Extract cross-feature interactions"""
        df = data.get('transaction_df')
        if df is None:
            return np.array([])
        
        features = []
        
        # Interaction: Amount × Product
        if 'TransactionAmt' in df.columns and 'ProductCD' in df.columns:
            amt = df['TransactionAmt'].fillna(0).values
            product_encoded = pd.Categorical(df['ProductCD']).codes
            amt_product = amt * (product_encoded + 1)
            features.append(np.log1p(amt_product))
        
        # Interaction: Amount × Card type
        if 'TransactionAmt' in df.columns and 'card4' in df.columns:
            amt = df['TransactionAmt'].fillna(0).values
            card_encoded = pd.Categorical(df['card4']).codes
            amt_card = amt * (card_encoded + 1)
            features.append(np.log1p(amt_card))
        
        # Interaction: Distance × Address
        if 'dist1' in df.columns and 'addr1' in df.columns:
            dist = df['dist1'].fillna(0).values
            addr_encoded = pd.Categorical(df['addr1']).codes
            dist_addr = dist * (addr_encoded + 1)
            features.append(np.log1p(np.abs(dist_addr)))
        
        if features:
            return np.column_stack(features)
        return np.array([])
    
    def get_feature_names(self) -> List[str]:
        return ['amt_product_interaction', 'amt_card_interaction', 'dist_addr_interaction']

