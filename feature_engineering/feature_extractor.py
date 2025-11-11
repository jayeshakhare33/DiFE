"""
Graph-based Feature Extraction
Extracts comprehensive user-level node features and transaction-level edge features
"""

import numpy as np
import pandas as pd
import dgl
import torch as th
import networkx as nx
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict
from datetime import datetime
from scipy import stats

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts comprehensive user-level node features"""
    
    def __init__(self, transaction_df: Optional[pd.DataFrame] = None):
        """
        Initialize feature extractor
        
        Args:
            transaction_df: Optional transaction dataframe for computing transaction-based features
        """
        self.transaction_df = transaction_df
        logger.info("Initialized FeatureExtractor")
    
    def extract_all_features(self, g: dgl.DGLHeteroGraph, node_type: str = 'user',
                            transaction_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Extract all user-level node features (50 features total)
        
        Args:
            g: DGL heterogeneous graph
            node_type: Node type to extract features for (default: 'user')
            transaction_df: Transaction dataframe for computing transaction-based features
            
        Returns:
            DataFrame with extracted features
        """
        logger.info(f"Extracting features for node type: {node_type}")
        
        if transaction_df is None:
            transaction_df = self.transaction_df
        
        features = {}
        node_ids = g.nodes(node_type).numpy()
        
        # A. Transaction Statistics Features (15 features)
        transaction_features = self.extract_transaction_statistics(g, node_type, transaction_df)
        features.update(transaction_features)
        
        # B. Graph Topology Features (12 features)
        topology_features = self.extract_graph_topology_features(g, node_type)
        features.update(topology_features)
        
        # C. Temporal Features (10 features)
        temporal_features = self.extract_temporal_features(g, node_type, transaction_df)
        features.update(temporal_features)
        
        # D. Behavioral Features (8 features)
        behavioral_features = self.extract_behavioral_features(g, node_type, transaction_df)
        features.update(behavioral_features)
        
        # E. Fraud Propagation Features (5 features)
        fraud_features = self.extract_fraud_propagation_features(g, node_type, transaction_df)
        features.update(fraud_features)
        
        # Create DataFrame
        feature_df = pd.DataFrame(features, index=node_ids)
        logger.info(f"Extracted {len(feature_df.columns)} features for {len(feature_df)} nodes")
        
        return feature_df
    
    def extract_transaction_statistics(self, g: dgl.DGLHeteroGraph, node_type: str = 'user',
                                      transaction_df: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """
        Extract transaction statistics features (15 features)
        
        1. total_transactions_sent
        2. total_transactions_received
        3. avg_transaction_amount_sent
        4. avg_transaction_amount_received
        5. max_transaction_amount_sent
        6. max_transaction_amount_received
        7. std_transaction_amount_sent
        8. std_transaction_amount_received
        9. total_amount_sent
        10. total_amount_received
        11. net_flow
        12. transaction_frequency
        13. unique_receivers
        14. unique_senders
        15. avg_time_between_transactions
        """
        logger.info("Extracting transaction statistics features")
        
        node_ids = g.nodes(node_type).numpy()
        n_nodes = len(node_ids)
        
        # Initialize all features
        features = {
            'total_transactions_sent': np.zeros(n_nodes),
            'total_transactions_received': np.zeros(n_nodes),
            'avg_transaction_amount_sent': np.zeros(n_nodes),
            'avg_transaction_amount_received': np.zeros(n_nodes),
            'max_transaction_amount_sent': np.zeros(n_nodes),
            'max_transaction_amount_received': np.zeros(n_nodes),
            'std_transaction_amount_sent': np.zeros(n_nodes),
            'std_transaction_amount_received': np.zeros(n_nodes),
            'total_amount_sent': np.zeros(n_nodes),
            'total_amount_received': np.zeros(n_nodes),
            'net_flow': np.zeros(n_nodes),
            'transaction_frequency': np.zeros(n_nodes),
            'unique_receivers': np.zeros(n_nodes),
            'unique_senders': np.zeros(n_nodes),
            'avg_time_between_transactions': np.zeros(n_nodes)
        }
        
        if transaction_df is None:
            return features
        
        # Get transaction edge type (assuming 'transaction' edge type exists)
        # If not, we'll compute from out/in degrees
        for i, node_id in enumerate(node_ids):
            node_tensor = th.tensor([node_id], dtype=th.long)
            
            # Get out edges (sent transactions)
            out_edges = []
            out_amounts = []
            out_timestamps = []
            out_receivers = set()
            
            for etype in g.canonical_etypes:
                src, etype_name, dst = etype
                if src == node_type and 'transaction' in etype_name.lower():
                    out_src, out_dst = g.out_edges(node_tensor, etype=etype)
                    if len(out_src) > 0:
                        out_edges.extend(out_dst.tolist())
                        # Get edge features if available
                        if 'amount' in g.edges[etype].data:
                            out_amounts.extend(g.edges[etype].data['amount'][g.out_edges(node_tensor, etype=etype)[1]].tolist())
                        if 'timestamp' in g.edges[etype].data:
                            out_timestamps.extend(g.edges[etype].data['timestamp'][g.out_edges(node_tensor, etype=etype)[1]].tolist())
                        out_receivers.update(out_dst.tolist())
            
            # Get in edges (received transactions)
            in_edges = []
            in_amounts = []
            in_timestamps = []
            in_senders = set()
            
            for etype in g.canonical_etypes:
                src, etype_name, dst = etype
                if dst == node_type and 'transaction' in etype_name.lower():
                    in_src, in_dst = g.in_edges(node_tensor, etype=etype)
                    if len(in_src) > 0:
                        in_edges.extend(in_src.tolist())
                        # Get edge features if available
                        if 'amount' in g.edges[etype].data:
                            in_amounts.extend(g.edges[etype].data['amount'][g.in_edges(node_tensor, etype=etype)[0]].tolist())
                        if 'timestamp' in g.edges[etype].data:
                            in_timestamps.extend(g.edges[etype].data['timestamp'][g.in_edges(node_tensor, etype=etype)[0]].tolist())
                        in_senders.update(in_src.tolist())
            
            # Compute features
            features['total_transactions_sent'][i] = len(out_edges)
            features['total_transactions_received'][i] = len(in_edges)
            features['unique_receivers'][i] = len(out_receivers)
            features['unique_senders'][i] = len(in_senders)
            
            if len(out_amounts) > 0:
                out_amounts = np.array(out_amounts)
                features['total_amount_sent'][i] = np.sum(out_amounts)
                features['avg_transaction_amount_sent'][i] = np.mean(out_amounts)
                features['max_transaction_amount_sent'][i] = np.max(out_amounts)
                features['std_transaction_amount_sent'][i] = np.std(out_amounts) if len(out_amounts) > 1 else 0.0
            else:
                features['std_transaction_amount_sent'][i] = 0.0
            
            if len(in_amounts) > 0:
                in_amounts = np.array(in_amounts)
                features['total_amount_received'][i] = np.sum(in_amounts)
                features['avg_transaction_amount_received'][i] = np.mean(in_amounts)
                features['max_transaction_amount_received'][i] = np.max(in_amounts)
                features['std_transaction_amount_received'][i] = np.std(in_amounts) if len(in_amounts) > 1 else 0.0
            else:
                features['std_transaction_amount_received'][i] = 0.0
            
            features['net_flow'][i] = features['total_amount_received'][i] - features['total_amount_sent'][i]
            
            total_transactions = len(out_edges) + len(in_edges)
            if total_transactions > 0:
                # Transaction frequency (transactions per day, assuming we have time range)
                if len(out_timestamps) > 0 or len(in_timestamps) > 0:
                    all_timestamps = sorted(out_timestamps + in_timestamps)
                    if len(all_timestamps) > 1:
                        time_span = max(all_timestamps) - min(all_timestamps)
                        features['transaction_frequency'][i] = total_transactions / (time_span + 1)  # +1 to avoid division by zero
                        
                        # Average time between transactions
                        time_diffs = [all_timestamps[j+1] - all_timestamps[j] for j in range(len(all_timestamps)-1)]
                        if len(time_diffs) > 0:
                            features['avg_time_between_transactions'][i] = np.mean(time_diffs)
        
        return features
    
    def extract_graph_topology_features(self, g: dgl.DGLHeteroGraph, node_type: str = 'user') -> Dict[str, np.ndarray]:
        """
        Extract graph topology features (12 features)
        
        16. in_degree
        17. out_degree
        18. total_degree
        19. degree_centrality
        20. betweenness_centrality
        21. closeness_centrality
        22. pagerank_score
        23. clustering_coefficient
        24. katz_centrality
        25. eigenvector_centrality
        26. average_neighbor_degree
        27. triangles_count
        """
        logger.info("Extracting graph topology features")
        
        node_ids = g.nodes(node_type).numpy()
        n_nodes = len(node_ids)
        logger.info(f"Processing {n_nodes} nodes for topology features")
        
        features = {
            'in_degree': np.zeros(n_nodes),
            'out_degree': np.zeros(n_nodes),
            'total_degree': np.zeros(n_nodes),
            'degree_centrality': np.zeros(n_nodes),
            'betweenness_centrality': np.zeros(n_nodes),
            'closeness_centrality': np.zeros(n_nodes),
            'pagerank_score': np.zeros(n_nodes),
            'clustering_coefficient': np.zeros(n_nodes),
            'katz_centrality': np.zeros(n_nodes),
            'eigenvector_centrality': np.zeros(n_nodes),
            'average_neighbor_degree': np.zeros(n_nodes),
            'triangles_count': np.zeros(n_nodes)
        }
        
        # Compute degrees
        logger.info("Computing node degrees...")
        total_in_degree = np.zeros(n_nodes)
        total_out_degree = np.zeros(n_nodes)
        
        for etype in g.canonical_etypes:
            src, _, dst = etype
            if src == node_type:
                out_degrees = g.out_degrees(g.nodes(node_type), etype=etype).numpy()
                total_out_degree += out_degrees
            if dst == node_type:
                in_degrees = g.in_degrees(g.nodes(node_type), etype=etype).numpy()
                total_in_degree += in_degrees
        
        features['in_degree'] = total_in_degree
        features['out_degree'] = total_out_degree
        features['total_degree'] = total_in_degree + total_out_degree
        
        # Convert to NetworkX for centrality calculations
        try:
            logger.info("Converting to NetworkX graph...")
            target_nodes = g.nodes(node_type)
            
            # Create adjacency matrix
            adj_dict = defaultdict(set)
            for etype in g.canonical_etypes:
                src, _, dst = etype
                if src == node_type and dst == node_type:
                    src_nodes, dst_nodes = g.edges(etype=etype)
                    for s, d in zip(src_nodes.tolist(), dst_nodes.tolist()):
                        adj_dict[s].add(d)
            
            # Create NetworkX graph
            nx_g = nx.DiGraph()
            for node in target_nodes.tolist():
                nx_g.add_node(node)
            for src, dsts in adj_dict.items():
                for dst in dsts:
                    nx_g.add_edge(src, dst)
            
            logger.info(f"NetworkX graph created: {len(nx_g.nodes())} nodes, {len(nx_g.edges())} edges")
            
            if len(nx_g.nodes()) > 0:
                # Degree centrality (fast)
                logger.info("Computing degree centrality...")
                degree_centrality = nx.degree_centrality(nx_g)
                features['degree_centrality'] = np.array([degree_centrality.get(n, 0) for n in node_ids])
                
                # Betweenness centrality (slow for large graphs)
                logger.info("Computing betweenness centrality...")
                if len(nx_g.nodes()) < 5000:
                    betweenness = nx.betweenness_centrality(nx_g)
                    features['betweenness_centrality'] = np.array([betweenness.get(n, 0) for n in node_ids])
                elif len(nx_g.nodes()) < 50000:
                    sample_nodes = np.random.choice(list(nx_g.nodes()), min(1000, len(nx_g.nodes())), replace=False)
                    betweenness = nx.betweenness_centrality(nx_g, k=len(sample_nodes))
                    features['betweenness_centrality'] = np.array([betweenness.get(n, 0) for n in node_ids])
                else:
                    logger.warning("Graph too large for betweenness centrality, skipping...")
                    features['betweenness_centrality'] = np.zeros(n_nodes)
                
                # Closeness centrality (very slow for large graphs)
                logger.info("Computing closeness centrality...")
                if len(nx_g.nodes()) < 3000:
                    closeness = nx.closeness_centrality(nx_g)
                    features['closeness_centrality'] = np.array([closeness.get(n, 0) for n in node_ids])
                else:
                    logger.warning("Graph too large for closeness centrality, skipping...")
                    features['closeness_centrality'] = np.zeros(n_nodes)
                
                # PageRank (moderately fast)
                logger.info("Computing PageRank...")
                pagerank = nx.pagerank(nx_g, max_iter=100, tol=1e-06)
                features['pagerank_score'] = np.array([pagerank.get(n, 0) for n in node_ids])
                
                # Katz centrality (can be slow)
                logger.info("Computing Katz centrality...")
                if len(nx_g.nodes()) < 10000:
                    try:
                        katz = nx.katz_centrality(nx_g, max_iter=100, tol=1e-06)
                        features['katz_centrality'] = np.array([katz.get(n, 0) for n in node_ids])
                    except:
                        logger.warning("Katz centrality failed, using zeros")
                        features['katz_centrality'] = np.zeros(n_nodes)
                else:
                    logger.warning("Graph too large for Katz centrality, skipping...")
                    features['katz_centrality'] = np.zeros(n_nodes)
                
                # Eigenvector centrality (can be slow)
                logger.info("Computing eigenvector centrality...")
                if len(nx_g.nodes()) < 10000:
                    try:
                        eigenvector = nx.eigenvector_centrality(nx_g, max_iter=100, tol=1e-06)
                        features['eigenvector_centrality'] = np.array([eigenvector.get(n, 0) for n in node_ids])
                    except:
                        logger.warning("Eigenvector centrality failed, using zeros")
                        features['eigenvector_centrality'] = np.zeros(n_nodes)
                else:
                    logger.warning("Graph too large for eigenvector centrality, skipping...")
                    features['eigenvector_centrality'] = np.zeros(n_nodes)
                
                # Clustering coefficient (moderately fast)
                logger.info("Computing clustering coefficient...")
                if len(nx_g.nodes()) < 50000:
                    clustering = nx.clustering(nx_g.to_undirected())
                    features['clustering_coefficient'] = np.array([clustering.get(n, 0) for n in node_ids])
                else:
                    logger.warning("Graph too large for clustering coefficient, skipping...")
                    features['clustering_coefficient'] = np.zeros(n_nodes)
                
                # Average neighbor degree (fast)
                logger.info("Computing average neighbor degree...")
                avg_neighbor_degree = {}
                for node in node_ids:
                    neighbors = list(nx_g.neighbors(node))
                    if len(neighbors) > 0:
                        neighbor_degrees = [nx_g.degree(n) for n in neighbors]
                        avg_neighbor_degree[node] = np.mean(neighbor_degrees) if neighbor_degrees else 0.0
                    else:
                        avg_neighbor_degree[node] = 0.0
                features['average_neighbor_degree'] = np.array([avg_neighbor_degree.get(n, 0) for n in node_ids])
                
                # Triangles count (can be slow for large graphs)
                logger.info("Computing triangles count...")
                if len(nx_g.nodes()) < 20000:
                    triangles = nx.triangles(nx_g.to_undirected())
                    features['triangles_count'] = np.array([triangles.get(n, 0) for n in node_ids])
                else:
                    logger.warning("Graph too large for triangles count, skipping...")
                    features['triangles_count'] = np.zeros(n_nodes)
            else:
                # Empty graph - all zeros already set
                logger.warning("Empty graph, all topology features set to zero")
                
        except Exception as e:
            logger.warning(f"Topology calculation failed: {e}, using zeros")
            import traceback
            logger.debug(traceback.format_exc())
        
        return features
    
    def extract_temporal_features(self, g: dgl.DGLHeteroGraph, node_type: str = 'user',
                                  transaction_df: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """
        Extract temporal features (10 features)
        
        28. account_age_days
        29. first_transaction_timestamp
        30. last_transaction_timestamp
        31. time_since_last_transaction
        32. transactions_last_24h
        33. transactions_last_7d
        34. transactions_last_30d
        35. hour_of_day_mode
        36. day_of_week_mode
        37. transaction_time_variance
        """
        logger.info("Extracting temporal features")
        
        node_ids = g.nodes(node_type).numpy()
        n_nodes = len(node_ids)
        
        features = {
            'account_age_days': np.zeros(n_nodes),
            'first_transaction_timestamp': np.zeros(n_nodes),
            'last_transaction_timestamp': np.zeros(n_nodes),
            'time_since_last_transaction': np.zeros(n_nodes),
            'transactions_last_24h': np.zeros(n_nodes),
            'transactions_last_7d': np.zeros(n_nodes),
            'transactions_last_30d': np.zeros(n_nodes),
            'hour_of_day_mode': np.zeros(n_nodes),
            'day_of_week_mode': np.zeros(n_nodes),
            'transaction_time_variance': np.zeros(n_nodes)
        }
        
        if transaction_df is None:
            return features
        
        # Get current timestamp (assuming we have a reference)
        current_timestamp = transaction_df.get('timestamp', pd.Series([0])).max() if 'timestamp' in transaction_df.columns else 0
        
        for i, node_id in enumerate(node_ids):
            # Get all timestamps for this user
            timestamps = []
            hours = []
            days_of_week = []
            
            # Extract from edge data if available
            for etype in g.canonical_etypes:
                src, _, dst = etype
                if src == node_type or dst == node_type:
                    if 'timestamp' in g.edges[etype].data:
                        edge_timestamps = g.edges[etype].data['timestamp'].numpy()
                        if len(edge_timestamps) > 0:
                            timestamps.extend(edge_timestamps.tolist())
                            # Extract hour and day of week if available
                            if 'hour_of_day' in g.edges[etype].data:
                                hours.extend(g.edges[etype].data['hour_of_day'].numpy().tolist())
                            if 'day_of_week' in g.edges[etype].data:
                                days_of_week.extend(g.edges[etype].data['day_of_week'].numpy().tolist())
            
            if len(timestamps) > 0:
                timestamps = np.array(timestamps)
                features['first_transaction_timestamp'][i] = np.min(timestamps)
                features['last_transaction_timestamp'][i] = np.max(timestamps)
                features['account_age_days'][i] = (np.max(timestamps) - np.min(timestamps)) / (24 * 3600) if len(timestamps) > 1 else 0
                features['time_since_last_transaction'][i] = (current_timestamp - np.max(timestamps)) / (24 * 3600) if current_timestamp > 0 else 0
                
                # Transactions in time windows
                last_24h = current_timestamp - 24 * 3600
                last_7d = current_timestamp - 7 * 24 * 3600
                last_30d = current_timestamp - 30 * 24 * 3600
                
                features['transactions_last_24h'][i] = np.sum(timestamps >= last_24h)
                features['transactions_last_7d'][i] = np.sum(timestamps >= last_7d)
                features['transactions_last_30d'][i] = np.sum(timestamps >= last_30d)
                
                # Mode hour and day of week
                if len(hours) > 0:
                    features['hour_of_day_mode'][i] = stats.mode(hours)[0][0] if len(hours) > 0 else 0
                if len(days_of_week) > 0:
                    features['day_of_week_mode'][i] = stats.mode(days_of_week)[0][0] if len(days_of_week) > 0 else 0
                
                # Transaction time variance
                if len(timestamps) > 1:
                    features['transaction_time_variance'][i] = np.var(timestamps)
        
        return features
    
    def extract_behavioral_features(self, g: dgl.DGLHeteroGraph, node_type: str = 'user',
                                    transaction_df: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """
        Extract behavioral features (8 features)
        
        38. round_amount_ratio
        39. threshold_amount_ratio
        40. transaction_mode_diversity
        41. failed_transaction_ratio
        42. reversal_ratio
        43. cross_border_ratio
        44. high_risk_country_ratio
        45. burst_score
        """
        logger.info("Extracting behavioral features")
        
        node_ids = g.nodes(node_type).numpy()
        n_nodes = len(node_ids)
        
        features = {
            'round_amount_ratio': np.zeros(n_nodes),
            'threshold_amount_ratio': np.zeros(n_nodes),
            'transaction_mode_diversity': np.zeros(n_nodes),
            'failed_transaction_ratio': np.zeros(n_nodes),
            'reversal_ratio': np.zeros(n_nodes),
            'cross_border_ratio': np.zeros(n_nodes),
            'high_risk_country_ratio': np.zeros(n_nodes),
            'burst_score': np.zeros(n_nodes)
        }
        
        # Extract from edge data
        for i, node_id in enumerate(node_ids):
            amounts = []
            transaction_modes = []
            failed_count = 0
            reversal_count = 0
            cross_border_count = 0
            high_risk_country_count = 0
            total_transactions = 0
            
            for etype in g.canonical_etypes:
                src, _, dst = etype
                if src == node_type or dst == node_type:
                    if 'amount' in g.edges[etype].data:
                        edge_amounts = g.edges[etype].data['amount'].numpy()
                        amounts.extend(edge_amounts.tolist())
                        total_transactions += len(edge_amounts)
                    
                    if 'transaction_mode' in g.edges[etype].data:
                        modes = g.edges[etype].data['transaction_mode'].numpy()
                        transaction_modes.extend(modes.tolist())
                    
                    # These would need to be in edge data
                    # For now, we'll compute what we can
                    if 'is_failed' in g.edges[etype].data:
                        failed_count += np.sum(g.edges[etype].data['is_failed'].numpy())
                    if 'is_reversal' in g.edges[etype].data:
                        reversal_count += np.sum(g.edges[etype].data['is_reversal'].numpy())
                    if 'is_cross_border' in g.edges[etype].data:
                        cross_border_count += np.sum(g.edges[etype].data['is_cross_border'].numpy())
                    if 'is_high_risk_country' in g.edges[etype].data:
                        high_risk_country_count += np.sum(g.edges[etype].data['is_high_risk_country'].numpy())
            
            if len(amounts) > 0:
                amounts = np.array(amounts)
                # Round amount ratio (amounts ending in .00, .50, etc.)
                round_amounts = np.sum((amounts % 1 == 0) | (amounts % 0.5 == 0))
                features['round_amount_ratio'][i] = round_amounts / len(amounts)
                
                # Threshold amount ratio (common thresholds like 100, 1000, etc.)
                thresholds = [100, 500, 1000, 5000, 10000]
                threshold_count = np.sum([np.sum(amounts == t) for t in thresholds])
                features['threshold_amount_ratio'][i] = threshold_count / len(amounts)
            
            if len(transaction_modes) > 0:
                unique_modes = len(np.unique(transaction_modes))
                features['transaction_mode_diversity'][i] = unique_modes / len(transaction_modes) if len(transaction_modes) > 0 else 0
            
            if total_transactions > 0:
                features['failed_transaction_ratio'][i] = failed_count / total_transactions
                features['reversal_ratio'][i] = reversal_count / total_transactions
                features['cross_border_ratio'][i] = cross_border_count / total_transactions
                features['high_risk_country_ratio'][i] = high_risk_country_count / total_transactions
            
            # Burst score (transactions in short time windows)
            # This would require timestamp data
            features['burst_score'][i] = 0.0  # Placeholder
        
        return features
    
    def extract_fraud_propagation_features(self, g: dgl.DGLHeteroGraph, node_type: str = 'user',
                                           transaction_df: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """
        Extract fraud propagation features (5 features)
        
        46. connected_to_fraud_count
        47. fraud_propagation_score
        48. distance_to_nearest_fraud
        49. common_neighbors_with_frauds
        50. fraud_cluster_membership
        """
        logger.info("Extracting fraud propagation features")
        
        node_ids = g.nodes(node_type).numpy()
        n_nodes = len(node_ids)
        
        features = {
            'connected_to_fraud_count': np.zeros(n_nodes),
            'fraud_propagation_score': np.zeros(n_nodes),
            'distance_to_nearest_fraud': np.zeros(n_nodes),
            'common_neighbors_with_frauds': np.zeros(n_nodes),
            'fraud_cluster_membership': np.zeros(n_nodes)
        }
        
        # Get fraud labels if available
        fraud_nodes = set()
        if 'is_fraud' in g.nodes[node_type].data:
            fraud_mask = g.nodes[node_type].data['is_fraud'].numpy()
            fraud_nodes = set(node_ids[fraud_mask == 1])
        
        if len(fraud_nodes) == 0:
            return features
        
        # Build NetworkX graph for path finding
        try:
            nx_g = nx.DiGraph()
            for node in node_ids:
                nx_g.add_node(node)
            
            for etype in g.canonical_etypes:
                src, _, dst = etype
                if src == node_type and dst == node_type:
                    src_nodes, dst_nodes = g.edges(etype=etype)
                    for s, d in zip(src_nodes.tolist(), dst_nodes.tolist()):
                        nx_g.add_edge(s, d)
            
            for i, node_id in enumerate(node_ids):
                if node_id in fraud_nodes:
                    features['fraud_cluster_membership'][i] = 1.0
                    continue
                
                # Count direct connections to fraud nodes
                fraud_neighbors = 0
                for fraud_node in fraud_nodes:
                    if nx_g.has_edge(node_id, fraud_node) or nx_g.has_edge(fraud_node, node_id):
                        fraud_neighbors += 1
                
                features['connected_to_fraud_count'][i] = fraud_neighbors
                
                # Distance to nearest fraud
                min_distance = float('inf')
                for fraud_node in fraud_nodes:
                    try:
                        if nx.has_path(nx_g, node_id, fraud_node):
                            distance = nx.shortest_path_length(nx_g, node_id, fraud_node)
                            min_distance = min(min_distance, distance)
                        if nx.has_path(nx_g, fraud_node, node_id):
                            distance = nx.shortest_path_length(nx_g, fraud_node, node_id)
                            min_distance = min(min_distance, distance)
                    except:
                        pass
                
                features['distance_to_nearest_fraud'][i] = min_distance if min_distance != float('inf') else -1
                
                # Common neighbors with frauds
                node_neighbors = set(nx_g.neighbors(node_id)) | set(nx_g.predecessors(node_id))
                common_neighbors = 0
                for fraud_node in fraud_nodes:
                    fraud_neighbors = set(nx_g.neighbors(fraud_node)) | set(nx_g.predecessors(fraud_node))
                    common_neighbors += len(node_neighbors & fraud_neighbors)
                
                features['common_neighbors_with_frauds'][i] = common_neighbors
                
                # Fraud propagation score (weighted by distance)
                propagation_score = 0.0
                for fraud_node in fraud_nodes:
                    try:
                        if nx.has_path(nx_g, fraud_node, node_id):
                            distance = nx.shortest_path_length(nx_g, fraud_node, node_id)
                            propagation_score += 1.0 / (distance + 1)
                    except:
                        pass
                
                features['fraud_propagation_score'][i] = propagation_score
                
        except Exception as e:
            logger.warning(f"Fraud propagation calculation failed: {e}")
        
        return features


class EdgeFeatureExtractor:
    """Extracts transaction-level edge features"""
    
    def __init__(self):
        """Initialize edge feature extractor"""
        logger.info("Initialized EdgeFeatureExtractor")
    
    def extract_all_edge_features(self, g: dgl.DGLHeteroGraph, edge_type: str = 'transaction',
                                  transaction_df: Optional[pd.DataFrame] = None) -> Dict[str, th.Tensor]:
        """
        Extract all transaction-level edge features (12 features)
        
        1. amount
        2. timestamp
        3. hour_of_day
        4. day_of_week
        5. is_weekend
        6. transaction_mode
        7. time_since_last_between_users
        8. amount_percentile_sender
        9. amount_percentile_receiver
        10. is_reciprocal
        11. reciprocal_time_gap
        12. geographic_distance
        """
        logger.info(f"Extracting edge features for edge type: {edge_type}")
        
        # Find the edge type
        etype = None
        for canonical_etype in g.canonical_etypes:
            if edge_type in str(canonical_etype) or edge_type == canonical_etype[1]:
                etype = canonical_etype
                break
        
        if etype is None:
            # Try to use the first edge type if available
            if len(g.canonical_etypes) > 0:
                etype = g.canonical_etypes[0]
                logger.info(f"Using first available edge type: {etype}")
            else:
                logger.warning(f"Edge type {edge_type} not found in graph")
                return {}
        
        n_edges = g.number_of_edges(etype)
        src_nodes, dst_nodes = g.edges(etype=etype)
        
        features = {}
        
        # 1. amount
        if 'amount' in g.edges[etype].data:
            features['amount'] = g.edges[etype].data['amount']
        else:
            features['amount'] = th.zeros(n_edges)
        
        # 2. timestamp
        if 'timestamp' in g.edges[etype].data:
            features['timestamp'] = g.edges[etype].data['timestamp']
        else:
            features['timestamp'] = th.zeros(n_edges)
        
        # 3. hour_of_day
        if 'hour_of_day' in g.edges[etype].data:
            features['hour_of_day'] = g.edges[etype].data['hour_of_day']
        else:
            # Extract from timestamp if available
            if 'timestamp' in features:
                timestamps = features['timestamp'].numpy()
                hours = np.array([datetime.fromtimestamp(ts).hour if ts > 0 else 0 for ts in timestamps])
                features['hour_of_day'] = th.from_numpy(hours.astype(np.float32))
            else:
                features['hour_of_day'] = th.zeros(n_edges)
        
        # 4. day_of_week
        if 'day_of_week' in g.edges[etype].data:
            features['day_of_week'] = g.edges[etype].data['day_of_week']
        else:
            # Extract from timestamp if available
            if 'timestamp' in features:
                timestamps = features['timestamp'].numpy()
                days = np.array([datetime.fromtimestamp(ts).weekday() if ts > 0 else 0 for ts in timestamps])
                features['day_of_week'] = th.from_numpy(days.astype(np.float32))
            else:
                features['day_of_week'] = th.zeros(n_edges)
        
        # 5. is_weekend
        if 'day_of_week' in features:
            days = features['day_of_week'].numpy()
            is_weekend = (days == 5) | (days == 6)  # Saturday or Sunday
            features['is_weekend'] = th.from_numpy(is_weekend.astype(np.float32))
        else:
            features['is_weekend'] = th.zeros(n_edges)
        
        # 6. transaction_mode
        if 'transaction_mode' in g.edges[etype].data:
            features['transaction_mode'] = g.edges[etype].data['transaction_mode']
        else:
            features['transaction_mode'] = th.zeros(n_edges)
        
        # 7. time_since_last_between_users
        features['time_since_last_between_users'] = self._compute_time_since_last(g, etype, src_nodes, dst_nodes, features.get('timestamp'))
        
        # 8. amount_percentile_sender
        features['amount_percentile_sender'] = self._compute_amount_percentile_sender(g, etype, src_nodes, features.get('amount'))
        
        # 9. amount_percentile_receiver
        features['amount_percentile_receiver'] = self._compute_amount_percentile_receiver(g, etype, dst_nodes, features.get('amount'))
        
        # 10. is_reciprocal
        features['is_reciprocal'] = self._compute_is_reciprocal(g, etype, src_nodes, dst_nodes)
        
        # 11. reciprocal_time_gap
        features['reciprocal_time_gap'] = self._compute_reciprocal_time_gap(g, etype, src_nodes, dst_nodes, features.get('timestamp'))
        
        # 12. geographic_distance
        if 'geographic_distance' in g.edges[etype].data:
            features['geographic_distance'] = g.edges[etype].data['geographic_distance']
        else:
            features['geographic_distance'] = th.zeros(n_edges)
        
        logger.info(f"Extracted {len(features)} edge features for {n_edges} edges")
        return features
    
    def _compute_time_since_last(self, g: dgl.DGLHeteroGraph, etype: Tuple, src_nodes: th.Tensor,
                                 dst_nodes: th.Tensor, timestamps: Optional[th.Tensor]) -> th.Tensor:
        """Compute time since last transaction between each user pair"""
        n_edges = len(src_nodes)
        if timestamps is None:
            return th.zeros(n_edges)
        
        time_since = th.zeros(n_edges)
        user_pair_last_time = {}
        
        timestamps_np = timestamps.numpy()
        src_np = src_nodes.numpy()
        dst_np = dst_nodes.numpy()
        
        for i in range(n_edges):
            src, dst = src_np[i], dst_np[i]
            pair = (min(src, dst), max(src, dst))
            current_time = timestamps_np[i]
            
            if pair in user_pair_last_time:
                time_since[i] = current_time - user_pair_last_time[pair]
            else:
                time_since[i] = 0.0
            
            user_pair_last_time[pair] = current_time
        
        return time_since
    
    def _compute_amount_percentile_sender(self, g: dgl.DGLHeteroGraph, etype: Tuple, src_nodes: th.Tensor,
                                         amounts: Optional[th.Tensor]) -> th.Tensor:
        """Compute amount percentile for each sender"""
        n_edges = len(src_nodes)
        if amounts is None:
            return th.zeros(n_edges)
        
        percentiles = th.zeros(n_edges)
        sender_amounts = defaultdict(list)
        
        src_np = src_nodes.numpy()
        amounts_np = amounts.numpy()
        
        # Collect amounts per sender
        for i in range(n_edges):
            sender_amounts[src_np[i]].append((i, amounts_np[i]))
        
        # Compute percentiles
        for sender, amount_list in sender_amounts.items():
            if len(amount_list) > 1:
                sorted_amounts = sorted([a for _, a in amount_list])
                for idx, amount in amount_list:
                    percentile = sorted_amounts.index(amount) / len(sorted_amounts)
                    percentiles[idx] = percentile
        
        return percentiles
    
    def _compute_amount_percentile_receiver(self, g: dgl.DGLHeteroGraph, etype: Tuple, dst_nodes: th.Tensor,
                                            amounts: Optional[th.Tensor]) -> th.Tensor:
        """Compute amount percentile for each receiver"""
        n_edges = len(dst_nodes)
        if amounts is None:
            return th.zeros(n_edges)
        
        percentiles = th.zeros(n_edges)
        receiver_amounts = defaultdict(list)
        
        dst_np = dst_nodes.numpy()
        amounts_np = amounts.numpy()
        
        # Collect amounts per receiver
        for i in range(n_edges):
            receiver_amounts[dst_np[i]].append((i, amounts_np[i]))
        
        # Compute percentiles
        for receiver, amount_list in receiver_amounts.items():
            if len(amount_list) > 1:
                sorted_amounts = sorted([a for _, a in amount_list])
                for idx, amount in amount_list:
                    percentile = sorted_amounts.index(amount) / len(sorted_amounts)
                    percentiles[idx] = percentile
        
        return percentiles
    
    def _compute_is_reciprocal(self, g: dgl.DGLHeteroGraph, etype: Tuple, src_nodes: th.Tensor,
                               dst_nodes: th.Tensor) -> th.Tensor:
        """Check if transaction is reciprocal (bidirectional)"""
        n_edges = len(src_nodes)
        is_reciprocal = th.zeros(n_edges)
        
        # Create set of edges
        edge_set = set()
        src_np = src_nodes.numpy()
        dst_np = dst_nodes.numpy()
        
        for i in range(n_edges):
            edge_set.add((src_np[i], dst_np[i]))
        
        # Check for reciprocal edges
        for i in range(n_edges):
            if (dst_np[i], src_np[i]) in edge_set:
                is_reciprocal[i] = 1.0
        
        return is_reciprocal
    
    def _compute_reciprocal_time_gap(self, g: dgl.DGLHeteroGraph, etype: Tuple, src_nodes: th.Tensor,
                                     dst_nodes: th.Tensor, timestamps: Optional[th.Tensor]) -> th.Tensor:
        """Compute time gap for reciprocal transactions"""
        n_edges = len(src_nodes)
        if timestamps is None:
            return th.zeros(n_edges)
        
        time_gaps = th.zeros(n_edges)
        edge_times = {}
        
        src_np = src_nodes.numpy()
        dst_np = dst_nodes.numpy()
        timestamps_np = timestamps.numpy()
        
        # Store edge times
        for i in range(n_edges):
            edge_times[(src_np[i], dst_np[i])] = timestamps_np[i]
        
        # Find reciprocal time gaps
        for i in range(n_edges):
            reciprocal_edge = (dst_np[i], src_np[i])
            if reciprocal_edge in edge_times:
                time_gaps[i] = abs(timestamps_np[i] - edge_times[reciprocal_edge])
        
        return time_gaps
