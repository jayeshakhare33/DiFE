"""
Distributed Graph Construction
Converts transaction data into graph format (nodes=users, edges=transactions)
"""

import os
import pandas as pd
import numpy as np
import dgl
import torch as th
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds heterogeneous graphs from transaction data with distributed support"""
    
    def __init__(self, num_workers: int = None, chunk_size: int = 10000):
        """
        Initialize graph builder
        
        Args:
            num_workers: Number of parallel workers (default: CPU count)
            chunk_size: Size of chunks for parallel processing
        """
        self.num_workers = num_workers or mp.cpu_count()
        self.chunk_size = chunk_size
        logger.info(f"Initialized GraphBuilder with {self.num_workers} workers")
    
    def load_transaction_data(self, transaction_path: str, identity_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load transaction and identity data
        
        Args:
            transaction_path: Path to transaction CSV
            identity_path: Path to identity CSV (optional)
            
        Returns:
            Merged dataframe
        """
        logger.info(f"Loading transaction data from {transaction_path}")
        transaction_df = pd.read_csv(transaction_path)
        
        if identity_path and os.path.exists(identity_path):
            logger.info(f"Loading identity data from {identity_path}")
            identity_df = pd.read_csv(identity_path)
            # Merge on TransactionID
            full_df = transaction_df.merge(identity_df, on='TransactionID', how='left')
        else:
            full_df = transaction_df
        
        logger.info(f"Loaded {len(full_df)} transactions")
        return full_df
    
    def extract_identity_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Extract identity columns that will become entity nodes
        
        Args:
            df: Transaction dataframe
            
        Returns:
            List of identity column names
        """
        # Card columns
        card_cols = [col for col in df.columns if col.startswith('card')]
        
        # Address columns
        addr_cols = [col for col in df.columns if col.startswith('addr')]
        
        # Email columns
        email_cols = [col for col in df.columns if 'email' in col.lower()]
        
        # Product column
        product_cols = ['ProductCD'] if 'ProductCD' in df.columns else []
        
        # Identity columns (id_01 through id_38)
        id_cols = [col for col in df.columns if col.startswith('id_')]
        
        # Device columns
        device_cols = [col for col in df.columns if 'Device' in col]
        
        identity_cols = card_cols + addr_cols + email_cols + product_cols + id_cols + device_cols
        
        logger.info(f"Found {len(identity_cols)} identity columns")
        return identity_cols
    
    def create_edgelists_parallel(self, df: pd.DataFrame, identity_cols: List[str], 
                                  output_dir: str, target_col: str = 'TransactionID') -> List[str]:
        """
        Create edge lists in parallel for each identity column
        
        Args:
            df: Transaction dataframe
            identity_cols: List of identity column names
            output_dir: Output directory for edge lists
            target_col: Target node column name
            
        Returns:
            List of created edge list file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        def create_edgelist_chunk(args):
            """Helper function for parallel processing"""
            col, chunk_df, output_dir, target_col = args
            edge_list = []
            file_path = os.path.join(output_dir, f'relation_{col}_edgelist.csv')
            
            # Filter out NaN values
            valid_df = chunk_df[[target_col, col]].dropna()
            
            for _, row in valid_df.iterrows():
                edge_list.append((str(row[target_col]), str(row[col])))
            
            # Write to CSV
            if edge_list:
                edge_df = pd.DataFrame(edge_list, columns=[target_col, col])
                edge_df.to_csv(file_path, index=False, header=True)
                return file_path
            return None
        
        # Prepare arguments for parallel processing
        args_list = []
        for col in identity_cols:
            if col in df.columns:
                # Split dataframe into chunks for parallel processing
                for i in range(0, len(df), self.chunk_size):
                    chunk = df.iloc[i:i+self.chunk_size]
                    args_list.append((col, chunk, output_dir, target_col))
        
        # Process in parallel
        edge_files = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(create_edgelist_chunk, args): args for args in args_list}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    edge_files.append(result)
        
        # Merge chunks for each column
        merged_files = []
        for col in identity_cols:
            chunk_files = [f for f in edge_files if f'relation_{col}_edgelist.csv' in f]
            if chunk_files:
                # Combine chunks
                combined_df = pd.concat([pd.read_csv(f) for f in chunk_files], ignore_index=True)
                combined_df = combined_df.drop_duplicates()
                final_path = os.path.join(output_dir, f'relation_{col}_edgelist.csv')
                combined_df.to_csv(final_path, index=False, header=True)
                merged_files.append(final_path)
                # Clean up chunk files
                for f in chunk_files:
                    if f != final_path:
                        os.remove(f)
        
        logger.info(f"Created {len(merged_files)} edge list files")
        return merged_files
    
    def build_heterograph(self, data_dir: str, target_ntype: str = 'TransactionID',
                         feature_file: str = 'features.csv') -> Tuple[dgl.DGLHeteroGraph, Dict, Dict]:
        """
        Build heterogeneous graph from edge lists and features
        
        Args:
            data_dir: Directory containing edge lists and features
            target_ntype: Target node type name
            feature_file: Feature file name
            
        Returns:
            Tuple of (graph, target_id_to_node, id_to_node)
        """
        logger.info("Building heterogeneous graph")
        
        # Get all edge list files
        edge_files = [f for f in os.listdir(data_dir) if f.startswith('relation_') and f.endswith('_edgelist.csv')]
        logger.info(f"Found {len(edge_files)} edge list files")
        
        edgelists = {}
        id_to_node = {}
        
        # Parse each edge list
        for edge_file in edge_files:
            edge_path = os.path.join(data_dir, edge_file)
            edge_list, rev_edge_list, id_to_node, src, dst = self._parse_edgelist(
                edge_path, id_to_node, header=True, target_ntype=target_ntype
            )
            
            if src == target_ntype:
                src = 'target'
            if dst == target_ntype:
                dst = 'target'
            
            if src == 'target' and dst == 'target':
                logger.info("Skipping self-loop, will add later")
            else:
                edgelists[(src, src + '<>' + dst, dst)] = edge_list
                edgelists[(dst, dst + '<>' + src, src)] = rev_edge_list
        
        # Add self-loops for target nodes
        if 'target' in id_to_node:
            edgelists[('target', 'self_relation', 'target')] = [
                (t, t) for t in id_to_node['target'].values()
            ]
        
        # Build graph
        g = dgl.heterograph(edgelists)
        logger.info(f"Built graph with node types: {g.ntypes}, edge types: {len(g.canonical_etypes)}")
        
        # Load features
        feature_path = os.path.join(data_dir, feature_file)
        if os.path.exists(feature_path):
            features, _ = self._get_features(id_to_node.get('target', {}), feature_path)
            g.nodes['target'].data['features'] = th.from_numpy(features)
            logger.info(f"Loaded features with shape: {features.shape}")
        
        target_id_to_node = id_to_node.get('target', {})
        id_to_node['target'] = target_id_to_node
        
        return g, target_id_to_node, id_to_node
    
    def _parse_edgelist(self, edges: str, id_to_node: Dict, header: bool = False,
                       source_type: str = 'user', sink_type: str = 'user',
                       target_ntype: str = 'TransactionID') -> Tuple:
        """Parse edge list file"""
        edge_list = []
        rev_edge_list = []
        source_pointer, sink_pointer = 0, 0
        
        with open(edges, "r") as fh:
            for i, line in enumerate(fh):
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                    
                source, sink = parts[0], parts[1]
                
                if i == 0:
                    if header:
                        source_type, sink_type = source, sink
                    if source_type in id_to_node:
                        source_pointer = max(id_to_node[source_type].values()) + 1
                    if sink_type in id_to_node:
                        sink_pointer = max(id_to_node[sink_type].values()) + 1
                    continue
                
                source_node, id_to_node, source_pointer = self._get_node_idx(
                    id_to_node, source_type, source, source_pointer
                )
                
                if source_type == sink_type:
                    sink_node, id_to_node, source_pointer = self._get_node_idx(
                        id_to_node, sink_type, sink, source_pointer
                    )
                else:
                    sink_node, id_to_node, sink_pointer = self._get_node_idx(
                        id_to_node, sink_type, sink, sink_pointer
                    )
                
                edge_list.append((source_node, sink_node))
                rev_edge_list.append((sink_node, source_node))
        
        return edge_list, rev_edge_list, id_to_node, source_type, sink_type
    
    def _get_node_idx(self, id_to_node: Dict, node_type: str, node_id: str, ptr: int) -> Tuple:
        """Get or create node index"""
        if node_type not in id_to_node:
            id_to_node[node_type] = {}
        
        if node_id not in id_to_node[node_type]:
            id_to_node[node_type][node_id] = ptr
            ptr += 1
        
        return id_to_node[node_type][node_id], id_to_node, ptr
    
    def build_user_transaction_graph(self, df: pd.DataFrame, 
                                     sender_col: str = 'sender_id',
                                     receiver_col: str = 'receiver_id',
                                     amount_col: str = 'TransactionAmt',
                                     timestamp_col: str = 'TransactionDT',
                                     fraud_col: Optional[str] = 'isFraud') -> Tuple[dgl.DGLHeteroGraph, Dict, pd.DataFrame]:
        """
        Build user-to-user graph with transactions as edges
        
        Args:
            df: Transaction dataframe
            sender_col: Column name for sender/user ID
            receiver_col: Column name for receiver/user ID
            amount_col: Column name for transaction amount
            timestamp_col: Column name for transaction timestamp
            fraud_col: Column name for fraud label (optional)
            
        Returns:
            Tuple of (graph, user_id_to_node, transaction_df)
        """
        logger.info("Building user-to-user transaction graph")
        
        # If sender/receiver columns don't exist, create them from card1 or other identifier
        if sender_col not in df.columns:
            # Use card1 as sender identifier, or create from TransactionID
            if 'card1' in df.columns:
                df[sender_col] = df['card1'].fillna(df['TransactionID']).astype(str)
            else:
                df[sender_col] = df['TransactionID'].astype(str)
        
        if receiver_col not in df.columns:
            # Use card2 as receiver identifier, or create from TransactionID
            if 'card2' in df.columns:
                df[receiver_col] = df['card2'].fillna(df['TransactionID']).astype(str)
            else:
                # Create receiver as a variation of sender for demo
                df[receiver_col] = (df['TransactionID'] + 1000000).astype(str)
        
        # Create user ID mapping
        all_users = set(df[sender_col].unique()) | set(df[receiver_col].unique())
        user_id_to_node = {user_id: idx for idx, user_id in enumerate(sorted(all_users))}
        logger.info(f"Found {len(user_id_to_node)} unique users")
        
        # Create edge list with transaction data
        edge_list = []
        edge_features = {
            'amount': [],
            'timestamp': [],
            'hour_of_day': [],
            'day_of_week': [],
            'transaction_mode': [],
            'geographic_distance': [],
            'status': [],
            'is_cross_border': []
        }
        
        # Process transactions
        for idx, row in df.iterrows():
            sender_id = str(row[sender_col])
            receiver_id = str(row[receiver_col])
            
            if pd.isna(sender_id) or pd.isna(receiver_id):
                continue
            
            sender_node = user_id_to_node[sender_id]
            receiver_node = user_id_to_node[receiver_id]
            
            edge_list.append((sender_node, receiver_node))
            
            # Extract edge features
            amount = float(row[amount_col]) if amount_col in df.columns and not pd.isna(row[amount_col]) else 0.0
            edge_features['amount'].append(amount)
            
            # Timestamp
            if timestamp_col in df.columns and not pd.isna(row[timestamp_col]):
                timestamp = float(row[timestamp_col])
                edge_features['timestamp'].append(timestamp)
                
                # Extract hour and day of week from timestamp
                # Handle both epoch seconds and epoch milliseconds
                try:
                    from datetime import datetime
                    # Check if timestamp is in milliseconds (> 1e10) or seconds
                    if timestamp > 1e10:
                        dt = datetime.fromtimestamp(timestamp / 1000.0)
                    else:
                        dt = datetime.fromtimestamp(timestamp)
                    edge_features['hour_of_day'].append(float(dt.hour))
                    edge_features['day_of_week'].append(float(dt.weekday()))
                except:
                    edge_features['hour_of_day'].append(0.0)
                    edge_features['day_of_week'].append(0.0)
            else:
                edge_features['timestamp'].append(0.0)
                edge_features['hour_of_day'].append(0.0)
                edge_features['day_of_week'].append(0.0)
            
            # Transaction mode (prefer 'mode' column, fallback to 'ProductCD')
            if 'mode' in df.columns and not pd.isna(row['mode']):
                # Encode mode as numeric (upi, wallet, card, net_banking, neft, imps)
                mode_map = {'upi': 0, 'wallet': 1, 'card': 2, 'net_banking': 3, 'neft': 4, 'imps': 5}
                mode_str = str(row['mode']).lower()
                edge_features['transaction_mode'].append(float(mode_map.get(mode_str, 0)))
            elif 'ProductCD' in df.columns and not pd.isna(row['ProductCD']):
                # Encode ProductCD as numeric (fallback for IEEE format)
                mode_map = {'W': 0, 'R': 1, 'C': 2, 'H': 3, 'S': 4}
                edge_features['transaction_mode'].append(float(mode_map.get(str(row['ProductCD']), 0)))
            else:
                edge_features['transaction_mode'].append(0.0)
            
            # Geographic distance (use geo_distance_km if available)
            if 'geo_distance_km' in df.columns and not pd.isna(row['geo_distance_km']):
                edge_features['geographic_distance'].append(float(row['geo_distance_km']))
            else:
                edge_features['geographic_distance'].append(0.0)
            
            # Transaction status (success=1, failed=0, reversed=-1)
            if 'status' in df.columns and not pd.isna(row['status']):
                status_str = str(row['status']).lower()
                if status_str == 'success':
                    edge_features['status'].append(1.0)
                elif status_str == 'failed':
                    edge_features['status'].append(0.0)
                elif status_str == 'reversed':
                    edge_features['status'].append(-1.0)
                else:
                    edge_features['status'].append(0.0)
            else:
                edge_features['status'].append(0.0)
            
            # Cross-border flag
            if 'is_cross_border' in df.columns and not pd.isna(row['is_cross_border']):
                is_cross = bool(row['is_cross_border'])
                edge_features['is_cross_border'].append(1.0 if is_cross else 0.0)
            else:
                edge_features['is_cross_border'].append(0.0)
        
        # Create graph
        # Build heterogeneous graph (user -> transaction -> user)
        # Create edge list for user-to-user transactions
        edgelists = {('user', 'transaction', 'user'): (th.tensor([e[0] for e in edge_list], dtype=th.long),
                                                       th.tensor([e[1] for e in edge_list], dtype=th.long))}
        g = dgl.heterograph(edgelists, num_nodes_dict={'user': len(user_id_to_node)})
        # Store user_id as string indices (node indices) instead of trying to convert to int
        # The actual user_id strings are preserved in user_id_to_node mapping
        g.nodes['user'].data['node_idx'] = th.tensor(list(range(len(user_id_to_node))), dtype=th.long)
        
        # Add edge features
        etype = ('user', 'transaction', 'user')
        g.edges[etype].data['amount'] = th.tensor(edge_features['amount'], dtype=th.float32)
        g.edges[etype].data['timestamp'] = th.tensor(edge_features['timestamp'], dtype=th.float32)
        g.edges[etype].data['hour_of_day'] = th.tensor(edge_features['hour_of_day'], dtype=th.float32)
        g.edges[etype].data['day_of_week'] = th.tensor(edge_features['day_of_week'], dtype=th.float32)
        g.edges[etype].data['transaction_mode'] = th.tensor(edge_features['transaction_mode'], dtype=th.float32)
        g.edges[etype].data['geographic_distance'] = th.tensor(edge_features['geographic_distance'], dtype=th.float32)
        g.edges[etype].data['status'] = th.tensor(edge_features['status'], dtype=th.float32)
        g.edges[etype].data['is_cross_border'] = th.tensor(edge_features['is_cross_border'], dtype=th.float32)
        
        # Add fraud labels to nodes if available
        if fraud_col and fraud_col in df.columns:
            # Aggregate fraud labels per user
            user_fraud = {}
            for idx, row in df.iterrows():
                sender_id = str(row[sender_col])
                receiver_id = str(row[receiver_col])
                is_fraud = int(row[fraud_col]) if not pd.isna(row[fraud_col]) else 0
                
                if sender_id in user_id_to_node:
                    if sender_id not in user_fraud:
                        user_fraud[sender_id] = 0
                    user_fraud[sender_id] = max(user_fraud[sender_id], is_fraud)
                
                if receiver_id in user_id_to_node:
                    if receiver_id not in user_fraud:
                        user_fraud[receiver_id] = 0
                    user_fraud[receiver_id] = max(user_fraud[receiver_id], is_fraud)
            
            fraud_labels = th.tensor([user_fraud.get(uid, 0) for uid in sorted(user_id_to_node.keys())], dtype=th.long)
            g.nodes['user'].data['is_fraud'] = fraud_labels
        
        logger.info(f"Built graph with {g.number_of_nodes()} users and {g.number_of_edges()} transactions")
        
        return g, user_id_to_node, df
    
    def _get_features(self, id_to_node: Dict, node_features: str) -> Tuple:
        """Load node features"""
        indices, features = [], []
        max_node = max(id_to_node.values()) if id_to_node and len(id_to_node) > 0 else -1
        
        with open(node_features, "r") as fh:
            for line in fh:
                node_feats = line.strip().split(",")
                if len(node_feats) < 2:
                    continue
                    
                node_id = node_feats[0]
                try:
                    feats = np.array(list(map(float, node_feats[1:])))
                    features.append(feats)
                    
                    if node_id not in id_to_node:
                        max_node += 1
                        id_to_node[node_id] = max_node
                    
                    indices.append(id_to_node[node_id])
                except (ValueError, IndexError):
                    continue
        
        if len(features) == 0:
            return np.array([]).astype('float32'), []
        
        features = np.array(features).astype('float32')
        features = features[np.argsort(indices), :]
        return features, []

