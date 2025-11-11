"""
Main entry point for Fraud Detection GNN System
Orchestrates graph construction, feature engineering, training, and inference
"""

import os
import yaml
import logging
import argparse
import torch
import dgl
import pandas as pd
from typing import Dict, Optional

from graph_processing import GraphBuilder, GraphPartitioner
from feature_engineering import FeatureExtractor, EdgeFeatureExtractor, GraphEmbeddings
from storage import FeatureStore
from gnn_training import DistributedTrainer, HeteroRGCN
from api import InferenceService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_graph(config: Dict) -> tuple:
    """Build graph from transaction data"""
    logger.info("Building graph from transaction data")
    
    # Initialize graph builder
    graph_config = config.get('graph_processing', {})
    builder = GraphBuilder(
        num_workers=graph_config.get('num_workers', 4),
        chunk_size=graph_config.get('chunk_size', 10000)
    )
    
    # Load data
    data_config = config.get('data', {})
    transaction_path = data_config.get('transaction_path')
    identity_path = data_config.get('identity_path')
    
    df = builder.load_transaction_data(transaction_path, identity_path)
    
    # Extract identity columns
    identity_cols = builder.extract_identity_columns(df)
    
    # Create edge lists
    output_dir = data_config.get('output_dir', './data')
    edge_files = builder.create_edgelists_parallel(df, identity_cols, output_dir)
    
    # Build graph
    g, target_id_to_node, id_to_node = builder.build_heterograph(
        data_dir=output_dir,
        target_ntype='TransactionID',
        feature_file=data_config.get('feature_file', 'features.csv')
    )
    
    logger.info(f"Built graph with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")
    return g, target_id_to_node, id_to_node


def extract_features(g: dgl.DGLHeteroGraph, config: Dict, feature_store: FeatureStore, 
                    transaction_df: Optional[pd.DataFrame] = None):
    """Extract graph-based features"""
    logger.info("Extracting graph-based features")
    
    # Initialize feature extractors
    node_extractor = FeatureExtractor(transaction_df=transaction_df)
    edge_extractor = EdgeFeatureExtractor()
    
    # Extract node features (user-level)
    node_type = 'user' if 'user' in g.ntypes else 'target'
    features_df = node_extractor.extract_all_features(g, node_type=node_type, transaction_df=transaction_df)
    
    # Extract edge features (transaction-level)
    # Find transaction edge type
    edge_type = None
    for etype in g.canonical_etypes:
        if 'transaction' in str(etype).lower():
            edge_type = etype[1]  # Use the edge type name
            break
    
    if edge_type is None and len(g.canonical_etypes) > 0:
        edge_type = g.canonical_etypes[0][1]  # Use first available edge type
    
    edge_features = {}
    if edge_type:
        edge_features = edge_extractor.extract_all_edge_features(g, edge_type=edge_type, transaction_df=transaction_df)
    
    # Save to feature store
    feature_store.save_features(features_df, 'node_features')
    logger.info(f"Extracted and saved {len(features_df.columns)} node features for {len(features_df)} nodes")
    
    # Save edge features to disk
    if edge_features:
        # Convert edge features dict to DataFrame
        # Get the number of edges from the first tensor
        n_edges = len(list(edge_features.values())[0]) if edge_features else 0
        edge_features_dict = {}
        for name, tensor in edge_features.items():
            if isinstance(tensor, torch.Tensor):
                edge_features_dict[name] = tensor.cpu().numpy()
            else:
                edge_features_dict[name] = tensor
        
        edge_features_df = pd.DataFrame(edge_features_dict)
        feature_store.save_features(edge_features_df, 'edge_features')
        logger.info(f"Extracted and saved {len(edge_features_df.columns)} edge features for {len(edge_features_df)} edges")
    else:
        logger.info("No edge features extracted")
    
    return features_df, edge_features


def train_model(g: dgl.DGLHeteroGraph, config: Dict):
    """Train GNN model"""
    logger.info("Training GNN model")
    
    # Get node type
    node_type = 'user' if 'user' in g.ntypes else 'target'
    
    # Get features and labels
    if 'features' in g.ndata:
        features = g.ndata['features']
    else:
        # Load features from feature store
        storage_config = config.get('storage', {})
        feature_store = FeatureStore(
            backend_type=storage_config.get('backend', 'csv'),
            base_dir=storage_config.get('base_dir', './data/features')
        )
        features_df = feature_store.load_features('node_features')
        features = torch.tensor(features_df.values, dtype=torch.float32)
    
    # Get labels
    if 'is_fraud' in g.ndata:
        labels = g.ndata['is_fraud']
    else:
        # Load labels from file
        data_config = config.get('data', {})
        labels_path = os.path.join(data_config.get('output_dir', './data'), 
                                   data_config.get('labels_file', 'tags.csv'))
        
        if os.path.exists(labels_path):
            labels_df = pd.read_csv(labels_path)
            labels = torch.tensor(labels_df['isFraud'].values, dtype=torch.long)
        else:
            logger.warning(f"Labels file not found: {labels_path}, using dummy labels")
            labels = torch.zeros(g.number_of_nodes(), dtype=torch.long)
    
    # Create train/test masks
    n_nodes = g.number_of_nodes()
    train_size = int(0.8 * n_nodes)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:train_size] = True
    test_mask = ~train_mask
    
    # Initialize distributed trainer
    dist_config = config.get('distributed', {})
    trainer = DistributedTrainer(
        world_size=dist_config.get('world_size', 1),
        backend=dist_config.get('backend', 'gloo')
    )
    
    # Train model
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    trainer.train(
        g=g,
        features=features,
        labels=labels,
        train_mask=train_mask,
        test_mask=test_mask,
        model_config=model_config,
        training_config=training_config
    )
    
    logger.info("Model training completed")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fraud Detection GNN System')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['build', 'train', 'infer', 'all'],
                       default='all', help='Operation mode')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.mode in ['build', 'all']:
        # Build graph
        data_config = config.get('data', {})
        graph_config = config.get('graph_processing', {})
        builder = GraphBuilder(
            num_workers=graph_config.get('num_workers', 4),
            chunk_size=graph_config.get('chunk_size', 10000)
        )
        
        # Load transaction data
        transaction_path = data_config.get('transaction_path')
        identity_path = data_config.get('identity_path')
        df = builder.load_transaction_data(transaction_path, identity_path)
        
        # Get column mappings from config or use defaults
        sender_col = data_config.get('sender_col', 'sender_id')
        receiver_col = data_config.get('receiver_col', 'receiver_id')
        amount_col = data_config.get('amount_col', 'amount')
        timestamp_col = data_config.get('timestamp_col', 'timestamp')
        fraud_col = data_config.get('fraud_col', 'is_fraud_txn')
        
        # Build user-to-user graph with transactions as edges
        g, user_id_to_node, df = builder.build_user_transaction_graph(
            df,
            sender_col=sender_col if sender_col in df.columns else None,
            receiver_col=receiver_col if receiver_col in df.columns else None,
            amount_col=amount_col,
            timestamp_col=timestamp_col,
            fraud_col=fraud_col if fraud_col in df.columns else None
        )
        
        # Initialize feature store
        storage_config = config.get('storage', {})
        feature_store = FeatureStore(
            backend_type=storage_config.get('backend', 'csv'),
            base_dir=storage_config.get('base_dir', './data/features')
        )
        
        # Extract features
        node_features_df, edge_features = extract_features(g, config, feature_store, transaction_df=df)
    
    if args.mode in ['train', 'all']:
        # Load graph (if not already built)
        if 'g' not in locals():
            data_config = config.get('data', {})
            graph_config = config.get('graph_processing', {})
            builder = GraphBuilder(
                num_workers=graph_config.get('num_workers', 4),
                chunk_size=graph_config.get('chunk_size', 10000)
            )
            
            # Load transaction data
            transaction_path = data_config.get('transaction_path')
            identity_path = data_config.get('identity_path')
            df = builder.load_transaction_data(transaction_path, identity_path)
            
            # Get column mappings from config or use defaults
            sender_col = data_config.get('sender_col', 'sender_id')
            receiver_col = data_config.get('receiver_col', 'receiver_id')
            amount_col = data_config.get('amount_col', 'amount')
            timestamp_col = data_config.get('timestamp_col', 'timestamp')
            fraud_col = data_config.get('fraud_col', 'is_fraud_txn')
            
            # Build user-to-user graph
            g, _, _ = builder.build_user_transaction_graph(
                df,
                sender_col=sender_col if sender_col in df.columns else None,
                receiver_col=receiver_col if receiver_col in df.columns else None,
                amount_col=amount_col,
                timestamp_col=timestamp_col,
                fraud_col=fraud_col if fraud_col in df.columns else None
            )
        
        # Train model
        train_model(g, config)
    
    if args.mode == 'infer':
        logger.info("Inference mode - use API server for inference")
        logger.info("Start API server with: uvicorn api.app:app --host 0.0.0.0 --port 8000")
    
    logger.info("Process completed")


if __name__ == '__main__':
    main()

