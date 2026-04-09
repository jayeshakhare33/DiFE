"""
Distributed Feature Extraction Script

This script can be run independently or as part of a distributed system
to extract features in parallel across multiple workers.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Add project root to path
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

from gnn.distributed_feature_pipeline import DistributedFeaturePipeline
from gnn.feature_engineering import initialize_default_registry
from gnn.advanced_features import (
    GraphCentralityExtractor, PatternMatchingExtractor, 
    CrossFeatureExtractor
)
from gnn.graph_utils import construct_graph
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Distributed Feature Extraction')
    parser.add_argument('--transaction-data', type=str, required=True,
                       help='Path to transaction CSV file')
    parser.add_argument('--identity-data', type=str, default=None,
                       help='Path to identity CSV file (optional)')
    parser.add_argument('--training-dir', type=str, default='./data',
                       help='Directory containing graph edge lists')
    parser.add_argument('--output-dir', type=str, default='./features',
                       help='Output directory for extracted features')
    parser.add_argument('--cache-dir', type=str, default='./feature_cache',
                       help='Cache directory for intermediate results')
    parser.add_argument('--n-workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--extractors', type=str, nargs='+', default=None,
                       help='List of extractor names to use (default: all)')
    parser.add_argument('--target-ntype', type=str, default='TransactionID',
                       help='Target node type name')
    parser.add_argument('--nodes', type=str, default='features.csv',
                       help='Node features file name')
    parser.add_argument('--edges', type=str, default='relation*',
                       help='Edge list pattern')
    
    args = parser.parse_args()
    
    logger.info("Starting distributed feature extraction")
    logger.info(f"Transaction data: {args.transaction_data}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Workers: {args.n_workers}")
    
    # Load transaction data
    logger.info("Loading transaction data...")
    transaction_df = pd.read_csv(args.transaction_data)
    logger.info(f"Loaded {len(transaction_df)} transactions")
    
    # Load identity data if provided
    identity_df = None
    if args.identity_data and os.path.exists(args.identity_data):
        logger.info("Loading identity data...")
        identity_df = pd.read_csv(args.identity_data)
        logger.info(f"Loaded {len(identity_df)} identity records")
    
    # Merge transaction and identity data
    if identity_df is not None:
        transaction_df = transaction_df.merge(identity_df, on='TransactionID', how='left')
        logger.info("Merged transaction and identity data")
    
    # Load graph if available
    graph_data = None
    if os.path.exists(args.training_dir):
        try:
            from gnn.graph_utils import get_edgelists
            edges = get_edgelists(args.edges, args.training_dir)
            if edges:
                logger.info(f"Loading graph from {args.training_dir}...")
                graph_data, _, _, _ = construct_graph(
                    args.training_dir, edges, args.nodes, args.target_ntype
                )
                logger.info("Graph loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load graph: {e}. Continuing without graph features.")
    
    # Initialize feature registry with advanced extractors
    registry = initialize_default_registry()
    registry.register(GraphCentralityExtractor())
    registry.register(PatternMatchingExtractor())
    registry.register(CrossFeatureExtractor())
    
    # Initialize pipeline
    pipeline = DistributedFeaturePipeline(
        registry=registry,
        n_workers=args.n_workers,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir
    )
    
    # Extract features
    logger.info("Extracting features...")
    features = pipeline.process_transaction_data(
        transaction_df=transaction_df,
        graph_data=graph_data,
        extractor_names=args.extractors,
        use_cache=True
    )
    
    logger.info(f"Extracted features shape: {features.shape}")
    
    # Save features
    output_path = os.path.join(args.output_dir, 'extracted_features.npy')
    np.save(output_path, features)
    logger.info(f"Saved features to {output_path}")
    
    # Save feature metadata
    metadata = {
        'n_samples': len(transaction_df),
        'n_features': features.shape[1],
        'extractor_names': args.extractors or registry.list_extractors(),
        'feature_shape': features.shape
    }
    
    metadata_path = os.path.join(args.output_dir, 'feature_metadata.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")
    
    logger.info("Feature extraction completed successfully!")


if __name__ == '__main__':
    main()

