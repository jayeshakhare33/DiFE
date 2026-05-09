import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from graph_processing.graph_builder import GraphBuilder
from feature_engineering.feature_extractor import FeatureExtractor, EdgeFeatureExtractor

def main():
    print("Loading transactions...")
    df = pd.read_csv('storage/india_fraud_data_explainable/transactions.csv')
    
    print("Building graph...")
    builder = GraphBuilder(num_workers=1, chunk_size=max(len(df), 1))
    g, user_id_to_node, df_ret = builder.build_user_transaction_graph(
        df.copy(), 
        sender_col='sender_id', 
        receiver_col='receiver_id', 
        amount_col='amount',
        timestamp_col='timestamp', 
        fraud_col='is_fraud_txn'
    )
    
    print("Extracting node features...")
    node_ext = FeatureExtractor(transaction_df=df)
    node_features = node_ext.extract_all_features(g, node_type='user')
    
    print("Extracting edge features...")
    edge_ext = EdgeFeatureExtractor()
    edge_features_dict = edge_ext.extract_all_edge_features(g, edge_type=('user', 'transaction', 'user'))
    
    edge_df = pd.DataFrame()
    for k, v in edge_features_dict.items():
        if hasattr(v, 'numpy'):
            edge_df[k] = v.numpy()
        else:
            edge_df[k] = v
            
    # Ensure directory exists
    os.makedirs('data/features', exist_ok=True)
    
    print("Saving to parquet...")
    node_features.to_parquet('data/features/node_features.parquet')
    edge_df.to_parquet('data/features/edge_features.parquet')
    
    print("Done!")

if __name__ == '__main__':
    main()
