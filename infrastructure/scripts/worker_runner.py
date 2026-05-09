#!/usr/bin/env python3
"""worker_runner.py – executed on each worker EC2 instance.
Workflow:
1. Load the full transaction CSV.
2. Build a minimal DGL heterogeneous graph.
3. Run either FeatureExtractor (nodes, 50 features) or EdgeFeatureExtractor (edges, 12 features).
4. Save the corresponding features and mapping to local Parquet files (bypassing S3 for local testing).
5. (In production) POST a JSON callback to master and self-terminate.
"""

import argparse
import os
import sys
import pandas as pd

# Add repository root to path for importing the feature extractor
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(repo_root)

from feature_engineering.feature_extractor import FeatureExtractor, EdgeFeatureExtractor


def build_graph(df: pd.DataFrame):
    import dgl
    import torch as th

    users = pd.concat([df["sender_id"], df["receiver_id"]]).unique()
    user_id_map = {uid: idx for idx, uid in enumerate(users)}
    src = df["sender_id"].map(user_id_map).values
    dst = df["receiver_id"].map(user_id_map).values

    g = dgl.heterograph({("user", "transaction", "user"): (src, dst)})
    g.edges["transaction"].data["amount"] = th.tensor(df["amount"].values, dtype=th.float32)
    if "timestamp" in df.columns:
        g.edges["transaction"].data["timestamp"] = th.tensor(df["timestamp"].values, dtype=th.float32)
    return g, user_id_map


def main():
    parser = argparse.ArgumentParser(description="Worker runner for feature extraction classes")
    parser.add_argument("--partition-csv", required=True, help="Local path to the full transaction CSV")
    parser.add_argument("--worker-id", type=int, required=True, help="Numeric worker identifier")
    parser.add_argument("--extractor-class", required=True, choices=["FeatureExtractor", "EdgeFeatureExtractor"], help="Class to run")
    parser.add_argument("--output-dir", default="./data/features", help="Local directory to save parquets")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================================
    # S3 CSV DOWNLOAD (Commented out for you to add your S3 bucket logic later)
    # =========================================================================
    # print("[WORKER] Downloading transactions CSV from S3...")
    # s3 = boto3.client('s3')
    # s3_bucket_name = "YOUR_S3_BUCKET_NAME"
    # s3_csv_key = "YOUR_CSV_KEY"
    # s3.download_file(s3_bucket_name, s3_csv_key, args.partition_csv)
    # print("[WORKER] S3 Download complete.")
    # =========================================================================

    # Load CSV
    df = pd.read_csv(args.partition_csv)
    required = {"transaction_id", "sender_id", "receiver_id", "amount", "timestamp"}
    if not required.issubset(df.columns):
        print(f"[WORKER {args.worker_id}] Missing required columns in CSV")
        sys.exit(1)

    # Build graph
    g, user_map = build_graph(df)

    if args.extractor_class == "FeatureExtractor":
        extractor = FeatureExtractor(transaction_df=df)
        features_df = extractor.extract_all_features(g, node_type='user', transaction_df=df)
        feats_path = os.path.join(args.output_dir, f"worker_{args.worker_id}_node_features.parquet")
        features_df.to_parquet(feats_path)

        mapping_records = []
        for _, row in df.iterrows():
            mapping_records.append({
                "transaction_id": row["transaction_id"],
                "sender_id": row["sender_id"],
                "receiver_id": row["receiver_id"],
                "sender_node_idx": user_map[row["sender_id"]],
                "receiver_node_idx": user_map[row["receiver_id"]],
                "worker_id": args.worker_id,
            })
        mapping_df = pd.DataFrame(mapping_records)
        map_path = os.path.join(args.output_dir, f"worker_{args.worker_id}_node_mapping.parquet")
        mapping_df.to_parquet(map_path)
    else:
        extractor = EdgeFeatureExtractor()
        edge_features_dict = extractor.extract_all_edge_features(g, edge_type='transaction', transaction_df=df)
        edge_features_df = pd.DataFrame({k: v.numpy() for k, v in edge_features_dict.items()})
        edge_features_df['transaction_id'] = df['transaction_id'].values
        feats_path = os.path.join(args.output_dir, f"worker_{args.worker_id}_edge_features.parquet")
        edge_features_df.to_parquet(feats_path)
        
        map_path = os.path.join(args.output_dir, f"worker_{args.worker_id}_edge_mapping.parquet")
        mapping_df = pd.DataFrame({
            "transaction_id": df["transaction_id"].values,
            "edge_idx": range(len(df)),
            "worker_id": args.worker_id
        })
        mapping_df.to_parquet(map_path)

    # Send files to Master
    try:
        print(f"[WORKER {args.worker_id}] Uploading files to Master node at {args.master_upload_url}")
        
        # Upload features parquet
        with open(feats_path, 'rb') as f:
            files = {'file': (os.path.basename(feats_path), f, 'application/octet-stream')}
            data = {'worker_id': str(args.worker_id), 'extractor_class': args.extractor_class}
            resp = requests.post(args.master_upload_url, data=data, files=files, timeout=30)
            resp.raise_for_status()
            
        # Upload mapping parquet
        with open(map_path, 'rb') as f:
            files = {'file': (os.path.basename(map_path), f, 'application/octet-stream')}
            data = {'worker_id': str(args.worker_id), 'extractor_class': args.extractor_class}
            resp = requests.post(args.master_upload_url, data=data, files=files, timeout=30)
            resp.raise_for_status()

        print(f"[WORKER {args.worker_id}] Successfully uploaded files to master.")
    except Exception as e:
        print(f"[WORKER {args.worker_id}] Failed to upload to master: {e}")

if __name__ == "__main__":
    main()
