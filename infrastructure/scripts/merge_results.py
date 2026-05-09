#!/usr/bin/env python3
"""merge_results.py – executed on the master EC2 after all workers finish.
It downloads the per‑worker Parquet feature and mapping files from the shared S3 bucket,
concatenates the feature DataFrames, merges the mapping tables, and writes the combined
outputs to a local directory (./data/features). The script can be invoked directly
or imported by the master orchestrator.
"""

import argparse
import os
import sys
import boto3
import pandas as pd
from datetime import datetime

def download_parquet(s3_client, bucket, key, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(bucket, key, local_path)

def main():
    parser = argparse.ArgumentParser(description="Merge worker Parquet results")
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket containing results")
    parser.add_argument("--worker-count", type=int, default=4, help="Number of workers")
    parser.add_argument("--output-dir", default="./data/features", help="Local output directory")
    args = parser.parse_args()

    s3 = boto3.client('s3')
    feature_dfs = []
    mapping_dfs = []

    for i in range(args.worker_count):
        feat_key = f"features/worker_{i}_features.parquet"
        map_key = f"features/worker_{i}_mapping.parquet"
        local_feat = f"/tmp/worker_{i}_features.parquet"
        local_map = f"/tmp/worker_{i}_mapping.parquet"
        print(f"Downloading {feat_key} and {map_key}")
        download_parquet(s3, args.s3_bucket, feat_key, local_feat)
        download_parquet(s3, args.s3_bucket, map_key, local_map)
        feature_dfs.append(pd.read_parquet(local_feat))
        mapping_dfs.append(pd.read_parquet(local_map))

    # Concatenate
    all_features = pd.concat(feature_dfs, ignore_index=True)
    all_mapping = pd.concat(mapping_dfs, ignore_index=True)

    # Add timestamps for provenance
    now_iso = datetime.utcnow().isoformat()
    all_features['extraction_timestamp'] = now_iso
    all_mapping['extraction_timestamp'] = now_iso

    # Write out locally
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    all_features_path = os.path.join(out_dir, "all_features.parquet")
    all_mapping_path = os.path.join(out_dir, "global_mapping.parquet")
    all_features.to_parquet(all_features_path)
    all_mapping.to_parquet(all_mapping_path)
    print(f"Merged features saved to {all_features_path}")
    print(f"Merged mapping saved to {all_mapping_path}")

if __name__ == "__main__":
    main()
