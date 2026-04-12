#!/usr/bin/env python3
"""
Verify Features in Redis
"""
import sys
import os
import redis
import pickle
import json
from pathlib import Path

def verify_features():
    """Verify features stored in Redis"""
    try:
        # Connect to Redis
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', '6379'))
        redis_password = os.getenv('REDIS_PASSWORD', None)
        redis_db = int(os.getenv('REDIS_DB', '0'))
        
        r = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            db=redis_db,
            decode_responses=False
        )
        
        print("=" * 60)
        print("Verifying Features in Redis")
        print("=" * 60)
        
        # Check node features
        print("\n1. Node Features:")
        node_features_data = r.get('features:node_features')
        if node_features_data:
            node_features = pickle.loads(node_features_data)
            print(f"   ✅ Node features DataFrame found")
            print(f"   - Shape: {node_features.shape}")
            print(f"   - Columns: {len(node_features.columns)}")
            print(f"   - Sample columns: {list(node_features.columns[:5])}")
        else:
            print("   ❌ Node features DataFrame not found")
        
        # Check edge features
        print("\n2. Edge Features:")
        edge_features_data = r.get('features:edge_features')
        if edge_features_data:
            edge_features = pickle.loads(edge_features_data)
            print(f"   ✅ Edge features DataFrame found")
            print(f"   - Shape: {edge_features.shape}")
            print(f"   - Columns: {len(edge_features.columns)}")
            print(f"   - Sample columns: {list(edge_features.columns[:5])}")
        else:
            print("   ❌ Edge features DataFrame not found")
        
        # Check metadata
        print("\n3. Metadata:")
        node_metadata = r.get('features:metadata:node')
        if node_metadata:
            metadata = json.loads(node_metadata.decode('utf-8'))
            print(f"   ✅ Node metadata found:")
            print(f"   - Feature count: {len(metadata.get('feature_names', []))}")
            print(f"   - Node count: {metadata.get('node_count', 0)}")
            print(f"   - Last update: {metadata.get('last_update', 'N/A')}")
        
        edge_metadata = r.get('features:metadata:edge')
        if edge_metadata:
            metadata = json.loads(edge_metadata.decode('utf-8'))
            print(f"   ✅ Edge metadata found:")
            print(f"   - Feature count: {len(metadata.get('feature_names', []))}")
            print(f"   - Edge count: {metadata.get('edge_count', 0)}")
            print(f"   - Last update: {metadata.get('last_update', 'N/A')}")
        
        # Check individual node features
        print("\n4. Individual Node Features:")
        feature_keys = r.keys('features:node:*')
        print(f"   - Found {len(feature_keys)} individual node feature keys")
        if feature_keys:
            # Sample a few
            sample_key = feature_keys[0]
            sample_data = r.get(sample_key)
            if sample_data:
                sample_features = pickle.loads(sample_data)
                node_id = sample_key.decode('utf-8').split(':')[-1]
                print(f"   - Sample node ({node_id}): {len(sample_features)} features")
                print(f"   - Sample features: {list(sample_features.keys())[:5]}")
        
        print("\n" + "=" * 60)
        print("Verification Complete!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = verify_features()
    sys.exit(0 if success else 1)

