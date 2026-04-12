"""
Feature Loader Script
Loads features from Parquet files into Redis for fast access across containers
"""

import os
import sys
import logging
import time
from storage import FeatureStore
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_features_to_redis():
    """Load features from Parquet to Redis"""
    logger.info("="*60)
    logger.info("Loading Features to Redis")
    logger.info("="*60)
    
    # Get configuration from environment
    storage_base_dir = os.getenv('STORAGE_BASE_DIR', '/app/data/features')
    redis_host = os.getenv('REDIS_HOST', 'redis')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    
    logger.info(f"Storage base directory: {storage_base_dir}")
    logger.info(f"Redis host: {redis_host}:{redis_port}")
    
    # Initialize stores
    try:
        # Load from Parquet (source)
        parquet_store = FeatureStore(
            backend_type='parquet',
            base_dir=storage_base_dir
        )
        logger.info("✅ Parquet store initialized")
        
        # Save to Redis (destination)
        redis_store = FeatureStore(
            backend_type='redis',
            host=redis_host,
            port=redis_port
        )
        logger.info("✅ Redis store initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize stores: {e}")
        return False
    
    # Load and transfer features
    feature_keys = ['node_features', 'edge_features']
    loaded = []
    errors = []
    
    for key in feature_keys:
        try:
            logger.info(f"\n📦 Loading '{key}' from Parquet...")
            
            # Check if Parquet file exists
            if not parquet_store.exists(key):
                logger.warning(f"⚠️  Parquet file for '{key}' not found, skipping...")
                continue
            
            # Load from Parquet
            start_time = time.time()
            features = parquet_store.load_features(key)
            load_time = (time.time() - start_time) * 1000
            
            logger.info(f"   ✅ Loaded {len(features)} rows × {len(features.columns)} columns")
            logger.info(f"   ⏱️  Load time: {load_time:.2f} ms")
            
            # Save to Redis
            logger.info(f"💾 Saving '{key}' to Redis...")
            save_start = time.time()
            redis_store.save_features(features, key)
            save_time = (time.time() - save_start) * 1000
            
            logger.info(f"   ✅ Saved to Redis")
            logger.info(f"   ⏱️  Save time: {save_time:.2f} ms")
            
            # Verify
            if redis_store.exists(key):
                logger.info(f"   ✅ Verification: Features exist in Redis")
                loaded.append(key)
            else:
                raise Exception(f"Features not found in Redis after save")
                
        except Exception as e:
            logger.error(f"   ❌ Error loading '{key}': {e}")
            errors.append((key, str(e)))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Feature Loading Summary:")
    logger.info(f"   ✅ Loaded: {len(loaded)}")
    logger.info(f"   ❌ Errors: {len(errors)}")
    
    if loaded:
        logger.info(f"\n✅ Successfully loaded: {', '.join(loaded)}")
        logger.info("\n💡 Features are now available in Redis for fast access!")
        logger.info("   All containers can now access features via Redis.")
    
    if errors:
        logger.error(f"\n❌ Errors occurred: {errors}")
        return False
    
    return True


if __name__ == '__main__':
    success = load_features_to_redis()
    sys.exit(0 if success else 1)

