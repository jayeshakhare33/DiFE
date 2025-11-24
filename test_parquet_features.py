"""
Test script to verify Parquet feature storage is working correctly
"""

import time
import logging
from storage import FeatureStore
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_parquet_features():
    """Test loading features from Parquet"""
    logger.info("="*60)
    logger.info("Testing Parquet Feature Storage")
    logger.info("="*60)
    
    # Initialize Parquet store
    store = FeatureStore(backend_type='parquet', base_dir='./data/features')
    
    # Test node features
    logger.info("\n1. Testing Node Features...")
    try:
        start = time.time()
        node_features = store.load_features('node_features')
        load_time = (time.time() - start) * 1000  # Convert to ms
        
        logger.info(f"   ✅ Loaded successfully!")
        logger.info(f"   📊 Shape: {node_features.shape[0]} rows × {node_features.shape[1]} columns")
        logger.info(f"   ⏱️  Load time: {load_time:.2f} ms")
        logger.info(f"   📝 Columns: {list(node_features.columns[:5])}... (showing first 5)")
        
        # Show sample data
        logger.info(f"\n   Sample data (first 3 rows, first 5 columns):")
        print(node_features.iloc[:3, :5].to_string())
        
    except Exception as e:
        logger.error(f"   ❌ Error loading node features: {e}")
        return False
    
    # Test edge features
    logger.info("\n2. Testing Edge Features...")
    try:
        start = time.time()
        edge_features = store.load_features('edge_features')
        load_time = (time.time() - start) * 1000
        
        logger.info(f"   ✅ Loaded successfully!")
        logger.info(f"   📊 Shape: {edge_features.shape[0]} rows × {edge_features.shape[1]} columns")
        logger.info(f"   ⏱️  Load time: {load_time:.2f} ms")
        logger.info(f"   📝 Columns: {list(edge_features.columns)}")
        
        # Show sample data
        logger.info(f"\n   Sample data (first 3 rows):")
        print(edge_features.iloc[:3].to_string())
        
    except Exception as e:
        logger.error(f"   ❌ Error loading edge features: {e}")
        return False
    
    # Performance comparison (if CSV files exist)
    logger.info("\n3. Performance Comparison...")
    try:
        csv_store = FeatureStore(backend_type='csv', base_dir='./data/features')
        
        if csv_store.exists('node_features'):
            # Warm up
            _ = csv_store.load_features('node_features')
            _ = store.load_features('node_features')
            
            # Test CSV
            times_csv = []
            for _ in range(3):
                start = time.time()
                _ = csv_store.load_features('node_features')
                times_csv.append((time.time() - start) * 1000)
            
            # Test Parquet
            times_parquet = []
            for _ in range(3):
                start = time.time()
                _ = store.load_features('node_features')
                times_parquet.append((time.time() - start) * 1000)
            
            avg_csv = sum(times_csv) / len(times_csv)
            avg_parquet = sum(times_parquet) / len(times_parquet)
            speedup = avg_csv / avg_parquet if avg_parquet > 0 else 0
            
            logger.info(f"   CSV average:    {avg_csv:.2f} ms")
            logger.info(f"   Parquet average: {avg_parquet:.2f} ms")
            if speedup > 1:
                logger.info(f"   🚀 Parquet is {speedup:.1f}x faster!")
            else:
                logger.info(f"   ⚠️  Similar performance (small dataset)")
    except Exception as e:
        logger.warning(f"   Could not compare performance: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ All tests passed! Parquet storage is working correctly.")
    logger.info("="*60)
    
    return True


if __name__ == '__main__':
    success = test_parquet_features()
    exit(0 if success else 1)

