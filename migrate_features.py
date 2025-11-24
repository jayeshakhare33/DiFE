"""
Migration script to convert CSV features to Parquet format
Run this to migrate from CSV to Parquet for better performance
"""

import os
import sys
import logging
from storage import FeatureStore
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_csv_to_parquet(base_dir: str = './data/features'):
    """
    Migrate features from CSV to Parquet format
    
    Args:
        base_dir: Base directory containing feature files
    """
    logger.info("Starting migration from CSV to Parquet...")
    
    # Initialize stores
    csv_store = FeatureStore(backend_type='csv', base_dir=base_dir)
    parquet_store = FeatureStore(backend_type='parquet', base_dir=base_dir)
    
    # List of feature keys to migrate
    feature_keys = ['node_features', 'edge_features']
    
    migrated = []
    skipped = []
    errors = []
    
    for key in feature_keys:
        try:
            # Check if CSV file exists
            if not csv_store.exists(key):
                logger.warning(f"CSV file for '{key}' not found, skipping...")
                skipped.append(key)
                continue
            
            # Load from CSV
            logger.info(f"Loading '{key}' from CSV...")
            features = csv_store.load_features(key)
            logger.info(f"  Loaded {len(features)} rows with {len(features.columns)} columns")
            
            # Save to Parquet
            logger.info(f"Saving '{key}' to Parquet...")
            parquet_store.save_features(features, key)
            
            # Verify
            if parquet_store.exists(key):
                # Compare file sizes
                csv_path = os.path.join(base_dir, f'{key}.csv')
                parquet_path = os.path.join(base_dir, f'{key}.parquet')
                
                csv_size = os.path.getsize(csv_path) if os.path.exists(csv_path) else 0
                parquet_size = os.path.getsize(parquet_path) if os.path.exists(parquet_path) else 0
                
                compression_ratio = (1 - parquet_size / csv_size) * 100 if csv_size > 0 else 0
                
                logger.info(f"  ✅ Successfully migrated '{key}'")
                logger.info(f"     CSV size: {csv_size / 1024:.2f} KB")
                logger.info(f"     Parquet size: {parquet_size / 1024:.2f} KB")
                logger.info(f"     Compression: {compression_ratio:.1f}%")
                
                migrated.append(key)
            else:
                raise Exception(f"Parquet file not created for '{key}'")
                
        except Exception as e:
            logger.error(f"  ❌ Error migrating '{key}': {e}")
            errors.append((key, str(e)))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Migration Summary:")
    logger.info(f"  ✅ Migrated: {len(migrated)}")
    logger.info(f"  ⏭️  Skipped: {len(skipped)}")
    logger.info(f"  ❌ Errors: {len(errors)}")
    
    if migrated:
        logger.info(f"\n✅ Successfully migrated: {', '.join(migrated)}")
        logger.info("\n💡 Next steps:")
        logger.info("   1. Update config.yaml: backend: 'parquet'")
        logger.info("   2. Test loading features with Parquet backend")
        logger.info("   3. (Optional) Keep CSV files as backup")
    
    if errors:
        logger.error(f"\n❌ Errors occurred: {errors}")
        return False
    
    return True


def compare_performance(base_dir: str = './data/features', key: str = 'node_features'):
    """
    Compare read performance between CSV and Parquet
    
    Args:
        base_dir: Base directory containing feature files
        key: Feature key to test
    """
    import time
    
    logger.info(f"\n{'='*60}")
    logger.info("Performance Comparison")
    logger.info(f"{'='*60}")
    
    csv_store = FeatureStore(backend_type='csv', base_dir=base_dir)
    parquet_store = FeatureStore(backend_type='parquet', base_dir=base_dir)
    
    if not csv_store.exists(key) or not parquet_store.exists(key):
        logger.warning(f"Both CSV and Parquet files must exist for '{key}'")
        return
    
    # Warm up (first read is slower due to caching)
    _ = csv_store.load_features(key)
    _ = parquet_store.load_features(key)
    
    # Test CSV
    times_csv = []
    for _ in range(5):
        start = time.time()
        _ = csv_store.load_features(key)
        times_csv.append(time.time() - start)
    
    # Test Parquet
    times_parquet = []
    for _ in range(5):
        start = time.time()
        _ = parquet_store.load_features(key)
        times_parquet.append(time.time() - start)
    
    avg_csv = sum(times_csv) / len(times_csv) * 1000  # Convert to ms
    avg_parquet = sum(times_parquet) / len(times_parquet) * 1000
    
    speedup = avg_csv / avg_parquet if avg_parquet > 0 else 0
    
    logger.info(f"\nResults for '{key}':")
    logger.info(f"  CSV average:    {avg_csv:.2f} ms")
    logger.info(f"  Parquet average: {avg_parquet:.2f} ms")
    logger.info(f"  Speedup:        {speedup:.1f}x faster")
    logger.info(f"{'='*60}\n")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate features from CSV to Parquet')
    parser.add_argument('--base-dir', type=str, default='./data/features',
                       help='Base directory for feature files')
    parser.add_argument('--compare', action='store_true',
                       help='Compare performance after migration')
    parser.add_argument('--key', type=str, default='node_features',
                       help='Feature key for performance comparison')
    
    args = parser.parse_args()
    
    # Migrate
    success = migrate_csv_to_parquet(args.base_dir)
    
    # Compare performance if requested
    if args.compare and success:
        compare_performance(args.base_dir, args.key)
    
    if success:
        logger.info("\n✅ Migration completed successfully!")
        sys.exit(0)
    else:
        logger.error("\n❌ Migration completed with errors")
        sys.exit(1)


if __name__ == '__main__':
    main()

