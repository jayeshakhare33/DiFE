"""
Orchestrate Distributed Feature Extraction
Coordinates Kafka producer, workers, and aggregator
"""

import sys
import logging
import time
import yaml
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from storage.kafka_producer import FeatureExtractionProducer
from storage.kafka_consumer import FeatureExtractionWorker
from storage.feature_aggregator import FeatureAggregator
from neo4j import GraphDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DistributedFeatureExtractionOrchestrator:
    """Orchestrates distributed feature extraction"""
    
    def __init__(self, config_path=None):
        """Initialize orchestrator"""
        if config_path is None:
            config_path = project_root / 'config.yaml'
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get node IDs from Neo4j
        neo4j_config = self.config.get('database', {}).get('neo4j', {})
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_config.get('uri', 'bolt://localhost:7687'),
            auth=(neo4j_config.get('user', 'neo4j'),
                  neo4j_config.get('password', 'neo4j123'))
        )
    
    def get_all_node_ids(self) -> List[str]:
        """Get all node IDs from Neo4j"""
        with self.neo4j_driver.session() as session:
            query = "MATCH (u:User) RETURN u.user_id as user_id ORDER BY u.user_id"
            result = session.run(query)
            node_ids = [record['user_id'] for record in result]
        return node_ids
    
    def partition_node_ids(self, node_ids: List[str], num_partitions: int = 4) -> List[List[str]]:
        """Partition node IDs into batches for workers"""
        partition_size = len(node_ids) // num_partitions
        partitions = []
        
        for i in range(num_partitions):
            start_idx = i * partition_size
            if i == num_partitions - 1:
                # Last partition gets remaining nodes
                end_idx = len(node_ids)
            else:
                end_idx = (i + 1) * partition_size
            
            partitions.append(node_ids[start_idx:end_idx])
        
        return partitions
    
    def trigger_extraction(self, num_workers: int = 4):
        """Trigger distributed feature extraction"""
        logger.info("="*60)
        logger.info("Starting Distributed Feature Extraction")
        logger.info("="*60)
        
        # Get all node IDs
        logger.info("Loading node IDs from Neo4j...")
        node_ids = self.get_all_node_ids()
        logger.info(f"Found {len(node_ids)} nodes")
        
        # Partition nodes
        partitions = self.partition_node_ids(node_ids, num_workers)
        logger.info(f"Partitioned into {len(partitions)} batches:")
        for i, partition in enumerate(partitions):
            logger.info(f"  Partition {i}: {len(partition)} nodes")
        
        # Create producer
        producer = FeatureExtractionProducer()
        
        # Publish extraction requests for each partition
        logger.info("Publishing extraction requests to Kafka...")
        for i, partition in enumerate(partitions):
            success = producer.publish_extraction_request(
                node_ids=partition,
                extraction_type='both',
                metadata={'partition': i, 'total_nodes': len(partition)}
            )
            if success:
                logger.info(f"Published request for partition {i} ({len(partition)} nodes)")
            time.sleep(0.1)  # Small delay between requests
        
        producer.close()
        
        logger.info("="*60)
        logger.info("Extraction requests published!")
        logger.info("="*60)
        logger.info("Next steps:")
        logger.info("1. Start workers: python storage/kafka_consumer.py --worker-id 0")
        logger.info("2. Start aggregator: python storage/feature_aggregator.py")
        logger.info("="*60)
    
    def close(self):
        """Close connections"""
        self.neo4j_driver.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Distributed Feature Extraction Orchestrator')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    args = parser.parse_args()
    
    orchestrator = DistributedFeatureExtractionOrchestrator()
    
    try:
        orchestrator.trigger_extraction(num_workers=args.workers)
    finally:
        orchestrator.close()

