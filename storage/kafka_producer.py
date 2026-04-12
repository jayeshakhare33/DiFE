"""
Kafka Producer for Feature Extraction Requests
Publishes feature extraction requests to Kafka for distributed processing
"""

import json
import logging
from typing import List, Dict, Any
from kafka import KafkaProducer
from kafka.errors import KafkaError
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureExtractionProducer:
    """Producer for feature extraction requests"""
    
    def __init__(self, config_path=None):
        """Initialize Kafka producer"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        kafka_config = config.get('kafka', {})
        self.bootstrap_servers = kafka_config.get('bootstrap_servers', ['localhost:9092'])
        self.topic = kafka_config.get('topics', {}).get('feature_extraction', 'feature-extraction')
        
        producer_config = kafka_config.get('producer', {})
        
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
            compression_type=producer_config.get('compression_type', 'gzip'),
            acks=producer_config.get('acks', 'all'),
            retries=3
        )
        
        logger.info(f"Initialized FeatureExtractionProducer for topic: {self.topic}")
    
    def publish_extraction_request(self, node_ids: List[str] = None, 
                                   edge_ids: List[str] = None,
                                   extraction_type: str = 'both',
                                   metadata: Dict[str, Any] = None):
        """
        Publish feature extraction request to Kafka
        
        Args:
            node_ids: List of node IDs to extract features for (None = all nodes)
            edge_ids: List of edge IDs to extract features for (None = all edges)
            extraction_type: 'node', 'edge', or 'both'
            metadata: Additional metadata
        """
        message = {
            'extraction_type': extraction_type,
            'node_ids': node_ids,
            'edge_ids': edge_ids,
            'metadata': metadata or {}
        }
        
        try:
            future = self.producer.send(self.topic, value=message)
            # Wait for message to be sent
            record_metadata = future.get(timeout=10)
            logger.info(f"Published extraction request to {self.topic} "
                       f"[partition={record_metadata.partition}, offset={record_metadata.offset}]")
            return True
        except KafkaError as e:
            logger.error(f"Failed to publish extraction request: {e}")
            return False
    
    def publish_batch_requests(self, batches: List[Dict[str, Any]]):
        """
        Publish multiple extraction requests in batch
        
        Args:
            batches: List of extraction request dictionaries
        """
        success_count = 0
        for batch in batches:
            if self.publish_extraction_request(**batch):
                success_count += 1
        
        logger.info(f"Published {success_count}/{len(batches)} extraction requests")
        return success_count
    
    def close(self):
        """Close the producer"""
        self.producer.close()
        logger.info("FeatureExtractionProducer closed")


if __name__ == "__main__":
    # Test the producer
    logging.basicConfig(level=logging.INFO)
    
    producer = FeatureExtractionProducer()
    
    # Publish a test request
    producer.publish_extraction_request(
        extraction_type='both',
        metadata={'test': True, 'timestamp': '2025-11-25'}
    )
    
    producer.close()
    print("✅ Test message published successfully!")

