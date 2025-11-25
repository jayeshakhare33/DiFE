"""
Feature Aggregator - Consumes computed features from Kafka and stores in Redis
"""

import json
import logging
import time
from typing import Dict, Any, List
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import yaml
from pathlib import Path
import sys
import pandas as pd
import redis

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from storage.storage_backend import RedisBackend

logger = logging.getLogger(__name__)


class FeatureAggregator:
    """Aggregates features from multiple workers and stores in Redis"""
    
    def __init__(self, config_path=None):
        """Initialize feature aggregator"""
        if config_path is None:
            config_path = project_root / 'config.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Kafka configuration
        kafka_config = config.get('kafka', {})
        self.bootstrap_servers = kafka_config.get('bootstrap_servers', ['localhost:9092'])
        
        topics = kafka_config.get('topics', {})
        self.computed_topic = topics.get('features_computed', 'features-computed')
        self.stored_topic = topics.get('features_stored', 'features-stored')
        
        consumer_config = kafka_config.get('consumer', {})
        
        # Kafka consumer for computed features
        self.consumer = KafkaConsumer(
            self.computed_topic,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='feature-aggregator-group',
            auto_offset_reset=consumer_config.get('auto_offset_reset', 'earliest'),
            enable_auto_commit=True
        )
        
        # Redis backend
        storage_config = config.get('storage', {})
        redis_config = storage_config.get('redis', {})
        
        self.redis_backend = RedisBackend(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            db=redis_config.get('db', 0),
            password=redis_config.get('password')
        )
        
        # Aggregation state
        self.node_features_collected = {}
        self.edge_features_collected = {}
        self.workers_completed = set()
        
        logger.info("Initialized FeatureAggregator")
    
    def aggregate_node_features(self, features_list: List[Dict]) -> pd.DataFrame:
        """Aggregate node features from multiple workers"""
        all_records = []
        
        for features_dict in features_list:
            if features_dict and 'data' in features_dict:
                all_records.extend(features_dict['data'])
        
        if not all_records:
            return None
        
        # Create DataFrame
        df = pd.DataFrame(all_records)
        
        # Remove duplicates (keep last)
        if 'node_id' in df.columns:
            df = df.drop_duplicates(subset=['node_id'], keep='last')
        elif len(features_list) > 0 and 'index' in features_list[0]:
            # Use index from first features dict
            df.index = features_list[0]['index'][:len(df)]
        
        return df
    
    def aggregate_edge_features(self, features_list: List[Dict]) -> Dict:
        """Aggregate edge features from multiple workers"""
        aggregated = {}
        
        for features_dict in features_list:
            if features_dict:
                for key, values in features_dict.items():
                    if key not in aggregated:
                        aggregated[key] = []
                    aggregated[key].extend(values)
        
        return aggregated
    
    def store_features(self, node_features: pd.DataFrame = None, edge_features: Dict = None):
        """Store aggregated features in Redis"""
        stored_count = 0
        
        # Store node features
        if node_features is not None and not node_features.empty:
            try:
                self.redis_backend.save_features(
                    node_features,
                    key_prefix='node_features'
                )
                stored_count += len(node_features)
                logger.info(f"Stored {len(node_features)} node features in Redis")
            except Exception as e:
                logger.error(f"Error storing node features: {e}")
        
        # Store edge features
        if edge_features:
            try:
                # Convert to DataFrame for storage
                edge_df = pd.DataFrame(edge_features)
                self.redis_backend.save_features(
                    edge_df,
                    key_prefix='edge_features'
                )
                stored_count += len(edge_df)
                logger.info(f"Stored {len(edge_df)} edge features in Redis")
            except Exception as e:
                logger.error(f"Error storing edge features: {e}")
        
        return stored_count
    
    def process_message(self, message):
        """Process a computed features message"""
        try:
            data = message.value
            worker_id = data.get('worker_id')
            extraction_type = data.get('extraction_type', 'both')
            features = data.get('features', {})
            node_ids = data.get('node_ids', [])
            
            logger.info(f"Aggregator: Received features from Worker-{worker_id}")
            
            # Collect features
            if extraction_type in ['node', 'both'] and 'node_features' in features:
                if worker_id not in self.node_features_collected:
                    self.node_features_collected[worker_id] = []
                self.node_features_collected[worker_id].append(features['node_features'])
            
            if extraction_type in ['edge', 'both'] and 'edge_features' in features:
                if worker_id not in self.edge_features_collected:
                    self.edge_features_collected[worker_id] = []
                self.edge_features_collected[worker_id].append(features['edge_features'])
            
            self.workers_completed.add(worker_id)
            
            # Check if we should aggregate (e.g., after all workers complete or timeout)
            # For now, aggregate after each message (can be optimized)
            return True
            
        except Exception as e:
            logger.error(f"Aggregator: Error processing message: {e}")
            return False
    
    def aggregate_and_store(self):
        """Aggregate all collected features and store in Redis"""
        logger.info("Aggregating features from all workers...")
        
        # Aggregate node features
        node_features_list = []
        for worker_id, features_list in self.node_features_collected.items():
            node_features_list.extend(features_list)
        
        node_features_df = self.aggregate_node_features(node_features_list) if node_features_list else None
        
        # Aggregate edge features
        edge_features_list = []
        for worker_id, features_list in self.edge_features_collected.items():
            edge_features_list.extend(features_list)
        
        edge_features_dict = self.aggregate_edge_features(edge_features_list) if edge_features_list else None
        
        # Store in Redis
        stored_count = self.store_features(node_features_df, edge_features_dict)
        
        logger.info(f"Aggregated and stored {stored_count} features in Redis")
        
        # Clear collected features
        self.node_features_collected.clear()
        self.edge_features_collected.clear()
        self.workers_completed.clear()
        
        return stored_count
    
    def run(self, aggregation_interval=5.0):
        """
        Run the aggregator - continuously consume and aggregate features
        
        Args:
            aggregation_interval: Seconds to wait before aggregating (0 = aggregate immediately)
        """
        logger.info(f"Aggregator: Starting to consume from {self.computed_topic}")
        
        last_aggregation = time.time()
        
        try:
            while True:
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                if message_batch:
                    for topic_partition, messages in message_batch.items():
                        for message in messages:
                            self.process_message(message)
                            self.consumer.commit()
                
                # Aggregate periodically or when batch is complete
                current_time = time.time()
                if aggregation_interval > 0 and (current_time - last_aggregation) >= aggregation_interval:
                    if self.node_features_collected or self.edge_features_collected:
                        self.aggregate_and_store()
                        last_aggregation = current_time
        
        except KeyboardInterrupt:
            logger.info("Aggregator: Shutting down...")
            # Final aggregation
            if self.node_features_collected or self.edge_features_collected:
                self.aggregate_and_store()
        finally:
            self.consumer.close()
    
    def close(self):
        """Close all connections"""
        self.consumer.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature Aggregator')
    parser.add_argument('--interval', type=float, default=5.0, help='Aggregation interval in seconds')
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    aggregator = FeatureAggregator()
    
    try:
        aggregator.run(aggregation_interval=args.interval)
    except KeyboardInterrupt:
        logger.info("Shutting down aggregator...")
    finally:
        aggregator.close()

