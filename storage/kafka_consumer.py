"""
Kafka Consumer for Feature Extraction Workers
Consumes feature extraction requests and publishes computed features
"""

import json
import logging
import time
from typing import Dict, Any, Optional
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import yaml
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neo4j import GraphDatabase
from feature_engineering.feature_extractor import FeatureExtractor, EdgeFeatureExtractor
import dgl
import torch as th
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FeatureExtractionWorker:
    """Worker that consumes extraction requests and computes features"""
    
    def __init__(self, worker_id: int, config_path=None):
        """
        Initialize feature extraction worker
        
        Args:
            worker_id: Unique worker ID (0, 1, 2, 3, etc.)
            config_path: Path to config.yaml
        """
        self.worker_id = worker_id
        
        if config_path is None:
            config_path = project_root / 'config.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Kafka configuration
        kafka_config = config.get('kafka', {})
        self.bootstrap_servers = kafka_config.get('bootstrap_servers', ['localhost:9092'])
        
        topics = kafka_config.get('topics', {})
        self.extraction_topic = topics.get('feature_extraction', 'feature-extraction')
        self.computed_topic = topics.get('features_computed', 'features-computed')
        
        consumer_config = kafka_config.get('consumer', {})
        
        # Kafka consumer for extraction requests
        self.consumer = KafkaConsumer(
            self.extraction_topic,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id=f'feature-extractor-group-{worker_id}',
            auto_offset_reset=consumer_config.get('auto_offset_reset', 'earliest'),
            enable_auto_commit=True,
            consumer_timeout_ms=1000  # Timeout for polling
        )
        
        # Kafka producer for computed features
        producer_config = kafka_config.get('producer', {})
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
            compression_type=producer_config.get('compression_type', 'gzip'),
            acks=producer_config.get('acks', 'all')
        )
        
        # Neo4j configuration
        neo4j_config = config.get('database', {}).get('neo4j', {})
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_config.get('uri', 'bolt://localhost:7687'),
            auth=(neo4j_config.get('user', 'neo4j'),
                  neo4j_config.get('password', 'neo4j123'))
        )
        
        # Feature extractors
        self.node_extractor = FeatureExtractor()
        self.edge_extractor = EdgeFeatureExtractor()
        
        logger.info(f"Initialized FeatureExtractionWorker-{worker_id}")
    
    def load_subgraph_from_neo4j(self, node_ids: Optional[list] = None):
        """Load subgraph from Neo4j"""
        with self.neo4j_driver.session() as session:
            # Load nodes
            if node_ids:
                query = """
                MATCH (u:User)
                WHERE u.user_id IN $node_ids
                RETURN u.user_id as user_id
                """
                result = session.run(query, node_ids=node_ids)
            else:
                query = "MATCH (u:User) RETURN u.user_id as user_id"
                result = session.run(query)
            
            node_ids_list = [record['user_id'] for record in result]
            node_map = {node_id: idx for idx, node_id in enumerate(node_ids_list)}
            
            # Load edges
            if node_ids:
                query = """
                MATCH (u1:User)-[:SENT]->(t:Transaction)-[:TO]->(u2:User)
                WHERE u1.user_id IN $node_ids OR u2.user_id IN $node_ids
                RETURN u1.user_id as sender, u2.user_id as receiver,
                       t.transaction_id as txn_id, t.amount as amount,
                       t.timestamp as timestamp, t.is_fraud as is_fraud
                ORDER BY t.timestamp
                """
                result = session.run(query, node_ids=node_ids)
            else:
                query = """
                MATCH (u1:User)-[:SENT]->(t:Transaction)-[:TO]->(u2:User)
                RETURN u1.user_id as sender, u2.user_id as receiver,
                       t.transaction_id as txn_id, t.amount as amount,
                       t.timestamp as timestamp, t.is_fraud as is_fraud
                ORDER BY t.timestamp
                """
                result = session.run(query)
            
            edges = []
            edge_data = {'amount': [], 'timestamp': [], 'is_fraud': []}
            
            for record in result:
                sender = record['sender']
                receiver = record['receiver']
                
                if sender in node_map and receiver in node_map:
                    edges.append((node_map[sender], node_map[receiver]))
                    
                    # Convert timestamp
                    timestamp = record['timestamp']
                    if hasattr(timestamp, 'to_native'):
                        dt = timestamp.to_native()
                        timestamp = pd.to_datetime(dt).timestamp() if hasattr(dt, 'timestamp') else pd.to_datetime(dt).timestamp()
                    elif isinstance(timestamp, str):
                        timestamp = pd.to_datetime(timestamp).timestamp()
                    elif hasattr(timestamp, 'timestamp'):
                        timestamp = timestamp.timestamp()
                    else:
                        timestamp = float(timestamp) if timestamp else 0.0
                    
                    edge_data['amount'].append(float(record['amount']))
                    edge_data['timestamp'].append(timestamp)
                    edge_data['is_fraud'].append(bool(record.get('is_fraud', False)))
            
            if not edges:
                return None
            
            # Create DGL graph
            src_nodes = [e[0] for e in edges]
            dst_nodes = [e[1] for e in edges]
            num_nodes = len(node_ids_list)
            
            g = dgl.heterograph({
                ('user', 'transaction', 'user'): (src_nodes, dst_nodes)
            }, num_nodes_dict={'user': num_nodes})
            
            # Add edge features
            if edge_data['amount']:
                g.edges[('user', 'transaction', 'user')].data['amount'] = th.tensor(edge_data['amount'], dtype=th.float32)
            if edge_data['timestamp']:
                g.edges[('user', 'transaction', 'user')].data['timestamp'] = th.tensor(edge_data['timestamp'], dtype=th.float32)
            if edge_data['is_fraud']:
                g.edges[('user', 'transaction', 'user')].data['is_fraud'] = th.tensor(edge_data['is_fraud'], dtype=th.float32)
            
            g.node_ids = node_ids_list
            
            return g
    
    def extract_features(self, extraction_type: str, node_ids: Optional[list] = None):
        """Extract features for given nodes/edges"""
        logger.info(f"Worker-{self.worker_id}: Extracting {extraction_type} features for {len(node_ids) if node_ids else 'all'} nodes")
        
        # Load graph from Neo4j
        g = self.load_subgraph_from_neo4j(node_ids)
        if g is None:
            logger.warning(f"Worker-{self.worker_id}: No graph data found")
            return None, None
        
        results = {}
        
        # Extract node features
        if extraction_type in ['node', 'both']:
            try:
                # Get transaction data for temporal features
                transaction_data = []
                with self.neo4j_driver.session() as session:
                    query = """
                    MATCH (u:User)-[:SENT]->(t:Transaction)
                    RETURN u.user_id as user_id, t.amount as amount, t.timestamp as timestamp
                    ORDER BY t.timestamp
                    """
                    result = session.run(query)
                    for record in result:
                        timestamp = record['timestamp']
                        if hasattr(timestamp, 'to_native'):
                            dt = timestamp.to_native()
                            timestamp = pd.to_datetime(dt).timestamp() if hasattr(dt, 'timestamp') else pd.to_datetime(dt).timestamp()
                        elif isinstance(timestamp, str):
                            timestamp = pd.to_datetime(timestamp).timestamp()
                        elif hasattr(timestamp, 'timestamp'):
                            timestamp = timestamp.timestamp()
                        else:
                            timestamp = float(timestamp) if timestamp else 0.0
                        
                        transaction_data.append({
                            'user_id': record['user_id'],
                            'amount': float(record['amount']),
                            'timestamp': timestamp
                        })
                
                transaction_df = pd.DataFrame(transaction_data) if transaction_data else None
                
                node_features = self.node_extractor.extract_all_features(
                    g, node_type='user', transaction_df=transaction_df
                )
                
                # Convert to dict for serialization
                results['node_features'] = {
                    'data': node_features.to_dict('records'),
                    'index': node_features.index.tolist(),
                    'columns': node_features.columns.tolist()
                }
                
                logger.info(f"Worker-{self.worker_id}: Extracted {len(node_features)} node features")
            except Exception as e:
                logger.error(f"Worker-{self.worker_id}: Error extracting node features: {e}")
                results['node_features'] = None
        
        # Extract edge features
        if extraction_type in ['edge', 'both']:
            try:
                edge_features_dict = self.edge_extractor.extract_all_edge_features(
                    g, edge_type=('user', 'transaction', 'user')
                )
                
                # Convert tensors to lists for serialization
                edge_features_serialized = {}
                for key, tensor in edge_features_dict.items():
                    edge_features_serialized[key] = tensor.numpy().tolist()
                
                results['edge_features'] = edge_features_serialized
                logger.info(f"Worker-{self.worker_id}: Extracted edge features")
            except Exception as e:
                logger.error(f"Worker-{self.worker_id}: Error extracting edge features: {e}")
                results['edge_features'] = None
        
        return results, g.node_ids if hasattr(g, 'node_ids') else None
    
    def process_message(self, message):
        """Process a feature extraction request message"""
        try:
            request = message.value
            extraction_type = request.get('extraction_type', 'both')
            node_ids = request.get('node_ids')
            edge_ids = request.get('edge_ids')
            metadata = request.get('metadata', {})
            
            logger.info(f"Worker-{self.worker_id}: Processing extraction request: type={extraction_type}")
            
            # Extract features
            features, processed_node_ids = self.extract_features(extraction_type, node_ids)
            
            if features is None:
                logger.warning(f"Worker-{self.worker_id}: No features extracted")
                return False
            
            # Publish computed features
            response = {
                'worker_id': self.worker_id,
                'extraction_type': extraction_type,
                'node_ids': processed_node_ids,
                'features': features,
                'metadata': metadata,
                'timestamp': time.time()
            }
            
            future = self.producer.send(self.computed_topic, value=response)
            record_metadata = future.get(timeout=10)
            
            logger.info(f"Worker-{self.worker_id}: Published features to {self.computed_topic} "
                       f"[partition={record_metadata.partition}]")
            return True
            
        except Exception as e:
            logger.error(f"Worker-{self.worker_id}: Error processing message: {e}")
            return False
    
    def run(self):
        """Run the worker - continuously consume and process messages"""
        logger.info(f"Worker-{self.worker_id}: Starting to consume from {self.extraction_topic}")
        
        try:
            while True:
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                if not message_batch:
                    continue
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        self.process_message(message)
                        self.consumer.commit()
        
        except KeyboardInterrupt:
            logger.info(f"Worker-{self.worker_id}: Shutting down...")
        finally:
            self.consumer.close()
            self.producer.close()
            self.neo4j_driver.close()
    
    def close(self):
        """Close all connections"""
        self.consumer.close()
        self.producer.close()
        self.neo4j_driver.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature Extraction Worker')
    parser.add_argument('--worker-id', type=int, required=True, help='Worker ID (0, 1, 2, 3, etc.)')
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    worker = FeatureExtractionWorker(worker_id=args.worker_id)
    
    try:
        worker.run()
    except KeyboardInterrupt:
        logger.info("Shutting down worker...")
    finally:
        worker.close()

