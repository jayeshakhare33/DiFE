#!/usr/bin/env python3
"""
Complete Feature Extraction Pipeline
Extracts all 62 features (50 node + 12 edge) and stores in Redis
"""
import sys
import os
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import redis
import pickle
import json
from datetime import datetime
from typing import Dict, List, Optional

from feature_engineering.feature_extractor import FeatureExtractor, EdgeFeatureExtractor
from storage.feature_store import FeatureStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompleteFeatureExtractor:
    """Extract all features from Neo4j and store in Redis"""
    
    def __init__(self):
        """Initialize connections"""
        # Neo4j connection
        neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
        neo4j_password = os.getenv('NEO4J_PASSWORD', 'neo4j123')
        
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        logger.info(f"Connected to Neo4j at {neo4j_uri}")
        
        # Redis connection
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', '6379'))
        redis_password = os.getenv('REDIS_PASSWORD', None)
        redis_db = int(os.getenv('REDIS_DB', '0'))
        
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            db=redis_db,
            decode_responses=False
        )
        self.redis_client.ping()
        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        
        # Feature extractors
        self.node_extractor = FeatureExtractor()
        self.edge_extractor = EdgeFeatureExtractor()
        
        # Feature store (for Parquet backup)
        self.feature_store = FeatureStore(
            backend_type='parquet',
            base_dir='./data/features'
        )
    
    def neo4j_to_dgl_graph(self, node_ids: Optional[List[str]] = None):
        """
        Convert Neo4j graph to DGL format
        This is a simplified version - you may need to adapt based on your DGL graph builder
        """
        import dgl
        import torch as th
        
        with self.neo4j_driver.session() as session:
            # Check if any User nodes exist
            result = session.run("MATCH (u:User) RETURN count(u) as count")
            user_count = result.single()['count']
            
            if user_count == 0:
                logger.error("No User nodes found in Neo4j!")
                logger.error("Please run: python scripts/sync_postgres_to_neo4j.py")
                return None
            
            # Get all users if node_ids not specified
            if node_ids is None:
                result = session.run("MATCH (u:User) RETURN u.user_id as user_id")
                node_ids = [record['user_id'] for record in result]
            
            if not node_ids:
                logger.error("No user IDs found!")
                return None
            
            logger.info(f"Processing {len(node_ids)} nodes")
            
            # Get all relationships - try different query patterns
            queries = [
                # Try with Transaction nodes
                """
                MATCH (u1:User)-[:SENT]->(t:Transaction)-[:TO]->(u2:User)
                WHERE u1.user_id IN $node_ids OR u2.user_id IN $node_ids
                RETURN u1.user_id as sender, u2.user_id as receiver, 
                       t.amount as amount, t.timestamp as timestamp, 
                       t.is_fraud as is_fraud
                ORDER BY t.timestamp
                """,
                # Fallback: Direct user-to-user if no Transaction nodes
                """
                MATCH (u1:User)-[r]->(u2:User)
                WHERE u1.user_id IN $node_ids OR u2.user_id IN $node_ids
                RETURN u1.user_id as sender, u2.user_id as receiver,
                       r.amount as amount, r.timestamp as timestamp,
                       false as is_fraud
                ORDER BY r.timestamp
                """
            ]
            
            edges = []
            edge_data = {
                'amount': [],
                'timestamp': [],
                'is_fraud': []
            }
            
            node_id_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
            
            # Try first query
            for query in queries:
                try:
                    result = session.run(query, node_ids=node_ids)
                    found_edges = False
                    
                    for record in result:
                        found_edges = True
                        sender = record.get('sender')
                        receiver = record.get('receiver')
                        
                        if sender and receiver and sender in node_id_map and receiver in node_id_map:
                            src_idx = node_id_map[sender]
                            dst_idx = node_id_map[receiver]
                            edges.append((src_idx, dst_idx))
                            
                            amount = record.get('amount', 0.0)
                            edge_data['amount'].append(float(amount) if amount else 0.0)
                            
                            # Convert timestamp to unix timestamp
                            timestamp = record.get('timestamp')
                            if timestamp:
                                if hasattr(timestamp, 'timestamp'):
                                    edge_data['timestamp'].append(timestamp.timestamp())
                                elif isinstance(timestamp, str):
                                    try:
                                        ts = pd.to_datetime(timestamp)
                                        edge_data['timestamp'].append(ts.timestamp())
                                    except:
                                        edge_data['timestamp'].append(0.0)
                                else:
                                    edge_data['timestamp'].append(0.0)
                            else:
                                edge_data['timestamp'].append(0.0)
                            
                            edge_data['is_fraud'].append(bool(record.get('is_fraud', False)))
                    
                    if found_edges:
                        break
                except Exception as e:
                    logger.debug(f"Query failed: {e}, trying next query...")
                    continue
            
            if not edges:
                logger.warning("No edges found in graph")
                logger.warning("This might be normal if you only have User nodes without relationships")
                logger.warning("You may need to sync data: python scripts/sync_postgres_to_neo4j.py")
                return None
            
            # Create DGL graph as heterogeneous graph with 'user' node type
            src_nodes = [e[0] for e in edges]
            dst_nodes = [e[1] for e in edges]
            
            # Ensure all node indices are valid
            max_node_idx = max(max(src_nodes), max(dst_nodes)) if edges else len(node_ids) - 1
            num_nodes = max(max_node_idx + 1, len(node_ids))
            
            # Create heterogeneous graph with 'user' node type
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
            
            # Store node ID mapping
            g.node_ids = node_ids
            
            logger.info(f"Created DGL graph with {g.number_of_nodes('user')} nodes and {g.number_of_edges(('user', 'transaction', 'user'))} edges")
            return g
    
    def extract_node_features(self, g) -> pd.DataFrame:
        """Extract all 50 node features"""
        logger.info("Extracting node features...")
        
        # Get transaction data from Neo4j for feature extraction
        transaction_data = []
        with self.neo4j_driver.session() as session:
            query = """
            MATCH (u:User)-[:SENT]->(t:Transaction)
            RETURN u.user_id as user_id, t.amount as amount, t.timestamp as timestamp
            ORDER BY t.timestamp
            """
            result = session.run(query)
            for record in result:
                # Convert Neo4j DateTime to unix timestamp
                timestamp = record['timestamp']
                if timestamp:
                    # Handle Neo4j DateTime objects
                    if hasattr(timestamp, 'to_native'):
                        # Neo4j DateTime object
                        dt = timestamp.to_native()
                        if hasattr(dt, 'timestamp'):
                            timestamp = dt.timestamp()
                        else:
                            timestamp = pd.to_datetime(dt).timestamp()
                    elif isinstance(timestamp, str):
                        timestamp = pd.to_datetime(timestamp).timestamp()
                    elif hasattr(timestamp, 'timestamp'):
                        timestamp = timestamp.timestamp()
                    else:
                        timestamp = float(timestamp) if timestamp else 0.0
                else:
                    timestamp = 0.0
                
                transaction_data.append({
                    'user_id': record['user_id'],
                    'amount': float(record['amount']),
                    'timestamp': timestamp
                })
        
        transaction_df = pd.DataFrame(transaction_data) if transaction_data else None
        
        # Extract features
        features_df = self.node_extractor.extract_all_features(
            g,
            node_type='user',
            transaction_df=transaction_df
        )
        
        # Set node IDs as index
        if hasattr(g, 'node_ids'):
            features_df.index = g.node_ids
        
        logger.info(f"Extracted {len(features_df.columns)} node features for {len(features_df)} nodes")
        return features_df
    
    def extract_edge_features(self, g) -> pd.DataFrame:
        """Extract all 12 edge features"""
        logger.info("Extracting edge features...")
        
        edge_features_dict = self.edge_extractor.extract_all_edge_features(
            g,
            edge_type=('user', 'transaction', 'user')
        )
        
        # Convert to DataFrame
        edge_features = {}
        for name, tensor in edge_features_dict.items():
            if hasattr(tensor, 'numpy'):
                edge_features[name] = tensor.numpy()
            else:
                edge_features[name] = tensor
        
        edge_df = pd.DataFrame(edge_features)
        logger.info(f"Extracted {len(edge_df.columns)} edge features for {len(edge_df)} edges")
        return edge_df
    
    def store_in_redis(self, node_features: pd.DataFrame, edge_features: Optional[pd.DataFrame] = None):
        """Store features in Redis"""
        logger.info("Storing features in Redis...")
        
        # Store full node features DataFrame
        if not node_features.empty:
            serialized = pickle.dumps(node_features)
            self.redis_client.set('features:node_features', serialized)
            logger.info(f"Stored node features DataFrame ({len(node_features)} nodes, {len(node_features.columns)} features)")
            
            # Store individual node features for fast lookup
            for node_id, row in node_features.iterrows():
                key = f'features:node:{node_id}'
                node_dict = row.to_dict()
                serialized = pickle.dumps(node_dict)
                self.redis_client.set(key, serialized, ex=86400)  # 24h TTL
            logger.info(f"Stored {len(node_features)} individual node feature records")
            
            # Store metadata
            metadata = {
                'feature_names': node_features.columns.tolist(),
                'node_count': len(node_features),
                'last_update': datetime.now().isoformat(),
                'schema_version': '1.0'
            }
            self.redis_client.set('features:metadata:node', json.dumps(metadata).encode('utf-8'), ex=3600)
        
        # Store full edge features DataFrame
        if edge_features is not None and not edge_features.empty:
            serialized = pickle.dumps(edge_features)
            self.redis_client.set('features:edge_features', serialized)
            logger.info(f"Stored edge features DataFrame ({len(edge_features)} edges, {len(edge_features.columns)} features)")
            
            # Store metadata
            metadata = {
                'feature_names': edge_features.columns.tolist(),
                'edge_count': len(edge_features),
                'last_update': datetime.now().isoformat(),
                'schema_version': '1.0'
            }
            self.redis_client.set('features:metadata:edge', json.dumps(metadata).encode('utf-8'), ex=3600)
    
    def store_in_parquet(self, node_features: pd.DataFrame, edge_features: Optional[pd.DataFrame] = None):
        """Store features in Parquet as backup"""
        logger.info("Storing features in Parquet...")
        
        if not node_features.empty:
            self.feature_store.save_features(node_features, 'node_features')
            logger.info("Saved node features to Parquet")
        
        if edge_features is not None and not edge_features.empty:
            self.feature_store.save_features(edge_features, 'edge_features')
            logger.info("Saved edge features to Parquet")
    
    def run(self):
        """Run complete feature extraction pipeline"""
        try:
            logger.info("=" * 60)
            logger.info("Starting Complete Feature Extraction Pipeline")
            logger.info("=" * 60)
            
            # Step 1: Convert Neo4j to DGL graph
            logger.info("\nStep 1: Loading graph from Neo4j...")
            g = self.neo4j_to_dgl_graph()
            
            if g is None:
                logger.error("Failed to create graph. Check Neo4j data.")
                logger.error("\nTo fix this:")
                logger.error("1. Ensure Neo4j has data: python scripts/sync_postgres_to_neo4j.py")
                logger.error("2. Or manually initialize Neo4j: Open http://localhost:7474 and run scripts/init_neo4j.cypher")
                return False
            
            # Step 2: Extract node features
            logger.info("\nStep 2: Extracting node features...")
            node_features = self.extract_node_features(g)
            
            # Step 3: Extract edge features
            logger.info("\nStep 3: Extracting edge features...")
            edge_features = self.extract_edge_features(g)
            
            # Step 4: Store in Redis
            logger.info("\nStep 4: Storing features in Redis...")
            self.store_in_redis(node_features, edge_features)
            
            # Step 5: Store in Parquet (backup)
            logger.info("\nStep 5: Storing features in Parquet (backup)...")
            self.store_in_parquet(node_features, edge_features)
            
            logger.info("\n" + "=" * 60)
            logger.info("Feature Extraction Complete!")
            logger.info("=" * 60)
            logger.info(f"Node Features: {len(node_features)} nodes × {len(node_features.columns)} features")
            if edge_features is not None:
                logger.info(f"Edge Features: {len(edge_features)} edges × {len(edge_features.columns)} features")
            logger.info("Features stored in Redis and Parquet")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {e}", exc_info=True)
            return False
        finally:
            self.neo4j_driver.close()
            self.redis_client.close()


def main():
    """Main entry point"""
    extractor = CompleteFeatureExtractor()
    success = extractor.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

