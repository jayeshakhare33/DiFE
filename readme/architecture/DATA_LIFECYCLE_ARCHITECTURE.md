# Data Lifecycle Architecture: Distributed Fraud Detection System

## Table of Contents

1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Data Lifecycle - High-Level Flow](#data-lifecycle---high-level-flow)
4. [Low-Level Implementation Details](#low-level-implementation-details)
5. [Component Specifications](#component-specifications)
6. [Kafka Integration](#kafka-integration)
7. [Feature Extraction Pipeline](#feature-extraction-pipeline)
8. [Data Storage Strategy](#data-storage-strategy)
9. [Implementation Workflow](#implementation-workflow)

---

## System Overview

This document describes a distributed fraud detection system that processes transaction data through multiple stages:

1. **Source**: PostgreSQL database containing transaction records
2. **Graph Database**: Neo4j for storing and querying transaction relationships
3. **Feature Extraction**: Distributed workers extracting 62 features (50 node + 12 edge features)
4. **Feature Store**: Redis for fast feature retrieval
5. **Communication**: Apache Kafka for asynchronous, distributed message passing

### Key Components

- **PostgreSQL**: Transactional database (source of truth)
- **Neo4j**: Graph database (relationship storage and graph queries)
- **Kafka**: Message broker (distributed communication)
- **Redis**: Feature store (high-speed feature cache)
- **Feature Extractors**: Distributed workers (compute features from graph)
- **GNN Model**: Graph Neural Network (fraud detection)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA LIFECYCLE ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│ PostgreSQL   │  Transaction Database (Source of Truth)
│  Database    │  - Transactions table
│              │  - Users table
│              │  - Locations table
└──────┬───────┘
       │
       │ [CDC / Batch Sync]
       │ Kafka Topic: transactions-raw
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          KAFKA MESSAGE BROKER                               │
│                                                                             │
│  Topics:                                                                    │
│  ├── transactions-raw      (Raw transaction data from PostgreSQL)           │
│  ├── graph-updates         (Graph structure updates for Neo4j)              │
│  ├── feature-extraction    (Feature extraction requests)                    │
│  ├── features-computed     (Computed features from extractors)              │
│  └── features-stored       (Confirmation of Redis storage)                  │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       │ [Consume: transactions-raw]
       ▼
┌──────────────┐
│   Neo4j      │  Graph Database
│              │  - User nodes
│              │  - Transaction nodes
│              │  - Relationship edges
│              │  - Entity nodes (cards, devices, locations)
└──────┬───────┘
       │
       │ [Graph Query / Subgraph Extraction]
       │ Kafka Topic: feature-extraction
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED FEATURE EXTRACTION WORKERS                   │
│                                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │Worker-0  │  │Worker-1  │  │Worker-2  │  │Worker-3  │                     │
│  │          │  │          │  │          │  │          │                     │
│  │ Extract: │  │ Extract: │  │ Extract: │  │ Extract: │                     │
│  │ - Node   │  │ - Node   │  │ - Node   │  │ - Node   │                     │
│  │  Features│  │  Features│  │  Features│  │  Features│                     │
│  │ - Edge   │  │ - Edge   │  │ - Edge   │  │ - Edge   │                     │
│  │  Features│  │  Features│  │  Features│  │  Features│                     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘                     │
│       │             │             │             │                           │
│       └─────────────┴─────────────┴─────────────┘                           │
│                    │                                                        │
│                    │ [Publish: features-computed]                           │
└────────────────────┼────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KAFKA CONSUMER (Feature Aggregator)                      │
│                                                                             │
│  - Consumes: features-computed                                              │
│  - Aggregates features from multiple workers                                │
│  - Deduplicates and merges feature DataFrames                               │
│  - Stores aggregated features in Redis                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────┐
│    Redis     │  Feature Store
│              │  - node_features (50 features per node)
│              │  - edge_features (12 features per edge)
│              │  - Feature lookups by node_id / edge_id
└──────────────┘
       │
       │ [Feature Retrieval]
       ▼
┌──────────────┐
│  GNN Model   │  Fraud Detection
│  / API       │  - Load features from Redis
│              │  - Real-time inference
└──────────────┘
```

---

## Data Lifecycle - High-Level Flow

### Phase 1: Data Ingestion (PostgreSQL → Kafka → Neo4j)

**Objective**: Move transaction data from PostgreSQL to Neo4j graph database

1. **PostgreSQL Transaction Events**
   - New transactions are inserted/updated in PostgreSQL
   - Change Data Capture (CDC) or scheduled batch jobs detect changes
   - Transaction records include: sender_id, receiver_id, amount, timestamp, location, device, etc.

2. **Kafka Producer (PostgreSQL Connector)**
   - Publishes raw transaction data to `transactions-raw` topic
   - Message format: JSON with transaction fields
   - Partitioning: By transaction timestamp or sender_id

3. **Kafka Consumer (Neo4j Loader)**
   - Consumes from `transactions-raw` topic
   - Transforms transaction data into Neo4j graph structure
   - Creates/updates nodes and relationships in Neo4j

4. **Neo4j Graph Structure**
   - **Nodes**: Users, Transactions, Locations, Devices, Cards
   - **Relationships**: 
     - `(User)-[:SENT]->(Transaction)`
     - `(Transaction)-[:TO]->(User)`
     - `(User)-[:LOCATED_AT]->(Location)`
     - `(Transaction)-[:USED_DEVICE]->(Device)`
     - `(User)-[:HAS_CARD]->(Card)`

### Phase 2: Feature Extraction Trigger (Neo4j → Kafka)

**Objective**: Trigger feature extraction when graph updates occur

1. **Graph Update Detection**
   - Neo4j triggers or scheduled jobs detect graph changes
   - Identifies affected nodes/subgraphs requiring feature recomputation

2. **Kafka Producer (Feature Extraction Trigger)**
   - Publishes feature extraction requests to `feature-extraction` topic
   - Message contains: node_ids, subgraph_ids, extraction_type (node/edge/both)

### Phase 3: Distributed Feature Extraction (Kafka → Workers → Kafka)

**Objective**: Extract 62 features (50 node + 12 edge) from graph subgraphs

1. **Kafka Consumer (Feature Extraction Workers)**
   - Multiple workers consume from `feature-extraction` topic
   - Each worker processes a partition of the topic
   - Workers query Neo4j for subgraph data

2. **Feature Extraction Process**
   - **Node Features (50 total)**:
     - Transaction Statistics (15): volumes, amounts, frequencies
     - Graph Topology (12): centrality, clustering, degrees
     - Temporal (10): time patterns, activity windows
     - Behavioral (8): round amounts, thresholds, modes
     - Fraud Propagation (5): proximity to fraud, network connections
   
   - **Edge Features (12 total)**:
     - Amount, timestamp, temporal patterns
     - Relationship features (reciprocal, time gaps)
     - Geographic and percentile features

3. **Kafka Producer (Feature Workers)**
   - Publishes computed features to `features-computed` topic
   - Message format: DataFrame serialized (JSON or binary)
   - Includes metadata: worker_id, node_ids, feature_names, timestamp

### Phase 4: Feature Aggregation & Storage (Kafka → Redis)

**Objective**: Aggregate features from multiple workers and store in Redis

1. **Kafka Consumer (Feature Aggregator)**
   - Consumes from `features-computed` topic
   - Aggregates features from multiple workers
   - Deduplicates by node_id/edge_id (keeps latest)
   - Merges DataFrames

2. **Redis Storage**
   - Stores aggregated features in Redis
   - Key structure: `features:node:{node_id}` or `features:edge:{edge_id}`
   - Alternative: Store full DataFrames as `features:node_features` and `features:edge_features`
   - TTL: Optional expiration for cache management

### Phase 5: Feature Consumption (Redis → GNN/API)

**Objective**: Use stored features for fraud detection

1. **Feature Retrieval**
   - GNN training: Bulk load from Redis
   - Real-time inference: Lookup specific node/edge features
   - API endpoints: Query features by node_id

2. **Model Inference**
   - Load features from Redis
   - Pass to GNN model for fraud prediction
   - Return predictions via API

---

## Low-Level Implementation Details

### 1. PostgreSQL Schema

```sql
-- Transactions table
CREATE TABLE transactions (
    transaction_id BIGSERIAL PRIMARY KEY,
    sender_id VARCHAR(50) NOT NULL,
    receiver_id VARCHAR(50) NOT NULL,
    amount DECIMAL(15, 2) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    location_id VARCHAR(50),
    device_id VARCHAR(50),
    card_id VARCHAR(50),
    is_fraud BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Users table
CREATE TABLE users (
    user_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Locations table
CREATE TABLE locations (
    location_id VARCHAR(50) PRIMARY KEY,
    country VARCHAR(100),
    city VARCHAR(100),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8)
);
```

### 2. PostgreSQL → Kafka Connector

**Implementation Options:**

**Option A: Debezium CDC Connector**
```yaml
# Debezium PostgreSQL Connector Configuration
connector.class: io.debezium.connector.postgresql.PostgresConnector
database.hostname: postgres
database.port: 5432
database.user: postgres
database.password: password
database.dbname: fraud_detection
database.server.name: fraud-db
table.whitelist: public.transactions
topic.prefix: fraud-db
```

**Option B: Custom Producer Script**
```python
# PostgreSQL to Kafka Producer
import psycopg2
from kafka import KafkaProducer
import json
from datetime import datetime

def sync_transactions_to_kafka():
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host='postgres',
        database='fraud_detection',
        user='postgres',
        password='password'
    )
    
    # Connect to Kafka
    producer = KafkaProducer(
        bootstrap_servers=['kafka:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    # Query new/updated transactions
    cursor = conn.cursor()
    cursor.execute("""
        SELECT transaction_id, sender_id, receiver_id, amount, 
               timestamp, location_id, device_id, card_id, is_fraud
        FROM transactions
        WHERE updated_at > NOW() - INTERVAL '1 minute'
        ORDER BY timestamp
    """)
    
    for row in cursor:
        transaction = {
            'transaction_id': str(row[0]),
            'sender_id': row[1],
            'receiver_id': row[2],
            'amount': float(row[3]),
            'timestamp': row[4].isoformat(),
            'location_id': row[5],
            'device_id': row[6],
            'card_id': row[7],
            'is_fraud': row[8]
        }
        
        # Publish to Kafka
        producer.send('transactions-raw', value=transaction, 
                     key=transaction['transaction_id'].encode())
    
    producer.flush()
    conn.close()
```

### 3. Kafka Topic: `transactions-raw`

**Message Format:**
```json
{
  "transaction_id": "12345",
  "sender_id": "user_001",
  "receiver_id": "user_002",
  "amount": 1500.50,
  "timestamp": "2024-01-15T10:30:00Z",
  "location_id": "loc_001",
  "device_id": "dev_001",
  "card_id": "card_001",
  "is_fraud": false,
  "metadata": {
    "source": "postgresql",
    "sync_timestamp": "2024-01-15T10:30:05Z"
  }
}
```

**Topic Configuration:**
- Partitions: 6 (for parallel processing)
- Replication Factor: 3
- Retention: 7 days
- Compression: gzip

### 4. Neo4j Graph Structure

**Cypher Queries for Graph Construction:**

```cypher
// Create User nodes
MERGE (u:User {user_id: $sender_id})
MERGE (v:User {user_id: $receiver_id})

// Create Transaction node
CREATE (t:Transaction {
    transaction_id: $transaction_id,
    amount: $amount,
    timestamp: $timestamp,
    is_fraud: $is_fraud
})

// Create relationships
CREATE (u)-[:SENT {
    amount: $amount,
    timestamp: $timestamp
}]->(t)
CREATE (t)-[:TO {
    amount: $amount,
    timestamp: $timestamp
}]->(v)

// Create Location node and relationship
MERGE (l:Location {location_id: $location_id})
CREATE (u)-[:LOCATED_AT {
    timestamp: $timestamp
}]->(l)

// Create Device node and relationship
MERGE (d:Device {device_id: $device_id})
CREATE (t)-[:USED_DEVICE]->(d)

// Create Card node and relationship
MERGE (c:Card {card_id: $card_id})
CREATE (u)-[:HAS_CARD]->(c)
```

**Neo4j Indexes:**
```cypher
CREATE INDEX user_id_index FOR (u:User) ON (u.user_id);
CREATE INDEX transaction_id_index FOR (t:Transaction) ON (t.transaction_id);
CREATE INDEX transaction_timestamp_index FOR (t:Transaction) ON (t.timestamp);
```

### 5. Kafka Consumer: Neo4j Loader

```python
from kafka import KafkaConsumer
from neo4j import GraphDatabase
import json

class Neo4jGraphLoader:
    def __init__(self, kafka_servers, neo4j_uri, neo4j_user, neo4j_password):
        self.consumer = KafkaConsumer(
            'transactions-raw',
            bootstrap_servers=kafka_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='neo4j-loader-group',
            enable_auto_commit=True
        )
        self.driver = GraphDatabase.driver(neo4j_uri, 
                                          auth=(neo4j_user, neo4j_password))
    
    def process_transaction(self, transaction):
        with self.driver.session() as session:
            session.run("""
                MERGE (sender:User {user_id: $sender_id})
                MERGE (receiver:User {user_id: $receiver_id})
                CREATE (txn:Transaction {
                    transaction_id: $transaction_id,
                    amount: $amount,
                    timestamp: datetime($timestamp),
                    is_fraud: $is_fraud
                })
                CREATE (sender)-[:SENT {
                    amount: $amount,
                    timestamp: datetime($timestamp)
                }]->(txn)
                CREATE (txn)-[:TO {
                    amount: $amount,
                    timestamp: datetime($timestamp)
                }]->(receiver)
                
                WITH txn, sender, receiver
                OPTIONAL MATCH (loc:Location {location_id: $location_id})
                WHERE $location_id IS NOT NULL
                MERGE (loc:Location {location_id: $location_id})
                CREATE (sender)-[:LOCATED_AT {
                    timestamp: datetime($timestamp)
                }]->(loc)
                
                WITH txn
                OPTIONAL MATCH (dev:Device {device_id: $device_id})
                WHERE $device_id IS NOT NULL
                MERGE (dev:Device {device_id: $device_id})
                CREATE (txn)-[:USED_DEVICE]->(dev)
            """, **transaction)
    
    def run(self):
        for message in self.consumer:
            try:
                self.process_transaction(message.value)
                # Trigger feature extraction for affected nodes
                self.trigger_feature_extraction(message.value)
            except Exception as e:
                print(f"Error processing message: {e}")
    
    def trigger_feature_extraction(self, transaction):
        # Publish to feature-extraction topic
        # (Implementation in next section)
        pass
```

### 6. Kafka Topic: `feature-extraction`

**Message Format:**
```json
{
  "extraction_id": "ext_001",
  "type": "node_and_edge",
  "node_ids": ["user_001", "user_002"],
  "subgraph_query": "MATCH (u:User)-[*1..2]-(n) WHERE u.user_id IN $node_ids RETURN n",
  "priority": "high",
  "timestamp": "2024-01-15T10:30:00Z",
  "metadata": {
    "triggered_by": "transaction_insert",
    "transaction_id": "12345"
  }
}
```

### 7. Feature Extraction Worker

```python
from kafka import KafkaConsumer, KafkaProducer
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from feature_engineering.feature_extractor import FeatureExtractor, EdgeFeatureExtractor

class DistributedFeatureExtractor:
    def __init__(self, worker_id, kafka_servers, neo4j_uri, neo4j_user, neo4j_password):
        self.worker_id = worker_id
        
        # Kafka consumer for extraction requests
        self.consumer = KafkaConsumer(
            'feature-extraction',
            bootstrap_servers=kafka_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='feature-extractor-group',
            enable_auto_commit=True
        )
        
        # Kafka producer for computed features
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
            compression_type='gzip'
        )
        
        # Neo4j connection
        self.driver = GraphDatabase.driver(neo4j_uri, 
                                          auth=(neo4j_user, neo4j_password))
        
        # Feature extractors
        self.node_extractor = FeatureExtractor()
        self.edge_extractor = EdgeFeatureExtractor()
    
    def extract_node_features(self, node_ids, graph_data):
        """Extract 50 node features"""
        # Convert Neo4j graph to DGL format
        dgl_graph = self._neo4j_to_dgl(graph_data)
        
        # Extract features
        features_df = self.node_extractor.extract_all_features(
            dgl_graph, 
            node_type='user',
            transaction_df=None  # Can be populated from Neo4j if needed
        )
        
        return features_df
    
    def extract_edge_features(self, edge_data):
        """Extract 12 edge features"""
        dgl_graph = self._neo4j_to_dgl(edge_data)
        
        edge_features = self.edge_extractor.extract_all_edge_features(
            dgl_graph,
            edge_type='transaction'
        )
        
        # Convert to DataFrame
        edge_df = pd.DataFrame({
            k: v.numpy() if hasattr(v, 'numpy') else v
            for k, v in edge_features.items()
        })
        
        return edge_df
    
    def _neo4j_to_dgl(self, graph_data):
        """Convert Neo4j graph to DGL format"""
        # Implementation: Query Neo4j and convert to DGL graph
        # This is a simplified version
        import dgl
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)-[r]->(m)
                WHERE n.user_id IN $node_ids OR m.user_id IN $node_ids
                RETURN n, r, m
            """, node_ids=graph_data['node_ids'])
            
            # Build DGL graph from results
            # (Full implementation would construct nodes and edges)
            pass
    
    def process_extraction_request(self, request):
        """Process a feature extraction request"""
        extraction_id = request['extraction_id']
        node_ids = request['node_ids']
        extraction_type = request['type']
        
        # Query Neo4j for subgraph
        with self.driver.session() as session:
            # Get subgraph data
            graph_data = self._get_subgraph(session, node_ids)
        
        # Extract features
        node_features = None
        edge_features = None
        
        if extraction_type in ['node', 'node_and_edge']:
            node_features = self.extract_node_features(node_ids, graph_data)
        
        if extraction_type in ['edge', 'node_and_edge']:
            edge_features = self.extract_edge_features(graph_data)
        
        # Publish to Kafka
        self.publish_features(extraction_id, node_features, edge_features)
    
    def publish_features(self, extraction_id, node_features, edge_features):
        """Publish computed features to Kafka"""
        message = {
            'extraction_id': extraction_id,
            'worker_id': self.worker_id,
            'node_features': node_features.to_dict('records') if node_features is not None else None,
            'edge_features': edge_features.to_dict('records') if edge_features is not None else None,
            'node_feature_names': node_features.columns.tolist() if node_features is not None else [],
            'edge_feature_names': edge_features.columns.tolist() if edge_features is not None else [],
            'timestamp': datetime.now().isoformat()
        }
        
        self.producer.send('features-computed', value=message)
        self.producer.flush()
    
    def run(self):
        """Main worker loop"""
        for message in self.consumer:
            try:
                self.process_extraction_request(message.value)
            except Exception as e:
                print(f"Error in worker {self.worker_id}: {e}")
```

### 8. Kafka Topic: `features-computed`

**Message Format:**
```json
{
  "extraction_id": "ext_001",
  "worker_id": "worker-0",
  "node_features": [
    {
      "node_id": "user_001",
      "total_transactions_sent": 45,
      "total_transactions_received": 32,
      "avg_transaction_amount_sent": 1250.50,
      "in_degree": 32,
      "out_degree": 45,
      "degree_centrality": 0.15,
      "pagerank_score": 0.0023,
      "transactions_last_24h": 5,
      "round_amount_ratio": 0.12,
      "connected_to_fraud_count": 2,
      ... // All 50 node features
    }
  ],
  "edge_features": [
    {
      "edge_id": "edge_001",
      "amount": 1500.50,
      "timestamp": "2024-01-15T10:30:00Z",
      "hour_of_day": 10,
      "day_of_week": 1,
      "is_weekend": false,
      "time_since_last_between_users": 3600,
      "amount_percentile_sender": 0.75,
      "is_reciprocal": false,
      ... // All 12 edge features
    }
  ],
  "node_feature_names": ["total_transactions_sent", "in_degree", ...],
  "edge_feature_names": ["amount", "timestamp", ...],
  "timestamp": "2024-01-15T10:31:00Z"
}
```

### 9. Feature Aggregator & Redis Storage

```python
from kafka import KafkaConsumer
import redis
import pandas as pd
import json
from collections import defaultdict

class FeatureAggregator:
    def __init__(self, kafka_servers, redis_host, redis_port, batch_size=1000):
        # Kafka consumer
        self.consumer = KafkaConsumer(
            'features-computed',
            bootstrap_servers=kafka_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='feature-aggregator-group',
            enable_auto_commit=True
        )
        
        # Redis connection
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=False
        )
        
        # Batch aggregation
        self.node_features_batch = defaultdict(dict)
        self.edge_features_batch = []
        self.batch_size = batch_size
    
    def aggregate_and_store(self):
        """Aggregate features and store in Redis"""
        for message in self.consumer:
            try:
                data = message.value
                
                # Process node features
                if data['node_features']:
                    for node_feature in data['node_features']:
                        node_id = node_feature['node_id']
                        # Update or add node features (deduplicate by keeping latest)
                        self.node_features_batch[node_id].update(node_feature)
                
                # Process edge features
                if data['edge_features']:
                    self.edge_features_batch.extend(data['edge_features'])
                
                # Batch write to Redis
                if len(self.node_features_batch) >= self.batch_size:
                    self.flush_to_redis()
            
            except Exception as e:
                print(f"Error processing features: {e}")
    
    def flush_to_redis(self):
        """Flush aggregated features to Redis"""
        import pickle
        
        # Convert node features batch to DataFrame
        if self.node_features_batch:
            node_df = pd.DataFrame(list(self.node_features_batch.values()))
            node_df.set_index('node_id', inplace=True)
            
            # Store in Redis
            serialized = pickle.dumps(node_df)
            self.redis_client.set('features:node_features', serialized)
            
            # Also store individual node features for fast lookup
            for node_id, features in self.node_features_batch.items():
                key = f'features:node:{node_id}'
                serialized = pickle.dumps(features)
                self.redis_client.set(key, serialized, ex=86400)  # 24h TTL
            
            self.node_features_batch.clear()
        
        # Convert edge features batch to DataFrame
        if self.edge_features_batch:
            edge_df = pd.DataFrame(self.edge_features_batch)
            
            # Store in Redis
            serialized = pickle.dumps(edge_df)
            self.redis_client.set('features:edge_features', serialized)
            
            self.edge_features_batch.clear()
        
        print(f"Flushed features to Redis")
    
    def run(self):
        """Main aggregator loop"""
        try:
            self.aggregate_and_store()
        except KeyboardInterrupt:
            # Flush remaining batch
            self.flush_to_redis()
```

### 10. Redis Storage Structure

**Key Patterns:**

```
# Full feature DataFrames
features:node_features     -> Pickled DataFrame (all nodes, 50 features)
features:edge_features     -> Pickled DataFrame (all edges, 12 features)

# Individual node features (for fast lookup)
features:node:user_001     -> Pickled dict (50 features for user_001)
features:node:user_002     -> Pickled dict (50 features for user_002)
...

# Feature metadata
features:metadata:node      -> JSON (feature names, last_update, count)
features:metadata:edge      -> JSON (feature names, last_update, count)
```

**Redis Commands:**
```bash
# Get all node features
GET features:node_features

# Get specific node features
GET features:node:user_001

# Check if features exist
EXISTS features:node_features

# Get feature metadata
GET features:metadata:node
```

### 11. Feature Retrieval for GNN/API

```python
import redis
import pickle
import pandas as pd

class FeatureRetriever:
    def __init__(self, redis_host, redis_port):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=False
        )
    
    def get_node_features(self, node_ids):
        """Get node features for specific nodes"""
        if isinstance(node_ids, str):
            node_ids = [node_ids]
        
        features_list = []
        for node_id in node_ids:
            key = f'features:node:{node_id}'
            serialized = self.redis_client.get(key)
            
            if serialized:
                features = pickle.loads(serialized)
                features_list.append(features)
            else:
                # Fallback: Load from full DataFrame
                full_df = self.get_all_node_features()
                if node_id in full_df.index:
                    features_list.append(full_df.loc[node_id].to_dict())
        
        return pd.DataFrame(features_list)
    
    def get_all_node_features(self):
        """Get all node features as DataFrame"""
        serialized = self.redis_client.get('features:node_features')
        if serialized:
            return pickle.loads(serialized)
        return pd.DataFrame()
    
    def get_edge_features(self, edge_ids=None):
        """Get edge features"""
        serialized = self.redis_client.get('features:edge_features')
        if serialized:
            df = pickle.loads(serialized)
            if edge_ids:
                return df[df['edge_id'].isin(edge_ids)]
            return df
        return pd.DataFrame()
```

---

## Component Specifications

### PostgreSQL Database

**Purpose**: Source of truth for transaction data

**Tables**:
- `transactions`: Transaction records
- `users`: User information
- `locations`: Geographic data
- `devices`: Device information
- `cards`: Payment card data

**Sync Strategy**:
- **CDC (Change Data Capture)**: Real-time sync using Debezium
- **Batch Sync**: Scheduled jobs (every 1-5 minutes)
- **Hybrid**: CDC for critical updates, batch for bulk loads

### Neo4j Graph Database

**Purpose**: Store transaction relationships and enable graph queries

**Node Types**:
- `User`: User accounts
- `Transaction`: Individual transactions
- `Location`: Geographic locations
- `Device`: Devices used for transactions
- `Card`: Payment cards

**Relationship Types**:
- `SENT`: User sent transaction
- `TO`: Transaction to user
- `LOCATED_AT`: User at location
- `USED_DEVICE`: Transaction used device
- `HAS_CARD`: User has card

**Query Patterns**:
- Subgraph extraction for feature computation
- Path queries for fraud propagation
- Community detection for fraud rings

### Apache Kafka

**Purpose**: Asynchronous message passing between components

**Topics**:

1. **`transactions-raw`**
   - **Producers**: PostgreSQL connector
   - **Consumers**: Neo4j loader
   - **Partitions**: 6
   - **Retention**: 7 days

2. **`graph-updates`**
   - **Producers**: Neo4j loader
   - **Consumers**: Feature extraction trigger
   - **Partitions**: 3
   - **Retention**: 1 day

3. **`feature-extraction`**
   - **Producers**: Feature extraction trigger
   - **Consumers**: Feature extraction workers
   - **Partitions**: 4 (one per worker)
   - **Retention**: 1 day

4. **`features-computed`**
   - **Producers**: Feature extraction workers
   - **Consumers**: Feature aggregator
   - **Partitions**: 4
   - **Retention**: 1 day

5. **`features-stored`**
   - **Producers**: Feature aggregator
   - **Consumers**: Monitoring/audit services
   - **Partitions**: 1
   - **Retention**: 30 days

### Redis Feature Store

**Purpose**: High-speed feature storage and retrieval

**Data Structure**:
- **Full DataFrames**: Complete feature sets (for bulk operations)
- **Individual Records**: Per-node/per-edge features (for lookups)
- **Metadata**: Feature schemas, update timestamps

**TTL Strategy**:
- Individual node features: 24 hours
- Full DataFrames: No expiration (manual refresh)
- Metadata: 1 hour

**Memory Management**:
- Estimated size: ~100-200 KB per 1000 nodes
- Use Redis eviction policy: `allkeys-lru` if memory constrained

### Feature Extraction Workers

**Purpose**: Compute 62 features from graph data

**Worker Configuration**:
- Number of workers: 4 (configurable)
- Each worker processes one Kafka partition
- Workers are stateless (can be scaled horizontally)

**Feature Categories**:
1. **Node Features (50)**:
   - Transaction Statistics (15)
   - Graph Topology (12)
   - Temporal (10)
   - Behavioral (8)
   - Fraud Propagation (5)

2. **Edge Features (12)**:
   - Transaction attributes
   - Temporal patterns
   - Relationship features

**Processing Flow**:
1. Consume extraction request from Kafka
2. Query Neo4j for subgraph
3. Convert Neo4j graph to DGL format
4. Extract features using FeatureExtractor
5. Publish features to Kafka

---

## Kafka Integration

### Message Flow Diagram

```
PostgreSQL → [transactions-raw] → Neo4j Loader
                                        ↓
                              [graph-updates]
                                        ↓
                          Feature Extraction Trigger
                                        ↓
                          [feature-extraction]
                                        ↓
                    ┌───────────────────┼───────────────────┐
                    ↓                   ↓                   ↓
              Worker-0            Worker-1            Worker-2
                    ↓                   ↓                   ↓
              [features-computed]  [features-computed]  [features-computed]
                    └───────────────────┼───────────────────┘
                                        ↓
                              Feature Aggregator
                                        ↓
                                    Redis Store
```

### Kafka Consumer Groups

1. **`neo4j-loader-group`**
   - Consumers: Neo4j loader instances
   - Topic: `transactions-raw`
   - Parallelism: 1-3 consumers

2. **`feature-extractor-group`**
   - Consumers: Feature extraction workers
   - Topic: `feature-extraction`
   - Parallelism: 4 workers (one per partition)

3. **`feature-aggregator-group`**
   - Consumers: Feature aggregator instances
   - Topic: `features-computed`
   - Parallelism: 1-2 consumers (for redundancy)

### Message Serialization

**Format**: JSON (human-readable, debuggable)

**Compression**: gzip (for large feature payloads)

**Key Strategy**:
- `transactions-raw`: Transaction ID
- `feature-extraction`: Extraction ID
- `features-computed`: Worker ID + Extraction ID

---

## Feature Extraction Pipeline

### Node Feature Extraction (50 features)

**Input**: Neo4j subgraph (user nodes and relationships)

**Process**:

1. **Transaction Statistics (15 features)**
   ```python
   # Query Neo4j for transaction data
   MATCH (u:User)-[:SENT]->(t:Transaction)
   WHERE u.user_id IN $node_ids
   RETURN u.user_id, t.amount, t.timestamp
   
   # Compute statistics
   - total_transactions_sent
   - avg_transaction_amount_sent
   - max_transaction_amount_sent
   - std_transaction_amount_sent
   - total_amount_sent
   - ... (15 total)
   ```

2. **Graph Topology (12 features)**
   ```python
   # Query Neo4j for graph structure
   MATCH (u:User)-[r]-(n)
   WHERE u.user_id IN $node_ids
   RETURN u, r, n
   
   # Compute centrality measures
   - in_degree, out_degree, total_degree
   - degree_centrality
   - betweenness_centrality (if graph small enough)
   - closeness_centrality
   - pagerank_score
   - clustering_coefficient
   - katz_centrality
   - eigenvector_centrality
   - average_neighbor_degree
   - triangles_count
   ```

3. **Temporal Features (10 features)**
   ```python
   # Query Neo4j for temporal patterns
   MATCH (u:User)-[:SENT|TO]->(t:Transaction)
   WHERE u.user_id IN $node_ids
   RETURN u.user_id, t.timestamp
   ORDER BY t.timestamp
   
   # Compute temporal features
   - account_age_days
   - first_transaction_timestamp
   - last_transaction_timestamp
   - time_since_last_transaction
   - transactions_last_24h
   - transactions_last_7d
   - transactions_last_30d
   - hour_of_day_mode
   - day_of_week_mode
   - transaction_time_variance
   ```

4. **Behavioral Features (8 features)**
   ```python
   # Query Neo4j for transaction amounts and modes
   MATCH (u:User)-[:SENT]->(t:Transaction)
   WHERE u.user_id IN $node_ids
   RETURN u.user_id, t.amount, t.device_id, t.location_id
   
   # Compute behavioral features
   - round_amount_ratio
   - threshold_amount_ratio
   - transaction_mode_diversity
   - failed_transaction_ratio
   - reversal_ratio
   - cross_border_ratio
   - high_risk_country_ratio
   - burst_score
   ```

5. **Fraud Propagation (5 features)**
   ```python
   # Query Neo4j for fraud connections
   MATCH path = (u:User)-[*1..3]-(f:User {is_fraud: true})
   WHERE u.user_id IN $node_ids
   RETURN u, path, f
   
   # Compute fraud propagation features
   - connected_to_fraud_count
   - fraud_propagation_score
   - distance_to_nearest_fraud
   - common_neighbors_with_frauds
   - fraud_cluster_membership
   ```

**Output**: DataFrame with 50 columns (features) and N rows (nodes)

### Edge Feature Extraction (12 features)

**Input**: Neo4j transaction edges

**Process**:

```python
# Query Neo4j for edge data
MATCH (u1:User)-[:SENT]->(t:Transaction)-[:TO]->(u2:User)
WHERE u1.user_id IN $node_ids OR u2.user_id IN $node_ids
RETURN u1, t, u2

# Extract edge features
- amount
- timestamp
- hour_of_day
- day_of_week
- is_weekend
- transaction_mode
- time_since_last_between_users
- amount_percentile_sender
- amount_percentile_receiver
- is_reciprocal
- reciprocal_time_gap
- geographic_distance
```

**Output**: DataFrame with 12 columns (features) and M rows (edges)

---

## Data Storage Strategy

### Redis Storage Schema

**Full Feature Sets** (for bulk operations):
```
Key: features:node_features
Value: Pickled pandas DataFrame
  - Index: node_id
  - Columns: 50 feature columns
  - Size: ~100-200 KB per 1000 nodes

Key: features:edge_features
Value: Pickled pandas DataFrame
  - Index: edge_id (or auto-increment)
  - Columns: 12 feature columns
  - Size: ~50-100 KB per 1000 edges
```

**Individual Features** (for fast lookups):
```
Key: features:node:{node_id}
Value: Pickled dict
  - Keys: feature names (50 features)
  - Values: feature values
  - TTL: 24 hours

Key: features:edge:{edge_id}
Value: Pickled dict
  - Keys: feature names (12 features)
  - Values: feature values
  - TTL: 24 hours
```

**Metadata**:
```
Key: features:metadata:node
Value: JSON
{
  "feature_names": ["total_transactions_sent", ...],
  "last_update": "2024-01-15T10:30:00Z",
  "node_count": 10000,
  "schema_version": "1.0"
}

Key: features:metadata:edge
Value: JSON
{
  "feature_names": ["amount", ...],
  "last_update": "2024-01-15T10:30:00Z",
  "edge_count": 50000,
  "schema_version": "1.0"
}
```

### Feature Update Strategy

**Incremental Updates**:
- When new transactions arrive, only affected nodes/edges are re-extracted
- Features are updated in Redis (overwrite existing)
- Metadata is updated with new timestamp

**Full Refresh**:
- Scheduled daily/weekly full feature recomputation
- All features are re-extracted and stored
- Used for graph topology features that depend on full graph

**TTL Management**:
- Individual features: 24h TTL (auto-expire if not updated)
- Full DataFrames: No TTL (manual management)
- Metadata: 1h TTL (refresh frequently)

---

## Implementation Workflow

### Step 1: Setup Infrastructure

1. **Deploy PostgreSQL**
   - Create database and tables
   - Configure CDC connector or batch sync script

2. **Deploy Neo4j**
   - Create indexes
   - Configure connection settings

3. **Deploy Kafka**
   - Create topics with appropriate partitions
   - Configure retention policies

4. **Deploy Redis**
   - Configure memory limits
   - Set eviction policy

### Step 2: Data Ingestion Pipeline

1. **PostgreSQL → Kafka**
   - Deploy Debezium connector or custom producer
   - Verify messages in `transactions-raw` topic

2. **Kafka → Neo4j**
   - Deploy Neo4j loader consumer
   - Verify graph structure in Neo4j

### Step 3: Feature Extraction Pipeline

1. **Graph Update Detection**
   - Deploy feature extraction trigger
   - Publish to `feature-extraction` topic

2. **Feature Extraction Workers**
   - Deploy 4 worker instances
   - Each worker consumes from one partition
   - Workers query Neo4j and extract features

3. **Feature Aggregation**
   - Deploy feature aggregator
   - Consumes from `features-computed` topic
   - Aggregates and stores in Redis

### Step 4: Feature Consumption

1. **GNN Training**
   - Load features from Redis
   - Train model with 50 node + 12 edge features

2. **Real-time Inference**
   - API queries Redis for node/edge features
   - Pass features to GNN model
   - Return predictions

### Step 5: Monitoring & Optimization

1. **Kafka Lag Monitoring**
   - Monitor consumer lag
   - Scale workers if lag increases

2. **Redis Memory Monitoring**
   - Monitor memory usage
   - Adjust TTL if needed

3. **Feature Update Latency**
   - Measure time from transaction to feature update
   - Optimize slow components

---

## Performance Considerations

### Throughput Estimates

- **PostgreSQL → Kafka**: 10,000 transactions/second
- **Kafka → Neo4j**: 5,000 transactions/second (graph operations are slower)
- **Feature Extraction**: 1,000 nodes/second per worker (4 workers = 4,000 nodes/second)
- **Redis Storage**: 10,000 writes/second
- **Redis Retrieval**: 50,000 reads/second

### Latency Estimates

- **Transaction to Graph**: 1-5 seconds
- **Graph to Feature Extraction Trigger**: <1 second
- **Feature Extraction**: 10-30 seconds per batch (depends on graph size)
- **Feature Storage**: <1 second
- **Feature Retrieval**: <10ms (Redis lookup)

### Scalability

- **Horizontal Scaling**: Add more Kafka partitions and workers
- **Vertical Scaling**: Increase Neo4j/Redis memory
- **Caching**: Redis caching reduces Neo4j query load
- **Batch Processing**: Batch feature updates to reduce Redis writes

---

## Error Handling & Fault Tolerance

### Kafka

- **Message Retry**: Automatic retry on consumer errors
- **Dead Letter Queue**: Failed messages go to DLQ for manual review
- **Idempotent Producers**: Prevent duplicate messages

### Neo4j

- **Connection Retry**: Retry on connection failures
- **Transaction Rollback**: Rollback on errors
- **Query Timeout**: Timeout long-running queries

### Redis

- **Connection Pooling**: Reuse connections
- **Failover**: Redis Sentinel for high availability
- **Data Persistence**: RDB snapshots + AOF for durability

### Feature Extraction

- **Worker Failover**: If worker fails, Kafka rebalances partitions
- **Partial Results**: Store partial features if extraction fails
- **Retry Logic**: Retry failed extractions

---

## Conclusion

This architecture provides a scalable, distributed system for fraud detection that:

1. **Ingests** transaction data from PostgreSQL
2. **Stores** relationships in Neo4j graph database
3. **Extracts** 62 features (50 node + 12 edge) using distributed workers
4. **Stores** features in Redis for fast retrieval
5. **Uses** Kafka for asynchronous, fault-tolerant communication

The system is designed for:
- **High Throughput**: Process thousands of transactions per second
- **Low Latency**: Sub-second feature updates for real-time detection
- **Scalability**: Horizontal scaling of workers and Kafka partitions
- **Fault Tolerance**: Automatic recovery from component failures
- **Flexibility**: Easy to add new features or modify extraction logic

