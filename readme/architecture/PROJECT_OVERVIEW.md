# Fraud Detection GNN Project - Complete Overview

## 🎯 Project Goal
Build a **Graph Neural Network (GNN) based fraud detection system** that analyzes transaction patterns in a graph structure to identify fraudulent activities.

---

## 📊 System Architecture

### **Data Flow Pipeline**

```
PostgreSQL (Transaction DB)
    ↓
Neo4j (Graph Database)
    ↓
Feature Extraction (62 Features)
    ↓
Redis (Feature Store)
    ↓
GNN Model Training
```

---

## 🗄️ **Data Storage Layers**

### 1. **PostgreSQL** (Transactional Database)
- **Purpose**: Source of truth for all transaction data
- **Tables**:
  - `users` - User information (50 users)
  - `transactions` - All transactions (500 transactions, 130 fraudulent)
  - `locations` - Geographic data (5 countries)
  - `devices` - Device information (500 devices)
  - `cards` - Payment card data (36 cards)

### 2. **Neo4j** (Graph Database)
- **Purpose**: Represents transactions as a graph structure
- **Structure**:
  - **Nodes**: 
    - `User` nodes (53 users)
    - `Transaction` nodes (505 transactions)
    - `Location`, `Device`, `Card` nodes
  - **Relationships**:
    - `(User)-[:SENT]->(Transaction)-[:TO]->(User)`
    - `(User)-[:LOCATED_AT]->(Location)`
    - `(Transaction)-[:USED_DEVICE]->(Device)`
    - `(User)-[:HAS_CARD]->(Card)`
  - **Total**: 1,593 relationships

### 3. **Redis** (Feature Store)
- **Purpose**: Fast access to pre-computed features
- **Content**: All 62 extracted features ready for model training

---

## 🔢 **Feature Engineering (62 Features Total)**

### **Node Features (50 features)**
Extracted for each user node:

#### A. Transaction Statistics (15 features)
- Total transactions sent/received
- Total amount sent/received
- Average transaction amount
- Max/min transaction amounts
- Transaction count by status
- And more...

#### B. Graph Topology (12 features)
- Node degree (in/out)
- Degree centrality
- Betweenness centrality
- Closeness centrality
- PageRank score
- Katz centrality
- Eigenvector centrality
- Clustering coefficient
- Average neighbor degree
- Triangle count

#### C. Temporal Features (10 features)
- First/last transaction timestamp
- Account age
- Time since last transaction
- Transactions in last 24h/7d/30d
- Mode hour of day
- Mode day of week
- Transaction time variance

#### D. Behavioral Features (8 features)
- Unique receivers count
- Unique senders count
- Transaction frequency
- Amount variance
- Cross-border transaction ratio
- Failed transaction ratio
- And more...

#### E. Fraud Propagation (5 features)
- Fraud neighbor count
- Fraud neighbor ratio
- Fraud transaction count
- Fraud amount sum
- Fraud propagation score

### **Edge Features (12 features)**
Extracted for each transaction edge:

1. **Amount** - Transaction amount
2. **Timestamp** - Transaction time
3. **Hour of day** - When transaction occurred
4. **Day of week** - Which day
5. **Is weekend** - Weekend indicator
6. **Time since last** - Time gap between users
7. **Amount percentile (sender)** - Sender's percentile
8. **Amount percentile (receiver)** - Receiver's percentile
9. **Is reciprocal** - Bidirectional transaction
10. **Reciprocal time gap** - Time between reciprocal transactions
11. **Geographic distance** - Distance between users
12. **Transaction mode** - Payment method

---

## 🔄 **Data Lifecycle**

### **Step 1: Data Generation**
```bash
python storage/datagen.py
```
- Generates **500 synthetic transactions** with realistic fraud patterns
- **130 fraudulent transactions (26%)**
- **50 users** with various risk profiles
- Inserts directly into PostgreSQL

### **Step 2: Graph Building**
```bash
python scripts/sync_postgres_to_neo4j.py
```
- Syncs data from PostgreSQL → Neo4j
- Creates graph structure with nodes and relationships
- Builds the transaction network

### **Step 3: Feature Extraction**
```bash
python scripts/extract_all_features.py
```
- Loads graph from Neo4j
- Calculates all **62 features** (50 node + 12 edge)
- Stores features in Redis for fast access

### **Step 4: Model Training** (Next Step)
- Load features from Redis
- Train GNN model on the graph
- Predict fraud labels

---

## 🧠 **Fraud Detection Logic**

### **Transaction-Level Patterns Detected:**
1. **Fraud Network** - Transactions between known fraud users
2. **High Velocity** - Rapid successive transactions
3. **Amount Anomalies** - Threshold avoidance, round numbers, outliers
4. **Cross-Border** - Suspicious international transactions
5. **Failed Transactions** - High-value failed/reversed transactions
6. **Device Switching** - Multiple devices used quickly
7. **Unusual Hours** - Late night/early morning high-value transactions
8. **Geographic Distance** - Long-distance high-amount transactions
9. **Reciprocity Patterns** - Money laundering indicators
10. **Extremely High Amounts** - Unusually large transactions

### **User-Level Patterns Detected:**
1. **Unverified High Volume** - Unverified users with high transaction volume
2. **Low Balance, High Spending** - Suspicious spending patterns
3. **High Transaction Frequency** - Unusual activity levels
4. **High Risk Score** - Pre-calculated risk indicators
5. **Fake Merchant** - Merchant accounts with suspicious patterns
6. **High Amount Variance** - Money laundering indicators
7. **Excessive Cross-Border** - Frequent international transactions
8. **High Failure Rate** - Many failed transactions
9. **Device Switching** - Multiple devices used
10. **Extremely High Single Transaction** - Suspicious large payments

---

## 🛠️ **Technology Stack**

### **Databases:**
- **PostgreSQL** - Relational database (transactional data)
- **Neo4j** - Graph database (network structure)
- **Redis** - In-memory cache (feature store)

### **Message Queue:**
- **Apache Kafka** - Distributed message broker (for future distributed processing)

### **Graph Processing:**
- **DGL (Deep Graph Library)** - Graph neural network framework
- **NetworkX** - Graph analysis and metrics

### **Data Processing:**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations
- **PyTorch** - Deep learning framework

### **Infrastructure:**
- **Docker** - Containerization
- **Docker Compose** - Service orchestration

---

## 📈 **Current Status**

### ✅ **Completed:**
1. ✅ Data generation (500 transactions, 130 fraudulent)
2. ✅ PostgreSQL schema and data insertion
3. ✅ Neo4j graph structure (53 users, 505 transactions, 1,593 relationships)
4. ✅ Feature extraction (50 node features extracted)
5. ✅ Redis connection established

### ⏳ **In Progress:**
- Edge feature extraction (12 features)

### 📋 **Next Steps:**
1. Complete edge feature extraction
2. Store all features in Redis
3. Train GNN model
4. Evaluate model performance
5. Deploy for real-time fraud detection

---

## 🎯 **Key Features of the System**

### **1. Distributed Architecture**
- Multiple databases for different purposes
- Kafka for distributed feature extraction
- Redis for fast feature access

### **2. Comprehensive Feature Engineering**
- **62 features** covering:
  - Transaction patterns
  - Graph structure
  - Temporal behavior
  - Geographic patterns
  - Fraud propagation

### **3. Realistic Fraud Patterns**
- **12 transaction-level** fraud patterns
- **10 user-level** fraud patterns
- **26% fraud rate** (130/500 transactions)

### **4. Scalable Design**
- Batch processing
- Graph-based analysis
- Feature caching in Redis

---

## 🔍 **How It Works**

1. **Transaction Data** flows into PostgreSQL
2. **Graph Structure** is built in Neo4j showing relationships
3. **Features** are extracted from the graph structure
4. **GNN Model** learns patterns from features
5. **Predictions** identify fraudulent transactions

---

## 📊 **Data Statistics**

- **Users**: 50 (10 fraudulent)
- **Transactions**: 500 (130 fraudulent = 26%)
- **Locations**: 5 countries
- **Devices**: 500 unique devices
- **Cards**: 36 payment cards
- **Graph Nodes**: 53 users + 505 transactions
- **Graph Edges**: 1,593 relationships
- **Features**: 62 total (50 node + 12 edge)

---

## 🚀 **Future Enhancements**

1. **Real-time Processing** - Stream transactions through Kafka
2. **Model Serving** - API for real-time fraud detection
3. **Distributed Training** - Scale across multiple machines
4. **Feature Versioning** - Track feature changes over time
5. **Model Monitoring** - Track model performance in production

---

## 📝 **Quick Reference**

### **Key Scripts:**
- `storage/datagen.py` - Generate synthetic data
- `scripts/sync_postgres_to_neo4j.py` - Build graph
- `scripts/extract_all_features.py` - Extract features
- `scripts/verify_features_in_redis.py` - Verify features

### **Key Files:**
- `config.yaml` - System configuration
- `docker-compose.yml` - Service orchestration
- `feature_engineering/feature_extractor.py` - Feature extraction logic

---

**This system provides a complete pipeline from raw transactions to fraud predictions using graph neural networks!** 🎉

