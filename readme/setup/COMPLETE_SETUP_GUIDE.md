# Complete Setup Guide: Fraud Detection System

## Overview

This guide will help you set up the entire fraud detection system from scratch. You currently have PostgreSQL installed. We'll set up:
- PostgreSQL (already installed)
- Neo4j (Graph Database)
- Kafka (Message Broker)
- Redis (Feature Store)
- All Python dependencies
- Feature extraction pipeline

---

## Prerequisites Checklist

- [x] PostgreSQL installed and running
- [ ] Docker Desktop installed
- [ ] Python 3.9+ installed
- [ ] Git (optional, for cloning)

---

## Step 1: Install Docker Desktop

### Windows:
1. Download: https://www.docker.com/products/docker-desktop/
2. Install Docker Desktop
3. Restart your computer
4. Open Docker Desktop and ensure it's running (green icon in system tray)

### Verify Docker:
```bash
docker --version
docker-compose --version
```

---

## Step 2: Install Python Dependencies

### Create Virtual Environment (Recommended):
```bash
# Navigate to project directory
cd C:\GNN-project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate
```

### Install Requirements:
```bash
pip install -r requirements.txt
```

---

## Step 3: Start All Services with Docker Compose

### Start Services:
```bash
# Make sure you're in the project directory
cd C:\GNN-project

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

### Expected Services:
- PostgreSQL (if not using external)
- Neo4j
- Zookeeper
- Kafka
- Redis
- Kafka UI (optional)

---

## Step 4: Initialize Databases

### 4.1 PostgreSQL Setup

**If using external PostgreSQL:**
```bash
# Connect to your PostgreSQL
psql -U postgres
```

**If using Docker PostgreSQL:**
```bash
# Connect to containerized PostgreSQL
docker exec -it fraud-detection-postgres psql -U postgres -d fraud_detection
```

**Run initialization script:**
```bash
# From project root
psql -U postgres -d fraud_detection -f scripts/init_postgres.sql

# Or using Docker
docker exec -i fraud-detection-postgres psql -U postgres -d fraud_detection < scripts/init_postgres.sql
```

### 4.2 Neo4j Setup

1. **Open Neo4j Browser:**
   - URL: http://localhost:7474
   - Username: `neo4j`
   - Password: `neo4j123` (change on first login)

2. **Run initialization script:**
   - Copy contents of `scripts/init_neo4j.cypher`
   - Paste in Neo4j Browser
   - Click "Run"

### 4.3 Kafka Topics Setup

**Option 1: Using Python script (Recommended)**
```bash
python scripts/create_kafka_topics.py
```

**Option 2: Using Batch script (Windows)**
```bash
scripts\create_kafka_topics.bat
```

**Option 3: Manual creation**
```bash
docker exec fraud-detection-kafka kafka-topics --create --topic transactions-raw --bootstrap-server localhost:9092 --partitions 6 --replication-factor 1
docker exec fraud-detection-kafka kafka-topics --create --topic feature-extraction --bootstrap-server localhost:9092 --partitions 4 --replication-factor 1
docker exec fraud-detection-kafka kafka-topics --create --topic features-computed --bootstrap-server localhost:9092 --partitions 4 --replication-factor 1
```

---

## Step 5: Verify All Services

### Test Scripts:

Run each test script to verify connections:

```bash
# Test PostgreSQL
python scripts/test_postgres.py

# Test Neo4j
python scripts/test_neo4j.py

# Test Kafka
python scripts/test_kafka.py

# Test Redis
python scripts/test_redis.py
```

All should return ✅ success messages.

---

## Step 6: Load Transaction Data

### Option A: Use Existing Data

If you have transaction data in CSV format:

```bash
# Load data into PostgreSQL
python scripts/load_transactions_to_postgres.py --file ./storage/india_fraud_data_explainable/transactions.csv
```

### Option B: Generate Sample Data

```bash
# Generate sample transaction data
python scripts/generate_sample_data.py --count 10000
```

---

## Step 7: Sync Data to Neo4j

```bash
# Sync PostgreSQL transactions to Neo4j graph
python scripts/sync_postgres_to_neo4j.py
```

This will:
1. Read transactions from PostgreSQL
2. Create nodes and relationships in Neo4j
3. Publish to Kafka for feature extraction

---

## Step 8: Extract Features and Store in Redis

### Full Feature Extraction Pipeline:

```bash
# Run complete feature extraction
python scripts/extract_all_features.py
```

This script will:
1. Query Neo4j for graph data
2. Extract all 50 node features
3. Extract all 12 edge features
4. Store features in Redis
5. Also save to Parquet as backup

### Verify Features in Redis:

```bash
# Check features in Redis
python scripts/verify_features_in_redis.py
```

---

## Step 9: Verify Feature Store

### Check Redis Contents:

```bash
# Connect to Redis
docker exec -it fraud-detection-redis redis-cli

# In Redis CLI:
KEYS features:*
GET features:metadata:node
GET features:metadata:edge
```

### Python Verification:

```bash
python scripts/verify_features_in_redis.py
```

---

## Step 10: Start API Service (Optional)

```bash
# Start API for inference
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Or using Docker
docker-compose up api
```

Test API:
```bash
curl http://localhost:8000/health
```

---

## Troubleshooting

### Issue: Docker containers won't start
```bash
# Check logs
docker-compose logs

# Restart services
docker-compose down
docker-compose up -d
```

### Issue: Port already in use
```bash
# Find process using port
netstat -ano | findstr :5432  # PostgreSQL
netstat -ano | findstr :7474  # Neo4j
netstat -ano | findstr :9092  # Kafka
netstat -ano | findstr :6379  # Redis

# Change ports in docker-compose.yml if needed
```

### Issue: Neo4j password reset
```bash
docker exec -it fraud-detection-neo4j neo4j-admin dbms set-initial-password newpassword
```

### Issue: Kafka topics not created
```bash
# Wait for Kafka to be ready
docker exec fraud-detection-kafka kafka-broker-api-versions --bootstrap-server localhost:9092

# Then create topics
scripts\create_kafka_topics.bat
```

### Issue: Python package installation fails
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install packages one by one
pip install torch
pip install dgl
pip install redis
pip install neo4j
pip install kafka-python
```

---

## Quick Reference Commands

### Docker Services:
```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f [service-name]

# Restart service
docker-compose restart [service-name]
```

### Database Access:
```bash
# PostgreSQL
docker exec -it fraud-detection-postgres psql -U postgres -d fraud_detection

# Neo4j Browser
# Open: http://localhost:7474

# Redis CLI
docker exec -it fraud-detection-redis redis-cli
```

### Feature Extraction:
```bash
# Extract all features
python scripts/extract_all_features.py

# Verify features
python scripts/verify_features_in_redis.py
```

---

## Next Steps

After completing setup:

1. ✅ All services running
2. ✅ Databases initialized
3. ✅ Transaction data loaded
4. ✅ Features extracted and stored in Redis
5. ✅ System ready for training/inference

You can now:
- Train the GNN model: `python main.py --mode train`
- Run inference: `python main.py --mode infer`
- Use API: `curl http://localhost:8000/predict`

---

## Support

If you encounter issues:
1. Check service logs: `docker-compose logs [service]`
2. Verify connections: Run test scripts
3. Check DATA_LIFECYCLE_ARCHITECTURE.md for architecture details

