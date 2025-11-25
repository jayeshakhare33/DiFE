# Setup Checklist

Follow this checklist to get your fraud detection system running:

## Prerequisites
- [ ] Docker Desktop installed and running
- [ ] Python 3.9+ installed
- [ ] PostgreSQL installed (or use Docker version)
- [ ] Virtual environment created (recommended)

## Step 1: Install Python Dependencies
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install requirements
pip install -r requirements.txt
```

## Step 2: Start Docker Services
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

Expected services:
- [ ] postgres (PostgreSQL)
- [ ] neo4j (Neo4j)
- [ ] zookeeper (Zookeeper)
- [ ] kafka (Kafka)
- [ ] redis (Redis)
- [ ] kafka-ui (Kafka UI - optional)

## Step 3: Initialize Databases

### PostgreSQL
```bash
# If using external PostgreSQL, create database first:
createdb fraud_detection

# Run initialization script
psql -U postgres -d fraud_detection -f scripts/init_postgres.sql

# Or using Docker:
docker exec -i fraud-detection-postgres psql -U postgres -d fraud_detection < scripts/init_postgres.sql
```

### Neo4j
- [ ] Open Neo4j Browser: http://localhost:7474
- [ ] Login: username `neo4j`, password `neo4j123`
- [ ] Copy contents of `scripts/init_neo4j.cypher`
- [ ] Paste in Neo4j Browser and click "Run"

## Step 4: Create Kafka Topics

**Option 1: Python script (Recommended)**
```bash
python scripts/create_kafka_topics.py
```

**Option 2: Batch script (Windows)**
```bash
scripts\create_kafka_topics.bat
```

## Step 5: Verify All Services
```bash
# Test PostgreSQL
python scripts/test_postgres.py
# Expected: ✅ All tests passed

# Test Neo4j
python scripts/test_neo4j.py
# Expected: ✅ All tests passed

# Test Kafka
python scripts/test_kafka.py
# Expected: ✅ All tests passed

# Test Redis
python scripts/test_redis.py
# Expected: ✅ All tests passed
```

## Step 6: Extract Features
```bash
# Extract all features and store in Redis
python scripts/extract_all_features.py
```

This will:
- [ ] Load graph from Neo4j
- [ ] Extract 50 node features
- [ ] Extract 12 edge features
- [ ] Store in Redis
- [ ] Store in Parquet (backup)

## Step 7: Verify Features in Redis
```bash
python scripts/verify_features_in_redis.py
```

Expected output:
- [ ] Node features DataFrame found
- [ ] Edge features DataFrame found
- [ ] Metadata found
- [ ] Individual node features stored

## Step 8: (Optional) Load Transaction Data

If you have transaction data to load:

```bash
# Load from CSV
python scripts/load_transactions_to_postgres.py --file your_transactions.csv

# Sync to Neo4j
python scripts/sync_postgres_to_neo4j.py

# Re-extract features
python scripts/extract_all_features.py
```

## Troubleshooting

### Services won't start
```bash
# Check logs
docker-compose logs [service-name]

# Restart services
docker-compose restart [service-name]
```

### Port conflicts
- Check if ports are already in use: `netstat -ano | findstr :5432`
- Change ports in `docker-compose.yml` if needed

### Connection errors
- Wait a few seconds for services to fully start
- Check service health: `docker-compose ps`
- Verify network: `docker network ls`

### Feature extraction fails
- Verify Neo4j has data: Check in Neo4j Browser
- Check Neo4j connection: `python scripts/test_neo4j.py`
- Verify graph structure matches expected format

## Quick Commands Reference

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f [service]

# Test connections
python scripts/test_postgres.py
python scripts/test_neo4j.py
python scripts/test_kafka.py
python scripts/test_redis.py

# Extract features
python scripts/extract_all_features.py

# Verify features
python scripts/verify_features_in_redis.py
```

## Service URLs

- PostgreSQL: `localhost:5432`
- Neo4j Browser: http://localhost:7474
- Neo4j Bolt: `bolt://localhost:7687`
- Kafka: `localhost:9092`
- Kafka UI: http://localhost:8080
- Redis: `localhost:6379`
- API (if running): http://localhost:8000

## Success Criteria

You're ready when:
- ✅ All Docker services are running
- ✅ All test scripts pass
- ✅ Features are extracted and stored in Redis
- ✅ Verification script shows features in Redis

