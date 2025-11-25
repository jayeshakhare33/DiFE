# How to Run All Services

## Quick Start (3 Commands)

```bash
# 1. Start all services
docker-compose up -d

# 2. Wait for services to start (15-30 seconds)
# Then check status
docker-compose ps

# 3. Verify services are running
docker-compose logs
```

---

## Step-by-Step Instructions

### Step 1: Open Terminal/PowerShell

Navigate to your project directory:
```bash
cd C:\GNN-project
```

### Step 2: Start All Services

```bash
docker-compose up -d
```

The `-d` flag runs services in the background (detached mode).

**What this does:**
- Pulls Docker images (first time only)
- Creates containers for:
  - PostgreSQL
  - Neo4j
  - Zookeeper
  - Kafka
  - Redis
  - Kafka UI

### Step 3: Check Service Status

```bash
# See all running services
docker-compose ps

# Expected output:
# NAME                          STATUS
# fraud-detection-postgres      Up
# fraud-detection-neo4j         Up
# fraud-detection-zookeeper      Up
# fraud-detection-kafka          Up
# fraud-detection-redis          Up
# fraud-detection-kafka-ui       Up
```

### Step 4: Wait for Services to Be Ready

Services need 15-30 seconds to fully start. Check logs:

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs postgres
docker-compose logs neo4j
docker-compose logs kafka
docker-compose logs redis
```

Look for messages like:
- PostgreSQL: "database system is ready to accept connections"
- Neo4j: "Started"
- Kafka: "started (kafka.server.KafkaServer)"

### Step 5: Verify Services Are Running

**Quick Health Check:**
```bash
# Check if containers are running
docker ps

# Should show all 6 services running
```

**Test Each Service:**

1. **PostgreSQL:**
   ```bash
   docker exec fraud-detection-postgres psql -U postgres -d fraud_detection -c "SELECT version();"
   ```

2. **Neo4j:**
   - Open browser: http://localhost:7474
   - Login: `neo4j` / `neo4j123`

3. **Kafka:**
   ```bash
   docker exec fraud-detection-kafka kafka-broker-api-versions --bootstrap-server localhost:9092
   ```

4. **Redis:**
   ```bash
   docker exec fraud-detection-redis redis-cli ping
   # Should return: PONG
   ```

5. **Kafka UI:**
   - Open browser: http://localhost:8080

---

## Using Python Test Scripts

After services are running, test connections:

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

All should show ✅ success messages.

---

## Common Commands

### Start Services
```bash
docker-compose up -d
```

### Stop Services
```bash
docker-compose down
```

### Restart Services
```bash
docker-compose restart
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f postgres
docker-compose logs -f neo4j
docker-compose logs -f kafka
```

### Stop Specific Service
```bash
docker-compose stop postgres
```

### Start Specific Service
```bash
docker-compose start postgres
```

### Remove Everything (Clean Start)
```bash
# Stop and remove containers, networks
docker-compose down

# Also remove volumes (deletes data!)
docker-compose down -v
```

---

## Service URLs and Ports

| Service | URL/Connection | Default Credentials |
|---------|---------------|---------------------|
| PostgreSQL | `localhost:5432` | user: `postgres`, password: `postgres123` |
| Neo4j Browser | http://localhost:7474 | user: `neo4j`, password: `neo4j123` |
| Neo4j Bolt | `bolt://localhost:7687` | user: `neo4j`, password: `neo4j123` |
| Kafka | `localhost:9092` | No auth |
| Kafka UI | http://localhost:8080 | No auth |
| Redis | `localhost:6379` | No password |

---

## Troubleshooting

### Services Won't Start

**Check Docker Desktop:**
- Ensure Docker Desktop is running
- Check system tray for Docker icon

**Check Ports:**
```bash
# Check if ports are in use
netstat -ano | findstr :5432  # PostgreSQL
netstat -ano | findstr :7474  # Neo4j
netstat -ano | findstr :9092  # Kafka
netstat -ano | findstr :6379  # Redis
```

**View Error Logs:**
```bash
docker-compose logs [service-name]
```

### Service Keeps Restarting

Check logs for errors:
```bash
docker-compose logs -f [service-name]
```

Common issues:
- **Out of memory**: Increase Docker Desktop memory (Settings → Resources)
- **Port conflict**: Change port in docker-compose.yml
- **Volume permission**: Check file permissions

### Can't Connect to Service

1. **Wait longer**: Services take 15-30 seconds to start
2. **Check health:**
   ```bash
   docker-compose ps
   # Look for "healthy" status
   ```
3. **Check logs:**
   ```bash
   docker-compose logs [service-name]
   ```

### Reset Everything

```bash
# Stop all services
docker-compose down

# Remove volumes (deletes all data!)
docker-compose down -v

# Start fresh
docker-compose up -d
```

---

## Next Steps After Services Are Running

1. **Initialize Databases:**
   ```bash
   # PostgreSQL (if not auto-initialized)
   docker exec -i fraud-detection-postgres psql -U postgres -d fraud_detection < scripts/init_postgres.sql
   
   # Neo4j (manual - open browser)
   # Go to http://localhost:7474
   # Run scripts/init_neo4j.cypher
   ```

2. **Create Kafka Topics:**
   ```bash
   scripts\create_kafka_topics.bat
   ```

3. **Extract Features:**
   ```bash
   python scripts/extract_all_features.py
   ```

---

## Quick Reference Card

```bash
# START
docker-compose up -d

# STATUS
docker-compose ps

# LOGS
docker-compose logs -f

# STOP
docker-compose down

# RESTART
docker-compose restart

# CLEAN (removes data)
docker-compose down -v
```

---

## Verification Checklist

After running `docker-compose up -d`, verify:

- [ ] All 6 services show "Up" in `docker-compose ps`
- [ ] PostgreSQL accessible: `docker exec fraud-detection-postgres psql -U postgres -c "SELECT 1;"`
- [ ] Neo4j Browser opens: http://localhost:7474
- [ ] Kafka responds: `docker exec fraud-detection-kafka kafka-broker-api-versions --bootstrap-server localhost:9092`
- [ ] Redis responds: `docker exec fraud-detection-redis redis-cli ping` (returns PONG)
- [ ] Kafka UI opens: http://localhost:8080

Once all checkboxes are ✅, you're ready to proceed with feature extraction!

