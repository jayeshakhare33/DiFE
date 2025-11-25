# Multi-Container Docker Deployment Guide

## Overview

This guide explains how to deploy the GNN fraud detection system across multiple Docker containers with shared feature storage and inter-container communication.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Docker Network (fraud-detection-network)   │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│  │  Redis   │    │ Feature  │    │   API    │           │
│  │ (Cache)  │◄───│  Loader  │    │ (x2)     │           │
│  └────┬─────┘    └──────────┘    └────┬─────┘           │
│       │                               │                 │
│       │                               │                 │
│  ┌────▼─────┐    ┌──────────┐    ┌────▼─────┐           │
│  │ Trainer  │    │ Trainer  │    │ Trainer  │           │
│  │   -0     │    │   -1     │    │   -2     │           │
│  └──────────┘    └──────────┘    └──────────┘           │
│                                                         │
│  ┌──────────────────────────────────────────────┐       │
│  │  Shared Volume: ./data/features (Parquet)    │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

## Components

### 1. **Redis Service**
- In-memory cache for fast feature access
- Shared by all containers
- Persists data to disk

### 2. **Feature Loader**
- One-time service that loads Parquet features into Redis
- Runs at startup
- Ensures all containers have fast access to features

### 3. **API Services** (2 instances)
- Inference endpoints
- Load features from Redis (fast)
- Fallback to Parquet if Redis unavailable

### 4. **Training Workers** (4 instances)
- Distributed training across containers
- Each worker loads features from shared Parquet files
- Communicate via PyTorch Distributed

## Prerequisites
1. **Docker & Docker Compose** installed
2. **Features generated** (Parquet files in `./data/features/`)
3. **Sufficient resources**:
   - At least 8GB RAM
   - 4+ CPU cores recommended

## Quick Start

### 1. Generate Features (if not done)

```bash
# Generate features and save to Parquet
python main.py --mode build
```

### 2. Deploy Services

**Linux/Mac:**
```bash
chmod +x deploy.sh
./deploy.sh
```

**Windows (PowerShell):**
```powershell
.\deploy.ps1
```

**Manual deployment:**
```bash
# Build images
docker-compose build

# Start Redis
docker-compose up -d redis

# Load features into Redis
docker-compose up feature-loader

# Start API services
docker-compose up -d api
```

### 3. Start Distributed Training

```bash
# Start all training workers
docker-compose --profile training up trainer-0 trainer-1 trainer-2 trainer-3
```

## Container Communication

### Service Names

Containers communicate using Docker service names:

- **Redis**: `redis` (accessible at `redis:6379`)
- **API**: `api` (accessible at `api:8000`)
- **Trainers**: `trainer-0`, `trainer-1`, `trainer-2`, `trainer-3`

### Network Configuration

All containers are on the `fraud-detection-network` bridge network:
- Subnet: `172.20.0.0/16`
- Internal DNS resolution enabled
- Containers can reach each other by service name

### Example: API connecting to Redis

```python
# In container, use service name
redis_host = "redis"  # Not "localhost"!
redis_port = 6379
```

## Feature Storage Strategy

### Two-Tier Storage

1. **Parquet (Persistent)**
   - Location: `./data/features/` (shared volume)
   - Used by: Training workers
   - Benefits: Fast bulk loading, compressed

2. **Redis (Cache)**
   - Location: Redis container memory
   - Used by: API services
   - Benefits: Sub-millisecond access

### Feature Loading Flow

```
1. Features generated → Saved to Parquet
2. Feature loader → Reads Parquet → Writes to Redis
3. API containers → Read from Redis (fast)
4. Training containers → Read from Parquet (bulk loading)
```

## Configuration

### Environment Variables

Containers use environment variables for configuration:

```yaml
# API Container
FEATURE_STORE_BACKEND=redis
REDIS_HOST=redis
REDIS_PORT=6379
STORAGE_BASE_DIR=/app/data/features

# Training Container
FEATURE_STORE_BACKEND=parquet
RANK=0
WORLD_SIZE=4
MASTER_ADDR=trainer-0
MASTER_PORT=12355
```

### Config Files

- **Local**: `config.yaml` (for local development)
- **Docker**: `config.docker.yaml` (for container deployment)

## Useful Commands

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f trainer-0

# Last 100 lines
docker-compose logs --tail=100 api
```

### Check Container Status

```bash
# List all containers
docker-compose ps

# Check specific container
docker-compose ps api
```

### Execute Commands in Container

```bash
# Access API container shell
docker-compose exec api bash

# Run Python script in container
docker-compose exec api python test_parquet_features.py

# Check Redis
docker-compose exec redis redis-cli ping
docker-compose exec redis redis-cli keys "features:*"
```

### Restart Services

```bash
# Restart specific service
docker-compose restart api

# Restart all services
docker-compose restart

# Rebuild and restart
docker-compose up -d --build
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Stop specific service
docker-compose stop api
```

## Troubleshooting

### Redis Connection Issues

**Problem**: Containers can't connect to Redis

**Solution**:
```bash
# Check Redis is running
docker-compose ps redis

# Check Redis logs
docker-compose logs redis

# Test connection from container
docker-compose exec api python -c "import redis; r=redis.Redis(host='redis', port=6379); print(r.ping())"
```

### Features Not Loading

**Problem**: Features not found in Redis

**Solution**:
```bash
# Re-run feature loader
docker-compose up feature-loader

# Check Parquet files exist
ls -la ./data/features/*.parquet

# Manually load features
docker-compose exec api python scripts/load_features_to_redis.py
```

### Training Workers Can't Communicate

**Problem**: Distributed training fails with connection errors

**Solution**:
```bash
# Check all trainers are on same network
docker network inspect fraud-detection-network

# Verify trainer-0 is master
docker-compose logs trainer-0 | grep "master"

# Check port 12355 is accessible
docker-compose exec trainer-1 nc -zv trainer-0 12355
```

### Out of Memory

**Problem**: Containers running out of memory

**Solution**:
1. Reduce number of replicas in `docker-compose.yml`
2. Reduce resource limits
3. Use smaller batch sizes

```yaml
# In docker-compose.yml
deploy:
  resources:
    limits:
      memory: 2G  # Reduce from 4G
```

## Scaling

### Scale API Services

```bash
# Scale to 4 API instances
docker-compose up -d --scale api=4
```

### Add More Training Workers

1. Add new service to `docker-compose.yml`:
```yaml
trainer-4:
  # ... (copy from trainer-3, update RANK=4, node_rank=4)
```

2. Update `WORLD_SIZE` in all trainer environments

3. Start new worker:
```bash
docker-compose --profile training up trainer-4
```

## Performance Tips

1. **Use Redis for API**: API containers should use Redis for fast access
2. **Use Parquet for Training**: Training containers should use Parquet for bulk loading
3. **Preload Features**: Feature loader ensures Redis is populated at startup
4. **Shared Volumes**: All containers share the same feature directory
5. **Network Optimization**: Use bridge network for low latency

## Monitoring

### Check Resource Usage

```bash
# Container stats
docker stats

# Specific container
docker stats fraud-detection-api
```

### Check Network Traffic

```bash
# Network inspection
docker network inspect fraud-detection-network
```

## Security Considerations

1. **Network Isolation**: All containers on private network
2. **No External Exposure**: Only API port (8000) exposed
3. **Read-Only Volumes**: Feature volumes mounted read-only where possible
4. **Resource Limits**: CPU and memory limits set per container

## Next Steps

1. **Load Balancing**: Add nginx/HAProxy for API load balancing
2. **Monitoring**: Add Prometheus/Grafana for metrics
3. **Logging**: Centralize logs with ELK stack
4. **CI/CD**: Automate deployment with GitHub Actions

## Support

For issues or questions:
1. Check logs: `docker-compose logs -f`
2. Verify network: `docker network inspect fraud-detection-network`
3. Test connectivity: Use `docker-compose exec` to test connections

