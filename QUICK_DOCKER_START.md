# Quick Start: Multi-Container Deployment

## 🚀 Quick Start (3 Steps)

### 1. Ensure Features are Generated
```bash
python main.py --mode build
```

### 2. Deploy Services

**Windows (PowerShell):**
```powershell
.\deploy.ps1
```

**Linux/Mac:**
```bash
chmod +x deploy.sh
./deploy.sh
```

**Or manually:**
```bash
# Build and start
docker-compose build
docker-compose up -d redis
docker-compose up feature-loader
docker-compose up -d api
```

### 3. Verify Deployment
```bash
# Check services
docker-compose ps

# Test API
curl http://localhost:8000/health

# View logs
docker-compose logs -f api
```

## 📦 What Gets Deployed

- **Redis**: Feature cache (localhost:6379)
- **Feature Loader**: Loads Parquet → Redis (runs once)
- **API**: Inference service (localhost:8000)
- **Trainers**: 4 distributed training workers (optional)

## 🔄 Container Communication

All containers communicate via service names:
- `redis` - Redis service
- `api` - API service  
- `trainer-0`, `trainer-1`, etc. - Training workers

**Example**: API connects to Redis using `redis:6379` (not `localhost:6379`)

## 📁 Shared Storage

All containers share:
- `./data/features/` - Parquet feature files (read-only)
- `./model/` - Model files
- `./logs/` - Log files

## 🎯 Start Distributed Training

```bash
docker-compose --profile training up
```

This starts 4 training workers that:
- Load features from shared Parquet files
- Communicate via PyTorch Distributed
- Share the same network

## 🛠️ Useful Commands

```bash
# View all logs
docker-compose logs -f

# Stop everything
docker-compose down

# Restart API
docker-compose restart api

# Access container shell
docker-compose exec api bash

# Check Redis
docker-compose exec redis redis-cli ping
```

## 📚 Full Documentation

See `DOCKER_DEPLOYMENT.md` for complete details.

