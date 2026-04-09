# Step 5 — Docker Setup (Distributed Workers)

## Goal

Create a Docker Compose environment where:
- **N worker containers** each receive a copy of the graph and run `distributed_feature_extraction.py`
- **A coordinator service** aggregates results (optional for small N)
- **A Redis container** stores extracted features (optional — see Step 7)

---

## Prerequisites

| Requirement | How to check |
|-------------|-------------|
| Docker Desktop (Windows) | `docker --version` |
| Docker Compose v2 | `docker compose version` |
| Python 3.9+ on your machine | `python --version` |
| `./data/` folder populated | Must have `features.csv` + `relation_*.csv` from `gnn/graph_utils.py` |

**Install Docker Desktop for Windows:**
https://docs.docker.com/desktop/install/windows-install/

---

## File Structure to Create

```
graph-fraud-detection-main/
├── docker/
│   ├── Dockerfile.worker          ← Python worker image
│   ├── docker-compose.yml         ← Orchestrates N workers + Redis
│   └── entrypoint.sh              ← Worker startup script
├── scripts/
│   └── compress_graph.py          ← From Step 4
└── docs/
    └── ...
```

---

## Implementation

### 5.1 — Worker Dockerfile

Create `docker/Dockerfile.worker`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY gnn/ ./gnn/
COPY distributed_feature_extraction.py .

# Entry point
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
```

### 5.2 — `requirements.txt` (create in project root)

```
torch==2.1.0
torch-geometric==2.4.0
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
networkx>=3.0
redis>=5.0
```

> **Note:** torch-geometric wheel must match your torch+CUDA version.
> For CPU-only: `pip install torch-geometric` works directly.

### 5.3 — Worker Entrypoint Script

Create `docker/entrypoint.sh`:

```bash
#!/bin/bash
set -e

WORKER_ID=${WORKER_ID:-0}
N_WORKERS=${N_WORKERS:-2}
DATA_DIR=${DATA_DIR:-/app/data}
OUTPUT_DIR=${OUTPUT_DIR:-/app/features}
CACHE_DIR=${CACHE_DIR:-/app/cache}

echo "Starting Worker $WORKER_ID of $N_WORKERS"

python distributed_feature_extraction.py \
  --transaction-data $DATA_DIR/features.csv \
  --training-dir $DATA_DIR \
  --output-dir $OUTPUT_DIR/worker_$WORKER_ID \
  --cache-dir $CACHE_DIR/worker_$WORKER_ID \
  --n-workers 1
```

### 5.4 — Docker Compose

Create `docker/docker-compose.yml`:

```yaml
version: "3.9"

services:

  # --- Redis for feature storage (see Step 7) ---
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - fraud_net

  # --- Worker 1 ---
  worker-1:
    build:
      context: ..
      dockerfile: docker/Dockerfile.worker
    environment:
      - WORKER_ID=1
      - N_WORKERS=3
      - DATA_DIR=/app/data
      - OUTPUT_DIR=/app/features
      - CACHE_DIR=/app/cache
      - REDIS_HOST=redis
    volumes:
      - ../data:/app/data:ro           # Read-only graph data
      - ../features:/app/features      # Shared output
      - ../cache:/app/cache
    depends_on:
      - redis
    networks:
      - fraud_net

  # --- Worker 2 ---
  worker-2:
    build:
      context: ..
      dockerfile: docker/Dockerfile.worker
    environment:
      - WORKER_ID=2
      - N_WORKERS=3
      - DATA_DIR=/app/data
      - OUTPUT_DIR=/app/features
      - REDIS_HOST=redis
    volumes:
      - ../data:/app/data:ro
      - ../features:/app/features
      - ../cache:/app/cache
    depends_on:
      - redis
    networks:
      - fraud_net

  # --- Worker 3 ---
  worker-3:
    build:
      context: ..
      dockerfile: docker/Dockerfile.worker
    environment:
      - WORKER_ID=3
      - N_WORKERS=3
      - DATA_DIR=/app/data
      - OUTPUT_DIR=/app/features
      - REDIS_HOST=redis
    volumes:
      - ../data:/app/data:ro
      - ../features:/app/features
      - ../cache:/app/cache
    depends_on:
      - redis
    networks:
      - fraud_net

volumes:
  redis_data:

networks:
  fraud_net:
    driver: bridge
```

### 5.5 — How to Run

```bash
# From project root
cd docker

# Build the worker image
docker compose build

# Start all services
docker compose up -d

# Watch logs
docker compose logs -f worker-1

# Stop everything
docker compose down
```

---

## ❓ Inputs Needed From You

| Question | Why it matters |
|----------|---------------|
| How many worker containers do you want? | I'll adjust `N_WORKERS` and replicate service blocks |
| GPU available? | Need different torch+CUDA base image and device flags |
| Do you want Redis, or just flat CSV files for features? | Affects whether redis service is needed |
| Windows Docker — are you using WSL2 backend? | Affects volume path syntax |

---

## Tradeoffs

| Decision | Option A | Option B |
|----------|----------|----------|
| Feature storage | Redis (fast, in-memory) | CSV files (simpler, no extra service) |
| Scaling workers | Add more service blocks | Use Docker Swarm / Kubernetes |
| GPU support | Add `deploy.resources.reservations.devices` | CPU-only (simpler) |
| Cross-machine | Docker Swarm overlay network | Docker Compose (single host only) |

---

## Next Step → [Step 6: Distributed Feature Extraction](06_distributed_feature_extraction.md)
