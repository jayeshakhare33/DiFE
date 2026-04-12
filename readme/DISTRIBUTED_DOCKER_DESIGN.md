# Distributed Feature Extraction — Docker Architecture Design

> **Document Scope**: Full pipeline trace of the current local system, followed by a complete
> distributed architecture proposal using a single Docker network, 4 worker containers,
> centralised storage, and a master notification mechanism.

---

## Part 1 — Current Pipeline Flow (Local)

### How `python main.py --mode build` works today

```
main.py (entry point)
    │
    ├─► 1. load_config()
    │       Reads config.yaml → model, training, storage, data paths, kafka, etc.
    │
    ├─► 2. GraphBuilder (graph_processing/graph_builder.py)
    │       load_transaction_data()
    │           └─ reads ./storage/india_fraud_data_explainable/transactions.csv
    │              (+ users.csv, locations.csv from same dir)
    │
    │       build_user_transaction_graph()
    │           ├─ Maps every sender_id / receiver_id → integer node index
    │           ├─ Builds a DGL HeteroGraph:
    │           │     Node type : 'user'  (50 nodes in sample data)
    │           │     Edge type : ('user', 'transaction', 'user') (500 edges)
    │           └─ Attaches edge features on each edge tensor:
    │                 amount, timestamp, hour_of_day, day_of_week,
    │                 transaction_mode, geographic_distance, status, is_cross_border
    │           └─ Attaches node data:
    │                 is_fraud (fraud label aggregated per user)
    │
    ├─► 3. FeatureStore (storage/feature_store.py)
    │       Initialised with backend = 'parquet' (from config.yaml)
    │       Base directory → ./data/features
    │
    ├─► 4. extract_features() → calls FeatureExtractor (feature_engineering/feature_extractor.py)
    │       extract_all_features()
    │           ├─ A. Transaction Statistics  (15 features)
    │           │     total_sent, total_received, avg/max/std amounts,
    │           │     net_flow, frequency, unique_senders/receivers,
    │           │     avg_time_between_transactions
    │           │
    │           ├─ B. Graph Topology          (12 features)
    │           │     in/out/total degree, degree_centrality,
    │           │     betweenness, closeness, pagerank, clustering,
    │           │     katz, eigenvector, avg_neighbor_degree, triangles
    │           │     (uses NetworkX internally for centrality algorithms)
    │           │
    │           ├─ C. Temporal Features       (10 features)
    │           │     account_age_days, first/last_transaction_timestamp,
    │           │     time_since_last, tx_last_24h/7d/30d,
    │           │     hour_of_day_mode, day_of_week_mode, time_variance
    │           │
    │           ├─ D. Behavioral Features     (8 features)
    │           │     round_amount_ratio, threshold_amount_ratio,
    │           │     mode_diversity, failed_ratio, reversal_ratio,
    │           │     cross_border_ratio, high_risk_country_ratio, burst_score
    │           │
    │           └─ E. Fraud Propagation       (5 features)
    │                 connected_to_fraud_count, fraud_propagation_score,
    │                 distance_to_nearest_fraud, common_neighbors_with_frauds,
    │                 fraud_cluster_membership
    │
    │       EdgeFeatureExtractor (same file)
    │           extract_all_edge_features()
    │               └─ reads pre-built edge tensors already on the graph
    │                  (amount, timestamp, hour_of_day, day_of_week,
    │                   is_weekend, transaction_mode, geographic_distance,
    │                   reciprocal_time_gap, etc.)
    │
    ├─► 5. FeatureStore.save_features()
    │       node_features → ./data/features/node_features.parquet
    │       edge_features → ./data/features/edge_features.parquet
    │
    └─► (mode=train) DistributedTrainer (gnn_training/distributed_trainer.py)
            Uses torch.multiprocessing.spawn() to launch N worker processes
            Each worker:
                - Loads graph + features
                - Initialises HeteroRGCN model (gnn_training/gnn_model.py)
                - Trains with CrossEntropyLoss + Adam
                - Saves best model → ./model/best_model.pth
```

### API (inference only — separate process)
```
uvicorn api.app:app
    │
    ├─ on_startup: loads NeatureStore + InferenceService (api/inference.py)
    │               reads model from ./model/best_model.pth
    │
    ├─ GET  /health       → liveness check
    ├─ POST /predict      → takes node_ids → returns fraud probability
    └─ POST /features     → returns stored feature vectors
```

---

## Part 2 — Proposed Distributed Architecture

### Guiding Principles
1. **Logic is not changed** — the 50-feature extraction logic remains identical; only *which nodes* each worker processes changes.
2. **No GPU dependency** — all containers run CPU-only PyTorch + DGL.
3. **Single Docker network** — all containers communicate over one internal bridge network (`fraud_net`).
4. **Central storage** — raw CSV input is served from a shared volume (or lightweight object store); extracted feature Parquet files are written back to the same central location.
5. **Master notification** — each worker POSTs an HTTP callback to the master's `/worker/done` endpoint when it finishes, passing its worker ID and output path.

---

### High-Level Architecture Diagram

```
┌────────────────────────────────────────── Docker Network: fraud_net ──────────────────────────────────────────┐
│                                                                                                                │
│   ┌─────────────────────────────┐       HTTP callbacks (/worker/done)                                        │
│   │      MASTER CONTAINER       │◄──────────────────────────────────────────────────────────────────┐        │
│   │   (orchestrator service)    │                                                                    │        │
│   │                             │  dispatches work (node ranges)                                    │        │
│   │  - FastAPI orchestrator     │──────────────────────────────────────────────────────────────────►│        │
│   │  - Tracks worker status     │                                                                    │        │
│   │  - Merges outputs           │                                                                    │        │
│   │  - Port 9000                │                                                                    │        │
│   └──────────────┬──────────────┘                                                                    │        │
│                  │ reads/writes                                                                       │        │
│                  ▼                                                                                    │        │
│   ┌──────────────────────────────────────────────────────────────────────────────────────┐           │        │
│   │                  CENTRAL SHARED VOLUME  (MinIO / Docker Volume)                     │           │        │
│   │                                                                                      │           │        │
│   │  /data/raw/transactions.csv          (INPUT — read by all workers)                  │           │        │
│   │  /data/features/worker_0_node.parquet  (OUTPUT — written by worker 0)               │           │        │
│   │  /data/features/worker_1_node.parquet  (OUTPUT — written by worker 1)               │           │        │
│   │  /data/features/worker_2_node.parquet  (OUTPUT — written by worker 2)               │           │        │
│   │  /data/features/worker_3_node.parquet  (OUTPUT — written by worker 3)               │           │        │
│   │  /data/features/merged_node_features.parquet  (FINAL — written by master)           │           │        │
│   └──────────────────────────────────────────────────────────────────────────────────────┘           │        │
│         ▲  ▲  ▲  ▲  (each worker reads full CSV, processes its slice of nodes, writes parquet)       │        │
│         │  │  │  │                                                                                    │        │
│   ┌─────┘  │  │  └─────┐                                                                             │        │
│   │        │  │        │                                                                             │        │
│ ┌─┴──┐  ┌─┴──┐  ┌─┴──┐  ┌─┴──┐                                                                      │        │
│ │ W0 │  │ W1 │  │ W2 │  │ W3 │   ← Worker Containers (feature_worker image)                         │        │
│ └────┘  └────┘  └────┘  └────┘                                                                      │        │
│   │        │       │       └──────────────────────────────────────────────────────────────────────── ┘        │
│   └────────┴───────┴──── all POST to master:9000/worker/done when done                                       │
│                                                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Component Design

#### 1. Central Storage — Docker Named Volume (Local) or MinIO (Scalable)

**Option A — Docker Named Volume (Simplest, no external dependency)**
```yaml
# docker-compose.yml
volumes:
  fraud_data:
    driver: local
```
All containers mount the same volume at `/app/data`. Since they are on the same host, writes from one container are immediately visible to others.

**Option B — MinIO (Recommended for production / multi-host scaling)**

MinIO is an S3-compatible open-source object store that runs as a Docker container.
```yaml
services:
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"   # S3 API
      - "9001:9001"   # Console UI
    volumes:
      - minio_data:/data
    environment:
      MINIO_ROOT_USER: fraudadmin
      MINIO_ROOT_PASSWORD: fraudpass123
    networks:
      - fraud_net
```

Workers read/write via standard `boto3` (AWS S3 SDK — works identically with MinIO):
```python
import boto3

s3 = boto3.client(
    's3',
    endpoint_url='http://minio:9000',
    aws_access_key_id='fraudadmin',
    aws_secret_access_key='fraudpass123'
)

# Upload extracted features
s3.upload_file('/tmp/worker_0_node.parquet', 'fraud-features', 'worker_0_node.parquet')

# Download CSV data
s3.download_file('fraud-data', 'transactions.csv', '/tmp/transactions.csv')
```

> **Why MinIO and not AWS S3?**
> Running MinIO locally inside Docker costs zero, needs no credentials, has an
> identical S3 API, and can be swapped for real AWS S3 just by changing the
> `endpoint_url`. It keeps the project self-contained.

---

#### 2. Data Partitioning Strategy (Logic Preserved)

The full transaction CSV is **read by every worker** (it is a small file, 500 rows in the sample). What is partitioned is the **set of user nodes** that each worker is responsible for computing features on.

```
Total users = N
Worker 0 → users [0,         N//4)
Worker 1 → users [N//4,    2*N//4)
Worker 2 → users [2*N//4,  3*N//4)
Worker 3 → users [3*N//4,  N)
```

The full graph (`dgl.DGLHeteroGraph`) is built from the complete CSV by each worker — this is necessary because graph-topology features like `betweenness_centrality` and `fraud_propagation_score` require global graph knowledge. However, the final feature _computation_ (the expensive per-node loop) is done only on the assigned slice.

```python
# worker_entrypoint.py (pseudocode)
worker_id   = int(os.getenv("WORKER_ID"))          # 0, 1, 2, 3
total_workers = int(os.getenv("TOTAL_WORKERS", 4))
master_url  = os.getenv("MASTER_URL")              # http://master:9000

# 1. Load full CSV (from shared volume or MinIO)
df = pd.read_csv("/app/data/raw/transactions.csv")

# 2. Build complete graph (no logic change)
builder = GraphBuilder(num_workers=1, chunk_size=10000)
g, user_id_to_node, df = builder.build_user_transaction_graph(df, ...)

# 3. Determine this worker's node slice
all_node_ids = list(range(g.number_of_nodes('user')))
chunk = len(all_node_ids) // total_workers
start = worker_id * chunk
end   = (worker_id + 1) * chunk if worker_id < total_workers - 1 else len(all_node_ids)
my_nodes = all_node_ids[start:end]

# 4. Extract features for my slice (FeatureExtractor accepts a node_subset param)
extractor = FeatureExtractor(transaction_df=df)
features_df = extractor.extract_all_features(g, node_type='user', node_subset=my_nodes)

# 5. Save output locally, then upload to shared storage
out_path = f"/tmp/worker_{worker_id}_node_features.parquet"
features_df.to_parquet(out_path)
# upload to MinIO / copy to shared volume

# 6. Ping master
import requests
requests.post(f"{master_url}/worker/done", json={
    "worker_id": worker_id,
    "output_path": f"worker_{worker_id}_node_features.parquet",
    "status": "success",
    "rows": len(features_df)
})
```

> **Key Point**: The `FeatureExtractor` currently loops over all nodes. Adding a
> `node_subset` parameter (a list of node indices) to `extract_all_features()` in
> `feature_engineering/feature_extractor.py` is the **only code change needed** in the
> existing logic. All 50 feature formulas remain identical.

---

#### 3. Master Container — Orchestrator Service

The master is a FastAPI app (a new file, e.g., `orchestrator/master.py`) that:
- Knows the total number of workers
- Tracks which workers have reported completion
- Once all workers are done, downloads and merges all partial Parquet files into a single final `merged_node_features.parquet`

```python
# orchestrator/master.py (pseudocode)
from fastapi import FastAPI
import asyncio, pandas as pd

app = FastAPI()
worker_results = {}
TOTAL_WORKERS = int(os.getenv("TOTAL_WORKERS", 4))

@app.post("/worker/done")
async def worker_done(payload: dict):
    worker_id   = payload["worker_id"]
    output_path = payload["output_path"]
    status      = payload["status"]

    worker_results[worker_id] = {"path": output_path, "status": status}
    print(f"[Master] Worker {worker_id} done. ({len(worker_results)}/{TOTAL_WORKERS})")

    if len(worker_results) == TOTAL_WORKERS:
        # All workers finished — merge outputs
        await merge_features()

    return {"ack": True}

async def merge_features():
    dfs = []
    for wid, info in worker_results.items():
        # Download from MinIO or read from shared volume
        df = pd.read_parquet(f"/app/data/features/{info['path']}")
        dfs.append(df)

    merged = pd.concat(dfs).sort_index()
    merged.to_parquet("/app/data/features/merged_node_features.parquet")
    print("[Master] All features merged successfully.")
```

---

#### 4. Full `docker-compose.yml` Skeleton

```yaml
version: "3.9"

networks:
  fraud_net:
    driver: bridge

volumes:
  fraud_data:

services:

  # ── MinIO: Central Object Storage ──────────────────────────────────────────
  minio:
    image: minio/minio:latest
    container_name: minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: fraudadmin
      MINIO_ROOT_PASSWORD: fraudpass123
    volumes:
      - fraud_data:/data
    ports:
      - "9002:9000"   # exposed externally on 9002 to avoid conflicts
      - "9003:9001"   # MinIO console
    networks:
      - fraud_net
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 10s
      retries: 5

  # ── Master Orchestrator ─────────────────────────────────────────────────────
  master:
    build:
      context: .
      dockerfile: Dockerfile.orchestrator
    container_name: master
    environment:
      TOTAL_WORKERS: "4"
      MINIO_ENDPOINT: "http://minio:9000"
      MINIO_ACCESS_KEY: fraudadmin
      MINIO_SECRET_KEY: fraudpass123
    ports:
      - "9000:9000"
    networks:
      - fraud_net
    depends_on:
      minio:
        condition: service_healthy

  # ── Feature Worker 0 ────────────────────────────────────────────────────────
  worker_0:
    build:
      context: .
      dockerfile: Dockerfile.worker
    container_name: worker_0
    environment:
      WORKER_ID: "0"
      TOTAL_WORKERS: "4"
      MASTER_URL: "http://master:9000"
      MINIO_ENDPOINT: "http://minio:9000"
      MINIO_ACCESS_KEY: fraudadmin
      MINIO_SECRET_KEY: fraudpass123
    networks:
      - fraud_net
    depends_on:
      - master

  # ── Feature Worker 1 ────────────────────────────────────────────────────────
  worker_1:
    build:
      context: .
      dockerfile: Dockerfile.worker
    container_name: worker_1
    environment:
      WORKER_ID: "1"
      TOTAL_WORKERS: "4"
      MASTER_URL: "http://master:9000"
      MINIO_ENDPOINT: "http://minio:9000"
      MINIO_ACCESS_KEY: fraudadmin
      MINIO_SECRET_KEY: fraudpass123
    networks:
      - fraud_net
    depends_on:
      - master

  # ── Feature Worker 2 ────────────────────────────────────────────────────────
  worker_2:
    build:
      context: .
      dockerfile: Dockerfile.worker
    container_name: worker_2
    environment:
      WORKER_ID: "2"
      TOTAL_WORKERS: "4"
      MASTER_URL: "http://master:9000"
      MINIO_ENDPOINT: "http://minio:9000"
      MINIO_ACCESS_KEY: fraudadmin
      MINIO_SECRET_KEY: fraudpass123
    networks:
      - fraud_net
    depends_on:
      - master

  # ── Feature Worker 3 ────────────────────────────────────────────────────────
  worker_3:
    build:
      context: .
      dockerfile: Dockerfile.worker
    container_name: worker_3
    environment:
      WORKER_ID: "3"
      TOTAL_WORKERS: "4"
      MASTER_URL: "http://master:9000"
      MINIO_ENDPOINT: "http://minio:9000"
      MINIO_ACCESS_KEY: fraudadmin
      MINIO_SECRET_KEY: fraudpass123
    networks:
      - fraud_net
    depends_on:
      - master
```

---

#### 5. Worker Notification Mechanism

The chosen approach is **HTTP callback (webhook pattern)**. Each worker POSTs to the master on completion.

| Approach | Pros | Cons |
|---|---|---|
| **HTTP Callback (chosen)** | Zero extra infra, native with FastAPI, simple to debug | Worker must know master URL |
| Kafka message queue | Decoupled, fault tolerant, replay | Requires Kafka + Zookeeper containers |
| Redis pub/sub | Fast, lightweight | Extra Redis container, polling needed on consumer |
| Shared file flag | No network needed | Race conditions, polling delay |

**HTTP Callback Flow:**
```
Worker 0 finishes
    └─► POST http://master:9000/worker/done
        Body: { "worker_id": 0, "output_path": "worker_0_node_features.parquet",
                "status": "success", "rows": 13 }
            ↓
        Master receives → stores in worker_results dict
            ↓
        If len(worker_results) == 4:
            → merge all parquet files
            → write merged_node_features.parquet to MinIO
            → (optionally) notify downstream GNN trainer container
```

---

### Code Change Summary (Minimal)

Only **one change** is needed to the existing codebase:

```python
# feature_engineering/feature_extractor.py
# In extract_all_features(), add an optional node_subset parameter

def extract_all_features(self, g, node_type='user',
                         transaction_df=None,
                         node_subset=None):      # ← NEW parameter
    node_ids = g.nodes(node_type).numpy()

    # If a subset is specified, only process those nodes
    if node_subset is not None:
        node_ids = np.array(node_subset)

    # All feature extraction functions below remain unchanged
    ...
```

Everything else (the 50 feature formulas, the graph builder, the GNN trainer, the API) stays exactly the same.

---

### Storage Decision Matrix

| Option | Setup Cost | Persistence | Multi-host | Recommended for |
|---|---|---|---|---|
| Docker Named Volume | Zero | Yes (local disk) | No | Development / single machine |
| **MinIO** (S3 API) | Low | Yes (local disk) | Yes (with Docker Swarm) | **This project** |
| AWS S3 | Medium (AWS account) | Yes (cloud) | Yes | Production / cloud deployment |
| NFS shared volume | Medium | Yes | Yes (same LAN) | On-premise clusters |

---

### Deployment Steps (Once Implementation is Complete)

```bash
# 1. Start the full stack
docker-compose up --build -d

# 2. Upload the raw CSV to MinIO (one-time)
#    Use the MinIO console at http://localhost:9003
#    or use mc (MinIO client):
mc alias set local http://localhost:9002 fraudadmin fraudpass123
mc mb local/fraud-data
mc cp ./storage/india_fraud_data_explainable/transactions.csv local/fraud-data/

# 3. Workers start automatically, read CSV, extract features, ping master
# 4. Master merges all outputs into merged_node_features.parquet

# 5. Verify merged output
mc ls local/fraud-features/
```

---

### Summary

| Concern | Solution |
|---|---|
| Raw CSV access by 4 workers | MinIO bucket (S3-compatible), all workers download via `boto3` |
| Data partitioning | Each worker handles `N/4` user nodes; full graph still built by each (topology requires it) |
| Feature logic preservation | Only add `node_subset` param to `extract_all_features()`; all 50 formulas unchanged |
| Feature output storage | Workers write Parquet files to MinIO bucket; master reads and concatenates |
| Worker → Master notification | HTTP POST to `master:9000/worker/done` with worker ID and output path |
| Container networking | Single Docker bridge network `fraud_net`; all containers resolve each other by service name |
| No GPU dependency | CPU-only PyTorch (`torch==2.2.1+cpu`) and DGL (`dgl==2.2.1`) baked into worker image |
