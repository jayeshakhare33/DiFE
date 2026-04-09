# Step 7 — Feature Storage (Redis / CSV / API)

## Goal

After workers extract features (Step 6), those features need to be **stored in a way that**:
1. The training pipeline (`train.py`) can load them efficiently
2. The inference container (Step 8) can query them at prediction time

The architecture diagram shows three choices: **Redis**, **CSV**, or **API**.

---

## Option A — CSV Files (Simplest) ✅ Start Here

Features are already saved as `.npy` files by `DistributedFeaturePipeline`. The only addition is a small wrapper that converts them to CSV for interoperability.

**Create `scripts/save_features_csv.py`:**

```python
import numpy as np, pandas as pd, json, os

def npy_to_csv(features_dir: str = "./features", output_csv: str = "./features/features.csv"):
    feat = np.load(os.path.join(features_dir, "extracted_features.npy"))
    with open(os.path.join(features_dir, "feature_metadata.json")) as f:
        meta = json.load(f)
    
    n_cols = feat.shape[1]
    col_names = [f"feat_{i}" for i in range(n_cols)]
    df = pd.DataFrame(feat, columns=col_names)
    df.to_csv(output_csv, index=True, index_label="TransactionID")
    print(f"Saved {df.shape} → {output_csv}")

if __name__ == "__main__":
    npy_to_csv()
```

**Tradeoffs:**

| Pro | Con |
|-----|-----|
| Zero extra infrastructure | Slow random access (no indexing) |
| Easy to inspect / share | Not suitable for real-time inference |
| Already works with `train.py` `--feature-dir` | Full reload each inference call |

---

## Option B — Redis (Recommended for Inference)

Redis stores features as hash maps keyed by `TransactionID`, enabling sub-millisecond random access during inference.

### 7.1 — Start Redis

Redis runs as part of Docker Compose (already added in Step 5). To run standalone:

```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

### 7.2 — Push Features to Redis

**Create `scripts/push_features_to_redis.py`:**

```python
import numpy as np
import redis
import json
import os
import pandas as pd

def push_to_redis(
    features_npy: str = "./features/extracted_features.npy",
    node_ids_csv: str = "./data/features.csv",   # has TransactionID column
    redis_host: str = "localhost",
    redis_port: int = 6379,
    prefix: str = "feat:"
):
    r = redis.Redis(host=redis_host, port=redis_port, db=0)
    features = np.load(features_npy)
    
    # Load transaction IDs to use as keys
    ids_df = pd.read_csv(node_ids_csv, usecols=["TransactionID"])
    transaction_ids = ids_df["TransactionID"].values

    assert len(transaction_ids) == len(features), \
        f"ID/feature count mismatch: {len(transaction_ids)} vs {len(features)}"
    
    pipe = r.pipeline(transaction=False)
    for tid, feat_row in zip(transaction_ids, features):
        key = f"{prefix}{tid}"
        # Store as JSON list (float32 → list for JSON compatibility)
        pipe.set(key, json.dumps(feat_row.tolist()))
    
    pipe.execute()
    print(f"Pushed {len(transaction_ids)} feature vectors to Redis ({redis_host}:{redis_port})")

if __name__ == "__main__":
    push_to_redis()
```

**Run it:**
```bash
python scripts/push_features_to_redis.py
```

### 7.3 — Read Features From Redis (Inference side)

```python
import redis, json, numpy as np

def get_features(transaction_id: int, r: redis.Redis, prefix: str = "feat:") -> np.ndarray:
    key = f"{prefix}{transaction_id}"
    raw = r.get(key)
    if raw is None:
        return None
    return np.array(json.loads(raw), dtype=np.float32)
```

---

## Option C — REST API (Most Flexible)

Expose a `/features/{transaction_id}` endpoint backed by Redis or CSV.

**Create `api/feature_server.py`** using FastAPI:

```python
from fastapi import FastAPI, HTTPException
import redis, json, numpy as np

app = FastAPI(title="Feature Store API")
r = redis.Redis(host="redis", port=6379, db=0)

@app.get("/features/{transaction_id}")
def get_features(transaction_id: int):
    raw = r.get(f"feat:{transaction_id}")
    if raw is None:
        raise HTTPException(status_code=404, detail="Features not found")
    return {"transaction_id": transaction_id, "features": json.loads(raw)}

@app.get("/health")
def health():
    return {"status": "ok"}
```

Add to `docker-compose.yml`:
```yaml
feature-api:
  build:
    context: ..
    dockerfile: docker/Dockerfile.api
  ports:
    - "8001:8001"
  depends_on:
    - redis
  networks:
    - fraud_net
  command: uvicorn api.feature_server:app --host 0.0.0.0 --port 8001
```

---

## Storage Comparison

| Metric | CSV | Redis | REST API |
|--------|-----|-------|---------|
| Complexity | Low | Medium | High |
| Query speed | Slow (~100ms) | Fast (<1ms) | Medium (~5ms) |
| Persistence | Yes (file) | Optional (AOF/RDB) | Backed by Redis |
| Suitable for inference | No | Yes | Yes |
| Requires extra service | No | Yes (Docker) | Yes (Docker × 2) |

---

## ❓ Inputs Needed From You

| Question | Why it matters |
|----------|---------------|
| Do you need real-time inference, or batch-only? | Real-time needs Redis; batch can use CSV |
| Are you comfortable running Redis in Docker? | Determines if Option B/C is feasible |
| What's the expected query rate (e.g., 100 req/sec)? | Redis handles thousands; CSV cannot |

---

## Next Step → [Step 8: Inference Deployment](08_inference_deployment.md)
