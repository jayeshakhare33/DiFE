# Step 8 — ML Container & Inference Deployment

## Goal

Deploy the trained GNN model as a **containerized inference service** that:
1. Loads the saved model from `./model/`
2. Accepts a transaction (or batch) as input
3. Returns a **fraud probability score** and a **binary fraud label**

This corresponds to the right column of the architecture diagram:
```
Feature Storage (Redis) → ML Container → Inference / Fraud Detection
```

---

## What Exists in the Trained Model

After `train.py` completes, `./model/` contains:

| File | Content |
|------|---------|
| `model.pth` | PyTorch model weights |
| `metadata.pkl` | Graph schema (etypes, ntype_cnt, feat_mean, feat_std) |
| `<ntype>.csv` | Learned embeddings for non-target node types (card, email, device, etc.) |

---

## Implementation

### 8.1 — Inference Module

Create `api/inference.py`:

```python
import os
import pickle
import torch
import numpy as np
from gnn.pytorch_model import HeteroRGCN
from gnn.graph_utils import construct_graph, get_edgelists

class FraudDetector:
    def __init__(self, model_dir: str = "./model", data_dir: str = "./data"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.data_dir = data_dir
        self._load_model()
    
    def _load_model(self):
        # Load metadata
        with open(os.path.join(self.model_dir, "metadata.pkl"), "rb") as f:
            meta = pickle.load(f)
        
        self.feat_mean = meta["feat_mean"]
        self.feat_std  = meta["feat_std"]
        etypes         = meta["etypes"]
        ntype_cnt      = meta["ntype_cnt"]

        in_feats  = self.feat_mean.shape[0]
        n_classes = 2
        n_hidden  = 64    # Must match training hyperparams — make configurable
        n_layers  = 2

        self.model = HeteroRGCN(
            ntype_cnt, etypes, in_feats, n_hidden, n_classes, n_layers, in_feats
        )
        self.model.load_state_dict(
            torch.load(os.path.join(self.model_dir, "model.pth"), map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully")

    def predict(self, features: np.ndarray, graph) -> dict:
        """
        Predict fraud probability for a batch of transactions.

        Args:
            features: ndarray (N, D) — raw node features
            graph:    HeteroData — the transaction graph

        Returns:
            dict with 'fraud_prob' (float list) and 'fraud_label' (int list)
        """
        # Normalize
        feat_tensor = torch.from_numpy(features).float()
        feat_tensor = (feat_tensor - self.feat_mean) / (self.feat_std + 1e-8)
        feat_tensor = feat_tensor.to(self.device)
        graph = graph.to(self.device)

        with torch.no_grad():
            logits = self.model(graph, feat_tensor)
            probs  = torch.softmax(logits, dim=-1)[:, 1]   # P(fraud)
            labels = (probs > 0.5).int()

        return {
            "fraud_prob":  probs.cpu().tolist(),
            "fraud_label": labels.cpu().tolist()
        }
```

---

### 8.2 — REST API (FastAPI)

Create `api/app.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import redis, json

from api.inference import FraudDetector
from gnn.graph_utils import construct_graph, get_edgelists

app = FastAPI(title="Fraud Detection API", version="1.0")

# Load model once at startup
detector = FraudDetector(model_dir="/app/model", data_dir="/app/data")

# Load graph once at startup (for full-graph inference)
edges = get_edgelists("relation*", "/app/data")
graph, features, target_id_to_node, id_to_node = construct_graph(
    "/app/data", edges, "features.csv", "TransactionID"
)

# Redis connection (optional)
try:
    r = redis.Redis(host="redis", port=6379, db=0)
    r.ping()
    USE_REDIS = True
except Exception:
    USE_REDIS = False

class PredictRequest(BaseModel):
    transaction_ids: List[int]

class PredictResponse(BaseModel):
    transaction_ids: List[int]
    fraud_probabilities: List[float]
    fraud_labels: List[int]

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Look up node indices
    indices = []
    for tid in req.transaction_ids:
        if tid not in target_id_to_node:
            raise HTTPException(status_code=404, detail=f"TransactionID {tid} not in graph")
        indices.append(target_id_to_node[tid])
    
    # Get features — from Redis or from graph
    if USE_REDIS:
        feat_rows = []
        for tid in req.transaction_ids:
            raw = r.get(f"feat:{tid}")
            if raw:
                feat_rows.append(np.array(json.loads(raw), dtype=np.float32))
            else:
                # Fallback to graph features
                feat_rows.append(features[target_id_to_node[tid]])
        feat_matrix = np.stack(feat_rows)
    else:
        feat_matrix = features[indices]

    result = detector.predict(feat_matrix, graph)
    
    return PredictResponse(
        transaction_ids=req.transaction_ids,
        fraud_probabilities=result["fraud_prob"],
        fraud_labels=result["fraud_label"]
    )

@app.get("/health")
def health():
    return {"status": "ok", "redis": USE_REDIS}
```

---

### 8.3 — Dockerfile for Inference Container

Create `docker/Dockerfile.inference`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn redis

COPY gnn/ ./gnn/
COPY api/ ./api/
COPY train.py .

# Model and data are mounted at runtime (not baked in)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

---

### 8.4 — Add Inference Service to Docker Compose

```yaml
# Add to docker/docker-compose.yml
  inference-api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.inference
    ports:
      - "8000:8000"
    volumes:
      - ../model:/app/model:ro      # Read trained model
      - ../data:/app/data:ro        # Read graph data
      - ../features:/app/features:ro
    depends_on:
      - redis
    networks:
      - fraud_net
    environment:
      - REDIS_HOST=redis
```

---

### 8.5 — Test the API

```bash
# Start everything
docker compose -f docker/docker-compose.yml up -d

# Test health
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_ids": [2987004, 2987008, 2987010]}'

# Expected response:
# {
#   "transaction_ids": [2987004, 2987008, 2987010],
#   "fraud_probabilities": [0.03, 0.91, 0.12],
#   "fraud_labels": [0, 1, 0]
# }
```

---

## Hyperparameter Alignment Issue ⚠️

The inference container must use **the same** `n_hidden` and `n_layers` values that were used during training. These are **not currently saved** in `metadata.pkl`.

**Fix:** After training, save hyperparams in metadata:
```python
# In train.py save_model():
pickle.dump({
    'etypes': etype_list,
    'ntype_cnt': ntype_cnt,
    'feat_mean': mean,
    'feat_std': stdev,
    'n_hidden': hyperparams['n_hidden'],   # ADD THIS
    'n_layers': hyperparams['n_layers'],   # ADD THIS
    'in_feats': in_feats,                  # ADD THIS
}, f)
```

---

## ❓ Inputs Needed From You

| Question | Why it matters |
|----------|---------------|
| What `n_hidden` and `n_layers` did you use in training? | Required to rebuild model architecture at inference time |
| Do you want a public endpoint (ngrok / cloud deploy)? | Affects whether you need reverse proxy + auth |
| Batch inference or real-time per-transaction? | Affects API design (batch endpoint vs streaming) |
| GPU available in inference container? | Affects `Dockerfile.inference` base image |

---

## Tradeoffs Summary

| Design Choice | Simple (MVP) | Production-grade |
|---------------|-------------|-----------------|
| Features | Load from `.npy` file | Query Redis per transaction |
| Graph | Load entire graph at startup | Subgraph sampling per node |
| API | Single FastAPI worker | Multiple Gunicorn workers + load balancer |
| Auth | None | API key / JWT |
| Monitoring | Logs only | Prometheus + Grafana |
| Latency | ~200ms (full graph load) | <10ms (subgraph + cached features) |

---

## Full System Startup Sequence

```
1. docker compose up redis
2. python scripts/compress_graph.py       # Step 4
3. docker compose up worker-1 worker-2 worker-3  # Step 5/6
4. python scripts/merge_features.py       # Step 6
5. python scripts/push_features_to_redis.py      # Step 7
6. python train.py ...                    # Already works
7. docker compose up inference-api        # Step 8
8. curl http://localhost:8000/predict ... # Done!
```
