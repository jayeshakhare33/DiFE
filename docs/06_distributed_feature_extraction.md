# Step 6 — Distributed Feature Extraction

## Goal

Run `distributed_feature_extraction.py` across multiple workers (either local threads or Docker containers), producing a combined feature matrix that is ready for:
- Feeding into the existing `train.py` GNN trainer
- Storing in Redis/CSV (Step 7)
- Serving as input to the inference container (Step 8)

---

## What Already Exists (and what's broken)

| Component | State | Issue |
|-----------|-------|-------|
| `DistributedFeaturePipeline` | ✅ Defined | Never actually called end-to-end |
| `distributed_feature_extraction.py` | ✅ CLI entry point | Requires pre-built `./data/` graph files |
| `enhance_features_with_distributed_engineering()` in `train.py` | ⚠️ Partial | Returns `None` when transaction data not passed |
| Feature merging back into `train.py` | ❌ Missing | `np.hstack` branch is unreachable |

---

## The Fix Plan

### 6.1 — Make `train.py` Pass Transaction Data for On-the-fly Extraction

In `train.py`, the function `enhance_features_with_distributed_engineering()` needs the raw `transaction_df` passed to it, but currently `train.py` doesn't load the raw CSV — it only uses the processed graph.

**Fix:** Add an `--transaction-data` argument to `train.py` (already present in `gnn/utils.py` `parse_args`) and wire it through:

```python
# In train.py __main__, after loading graph:
if args.use_distributed_features and DISTRIBUTED_FEATURES_AVAILABLE:
    transaction_df = pd.read_csv(args.transaction_data)  # Load raw CSV here
    enhanced_features = enhance_features_with_distributed_engineering(
        features, g, args, target_id_to_node, transaction_df  # pass df
    )
```

Update `enhance_features_with_distributed_engineering` signature:
```python
def enhance_features_with_distributed_engineering(features, graph, args, target_id_to_node, transaction_df=None):
```

### 6.2 — Pre-extract Features (Recommended Workflow)

Run extraction **before** training so `train.py` just loads the saved `.npy` file:

```bash
# Step A: Run feature extraction (locally, using all CPU cores)
python distributed_feature_extraction.py \
  --transaction-data ./ieee-data/ieee-fraud-detection/train_transaction.csv \
  --identity-data ./ieee-data/ieee-fraud-detection/train_identity.csv \
  --training-dir ./data \
  --output-dir ./features \
  --cache-dir ./feature_cache \
  --n-workers 4

# Step B: Train using pre-extracted features
python train.py \
  --training-dir ./data \
  --feature-dir ./features \
  --use-distributed-features \
  --n-epochs 30
```

### 6.3 — Multi-Container Execution (Docker)

When running inside Docker (from Step 5), each container runs the same script but on a **partition** of the data:

**Partition strategy — shard by `TransactionID` range:**

```python
# scripts/partition_and_distribute.py
import pandas as pd
import os

df = pd.read_csv("./ieee-data/ieee-fraud-detection/train_transaction.csv")
n_workers = 3
chunk_size = len(df) // n_workers

for i in range(n_workers):
    chunk = df.iloc[i*chunk_size : (i+1)*chunk_size]
    chunk.to_csv(f"./data/partition_{i+1}.csv", index=False)
    print(f"Partition {i+1}: {len(chunk)} rows → data/partition_{i+1}.csv")
```

Each container's entrypoint then points to its own partition:
```bash
# Worker 1 reads partition_1.csv, Worker 2 reads partition_2.csv, etc.
python distributed_feature_extraction.py \
  --transaction-data /app/data/partition_${WORKER_ID}.csv \
  ...
```

### 6.4 — Merging Worker Outputs

After all workers complete, merge their output `.npy` files:

```python
# scripts/merge_features.py
import numpy as np, glob, os

feature_files = sorted(glob.glob("./features/worker_*/extracted_features.npy"))
all_features = [np.load(f) for f in feature_files]
merged = np.vstack(all_features)
np.save("./features/merged_features.npy", merged)
print(f"Merged shape: {merged.shape}")
```

---

## Feature Extractor Reference

| Extractor | Type | What it produces |
|-----------|------|-----------------|
| `GraphNeighborAggregator` | graph | mean/std/max/min of neighbor features per node |
| `TemporalFeatureExtractor` | temporal | hour_sin/cos, day_sin/cos, time_diff |
| `StatisticalFeatureExtractor` | statistical | z-scores and abs deviations per numeric column |
| `RiskScoreExtractor` | risk | amount log/zscore, card velocity, addr/email freq |
| `GraphCentralityExtractor` | graph | degree centrality, normalized degree |
| `PatternMatchingExtractor` | risk | rapid tx flag, round amounts, device sharing |
| `CrossFeatureExtractor` | statistical | amt×product, amt×card, dist×addr interactions |

---

## ❓ Inputs Needed From You

| Question | Why it matters |
|----------|---------------|
| Do you want to run extraction **before** training (recommended), or **on-the-fly during training**? | Architecture decision affects Step 7 storage design |
| How many CPU cores does your machine have? | Sets `--n-workers` for optimal throughput |
| Is the raw transaction CSV already in `./ieee-data/ieee-fraud-detection/`? | Extraction can't start without it |
| Do you want graph-based features (`GraphNeighborAggregator`) or only tabular ones? | Graph features are slow but higher quality |

---

## Tradeoffs

| Choice | Advantage | Disadvantage |
|--------|-----------|-------------|
| Pre-extract features | Training is fast, features reusable | Extra disk space, extra step |
| On-the-fly extraction | No extra step | Slows down every training run |
| More workers (higher `--n-workers`) | Faster extraction | High RAM usage |
| Skip graph-based extractors | Much faster | Misses important structural signals |

---

## Expected Output

After this step, `./features/` contains:

```
features/
├── extracted_features.npy      ← Combined feature matrix (N_transactions × N_features)
├── feature_metadata.json       ← Shape info + extractor list
├── graph_neighbor_aggregator.npy
├── temporal_features.npy
├── statistical_features.npy
├── risk_score_features.npy
├── graph_centrality.npy
├── pattern_matching.npy
└── cross_features.npy
```

---

## Next Step → [Step 7: Feature Storage](07_feature_storage.md)
