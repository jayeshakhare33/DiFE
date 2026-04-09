# Step 4 — Graph Compression & Distribution

## Goal

After `gnn/graph_utils.py` (via `train.py`) constructs the graph and writes edge lists + features to `./data/`, this step:

1. **Compresses** the graph (edge lists + node features) into a ZIP archive.
2. **Distributes** a full copy of that archive to every container that will run feature extraction.

This mirrors the top branch of the architecture diagram:
```
Graph Construction → Graph Compression (ZIP) → Graph Distribution (Copy to All Containers)
```

---

## What the Graph Data Looks Like (already produced)

After the graph construction pipeline (`train.py` → `gnn/graph_utils.py`) runs, `./data/` contains:

| File | Description |
|------|-------------|
| `features.csv` | Per-transaction numerical features (node table) |
| `relation_*.csv` | Edge lists, one per relationship type (email, card, device, etc.) |
| `tags.csv` | Fraud labels per TransactionID |

---

## Implementation Plan

### 4.1 — Compression Script

Create `scripts/compress_graph.py`:

```python
import zipfile, os, glob, argparse

def compress_graph(data_dir: str, output_zip: str):
    files = (
        glob.glob(os.path.join(data_dir, "features.csv")) +
        glob.glob(os.path.join(data_dir, "relation_*.csv")) +
        glob.glob(os.path.join(data_dir, "tags.csv"))
    )
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=os.path.basename(f))
    print(f"Graph compressed → {output_zip} ({os.path.getsize(output_zip)/1e6:.1f} MB)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--output-zip", default="./graph_snapshot.zip")
    args = ap.parse_args()
    compress_graph(args.data_dir, args.output_zip)
```

**Run it:**
```bash
python scripts/compress_graph.py --data-dir ./data --output-zip ./graph_snapshot.zip
```

### 4.2 — Distribution

The distribution method depends on **where your containers run**:

#### Option A — Single machine (Docker Compose with bind mounts) ✅ Recommended for local dev

No file transfer needed — all containers mount the **same host directory** via Docker volume.

```yaml
# docker-compose.yml (simplified)
services:
  worker-1:
    volumes:
      - ./data:/app/data:ro        # read-only shared graph data
  worker-2:
    volumes:
      - ./data:/app/data:ro
```

**Tradeoffs:**
- ✅ Zero copy overhead — all containers see the same on-disk files
- ✅ Simple to set up
- ❌ All workers share host disk I/O; can be a bottleneck for very large graphs

#### Option B — True multi-machine (copy via rsync / scp)

```bash
# Copy the zip to each remote host
for HOST in worker1 worker2 worker3; do
  scp graph_snapshot.zip $HOST:/app/data/
  ssh $HOST "cd /app/data && unzip -o graph_snapshot.zip"
done
```

**Tradeoffs:**
- ✅ Each worker has a local copy → no shared I/O bottleneck
- ❌ Large upfront transfer time (graph can be several hundred MB)
- ❌ Requires SSH access to all worker hosts

#### Option C — Shared network storage (NFS / S3 mount)

Mount the same S3 bucket or NFS share to all containers.

```yaml
volumes:
  graph_data:
    driver: local
    driver_opts:
      type: nfs
      o: addr=<NAS_IP>,ro
      device: ":/exports/graph_data"
```

**Tradeoffs:**
- ✅ One source of truth for graph data
- ❌ Requires network storage infrastructure

---

## ❓ Inputs Needed From You

| Question | Why it matters |
|----------|---------------|
| Are all containers on the same machine? | Determines Option A vs B vs C above |
| Do you have Docker Desktop installed? | Needed for any container-based approach |
| How big is your `./data/` directory right now? | Affects transfer times for multi-machine |

---

## Tradeoffs Summary

| Concern | Recommendation |
|---------|---------------|
| Simplicity | Option A (bind mount) |
| Performance at scale | Option B (local copy per worker) |
| Enterprise / cloud | Option C (S3/NFS) |

---

## Next Step → [Step 5: Docker Setup](05_docker_setup.md)
