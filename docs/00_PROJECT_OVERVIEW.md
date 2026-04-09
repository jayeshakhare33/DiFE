# Graph Fraud Detection — Implementation Plan

> **Status:** Feature Extraction stage onward needs to be built. Steps 1–3 (Data Preprocessing → Graph Construction) are already working.

## Workflow Overview

```
Data Sources (CSV)
      │
      ▼
Data Preprocessing          ✅  DONE  (gnn/data.py + gnn/graph_utils.py)
Clean, Validate, Engineer
      │
      ▼
Graph Construction          ✅  DONE  (gnn/graph_utils.py + train.py)
Nodes, Edges, Features
      │
      ├──────────────────────► Graph Compression (ZIP)   🔲 Step 4
      │                                │
      │                                ▼
      │                    Graph Distribution to Containers  🔲 Step 5
      │                       (Docker Compose setup)
      │                    Container 1 │ Container 2 │ Container N
      │                                ▼
      └──────────────────────► Feature Extraction         🔲 Step 6
                               Node Embeddings, Predictions
                                         │
                              ┌──────────┴──────────┐
                              ▼                     ▼
                       Feature Storage        GNN Training     ✅ DONE
                       (Redis / CSV / API)    (train.py)
                                         │
                                         ▼
                               ML Container for Inference    🔲 Step 7
                               Docker Deployment
                                         │
                                         ▼
                               Inference / Fraud Detection API 🔲 Step 8
```

## Document Index

| # | Document | Topic | Status |
|---|----------|--------|--------|
| 04 | [04_graph_compression_distribution.md](04_graph_compression_distribution.md) | Compress graph & copy to containers | 🔲 To Build |
| 05 | [05_docker_setup.md](05_docker_setup.md) | Docker Compose, container networking | 🔲 To Build |
| 06 | [06_distributed_feature_extraction.md](06_distributed_feature_extraction.md) | Run feature extraction across N workers | 🔲 To Build |
| 07 | [07_feature_storage.md](07_feature_storage.md) | Store features in Redis/CSV/API | 🔲 To Build |
| 08 | [08_inference_deployment.md](08_inference_deployment.md) | ML container + fraud detection REST API | 🔲 To Build |

## What Is Already Working

| Component | File | Notes |
|-----------|------|-------|
| Data loading & preprocessing | `gnn/data.py` + `gnn/graph_utils.py` | Reads processed edge lists + features, builds PyG HeteroData |
| Graph construction | `gnn/graph_utils.py` | Builds a PyG HeteroData object |
| GNN model definition | `gnn/pytorch_model.py` | HeteroRGCN model |
| Training loop | `train.py` | Full-graph RGCN training |
| Feature extractor classes | `gnn/feature_engineering.py` | Registry + Extractors defined |
| Advanced extractors | `gnn/advanced_features.py` | Centrality, Pattern, Cross-feature |
| Distributed pipeline class | `gnn/distributed_feature_pipeline.py` | ProcessPoolExecutor wrapper |
| Entry-point script | `distributed_feature_extraction.py` | CLI runner |

## Inputs Required From You

See each step document for a detailed list. High-level summary:

| Step | Input Needed |
|------|-------------|
| Step 5 (Docker) | How many containers? Local machine only or multiple hosts? |
| Step 5 (Docker) | Docker Desktop already installed? |
| Step 6 (Features) | Run on local CPU cores, or true multi-machine distributed? |
| Step 7 (Storage) | Redis preferred, or keep it CSV-only? |
| Step 8 (Inference) | REST API or batch inference? Public URL needed? |
