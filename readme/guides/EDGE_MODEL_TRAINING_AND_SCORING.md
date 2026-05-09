# Edge Model Training And Scoring

This guide explains how to:

1. Train the standalone GraphSAGE edge-classification fraud model
2. Save the trained model artifacts
3. Use the saved model to score a new transaction CSV

The workflow is intentionally independent of deployment strategy. You can train locally now and later load the same model artifacts from an EC2 service that reads transaction data from S3.

The current training path is:

- CPU-only
- based on precomputed parquet features already stored under `data/features`
- not intended to recompute node or edge features during training

## Files Used

- Training script: `scripts/train_edge_model.py`
- Scoring script: `scripts/score_transactions_csv.py`
- Model architecture: `gnn_training/edge_model.py`
- Default input dataset: `storage/india_fraud_data_explainable/transactions.csv`

## What The Model Learns

The model treats:

- `user/account` as a node
- `transaction` as a directed edge from `sender_id` to `receiver_id`
- `is_fraud_txn` as the edge label

It uses:

- node features extracted from transaction behavior and graph topology
- edge features extracted from each transaction row
- a GraphSAGE encoder for node embeddings
- an MLP edge scorer for fraud probability

## Prerequisites

Create and use the local virtual environment:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Notes:

- `requirements.txt` pins `torchdata==0.7.1` for compatibility with `dgl==2.2.1`
- if you already have the virtual environment, you only need the install command once

## Train The Model

Basic training command:

```powershell
& .\.venv\Scripts\python.exe scripts\train_edge_model.py `
  --epochs 40 `
  --patience 10 `
  --output-dir model\edge_fraud_local
```

What this command does:

- loads the transaction CSV from `config.yaml` by default
- builds the directed transaction graph
- loads node features from `data\features\node_features.parquet`
- loads edge features from `data\features\edge_features.parquet`
- removes label-derived fraud propagation features by default for safer training
- uses a time-based train/validation/test split
- runs on CPU only
- saves the best model and metadata

### Important Training Options

Use a different transaction file:

```powershell
& .\.venv\Scripts\python.exe scripts\train_edge_model.py `
  --transactions-path path\to\transactions.csv `
  --output-dir model\edge_fraud_custom
```

Use different feature parquet files:

```powershell
& .\.venv\Scripts\python.exe scripts\train_edge_model.py `
  --node-features-path data\features\node_features.parquet `
  --edge-features-path data\features\edge_features.parquet `
  --output-dir model\edge_fraud_cached
```

### Training Outputs

After training, the output directory contains:

- `model.pt`: trained model weights
- `metadata.json`: feature column order, scalers, threshold, and model settings
- `metrics.json`: train, validation, and test metrics
- `training_history.json`: epoch-by-epoch training log

Example:

```text
model/edge_fraud_local/
  model.pt
  metadata.json
  metrics.json
  training_history.json
```

## Score A New Transaction CSV

Use `scripts/score_transactions_csv.py` to score new transaction rows with the trained model.

### Best Practice

Provide both:

- a historical transaction CSV for graph context
- a new transaction CSV containing only the rows you want to score

This gives the model much better node and edge context than scoring the new rows in isolation.

### Scoring With History

```powershell
& .\.venv\Scripts\python.exe scripts\score_transactions_csv.py `
  --model-dir model\edge_fraud_local `
  --history-transactions-path path\to\history.csv `
  --new-transactions-path path\to\new_transactions.csv `
  --output-path data\predictions\scored_transactions.csv
```

### Scoring Without History

This works, but results are usually less reliable because the graph context is limited to only the new CSV:

```powershell
& .\.venv\Scripts\python.exe scripts\score_transactions_csv.py `
  --model-dir model\edge_fraud_local `
  --new-transactions-path path\to\new_transactions.csv `
  --output-path data\predictions\scored_transactions.csv
```

## Required Columns In The New CSV

Your new transaction CSV should contain at least:

- `sender_id`
- `receiver_id`
- `amount`
- `timestamp`

The current pipeline also benefits from these columns when available:

- `mode`
- `status`
- `currency`
- `sender_country`
- `receiver_country`
- `is_cross_border`
- `device_id`
- `geo_distance_km`

If the training label column exists in the new CSV, it is preserved in the output for comparison, but it is not required for scoring.

## Output Of The Scoring Script

The output CSV contains the original new transaction rows plus:

- `fraud_probability`
- `predicted_is_fraud`
- `model_threshold`

Example output columns:

```text
transaction_id,sender_id,receiver_id,amount,timestamp,...,fraud_probability,predicted_is_fraud,model_threshold
```

Interpretation:

- `fraud_probability`: model-estimated probability of fraud for the transaction
- `predicted_is_fraud`: `1` when `fraud_probability >= model_threshold`, else `0`
- `model_threshold`: threshold saved from training metadata

## Example End-To-End Workflow

Train:

```powershell
& .\.venv\Scripts\python.exe scripts\train_edge_model.py `
  --epochs 40 `
  --patience 10 `
  --output-dir model\edge_fraud_local
```

Score a new CSV:

```powershell
& .\.venv\Scripts\python.exe scripts\score_transactions_csv.py `
  --model-dir model\edge_fraud_local `
  --history-transactions-path storage\india_fraud_data_explainable\transactions.csv `
  --new-transactions-path path\to\new_transactions.csv `
  --output-path data\predictions\new_transactions_scored.csv
```

## Column Mapping Notes

Both scripts read defaults from `config.yaml`. If your CSV uses different column names, override them with:

- `--sender-col`
- `--receiver-col`
- `--amount-col`
- `--timestamp-col`
- `--fraud-col`

Example:

```powershell
& .\.venv\Scripts\python.exe scripts\score_transactions_csv.py `
  --model-dir model\edge_fraud_local `
  --new-transactions-path path\to\new_transactions.csv `
  --sender-col src_account `
  --receiver-col dst_account `
  --amount-col txn_amount `
  --timestamp-col txn_time `
  --output-path data\predictions\custom_scored.csv
```

## Operational Notes

- The trained model artifact is deployment-independent. You can later copy `model.pt` and `metadata.json` to an EC2 instance and load them there.
- Training currently assumes the precomputed parquet features already exist in `data/features`.
- For production, prefer scoring new rows against a historical transaction store instead of a new CSV alone.
- The current model is a first production-style training path, not the final serving system.
- Before building a web UI, the next logical step is wrapping `score_transactions_csv.py` logic into an inference API that reads new transactions from request payloads or S3-backed files.

## Troubleshooting

If DGL import fails with `torchdata.datapipes` errors:

- make sure the virtual environment uses the versions in `requirements.txt`
- reinstall dependencies inside `.venv`

If training fails because feature files are missing:

- confirm `data\features\node_features.parquet` exists
- confirm `data\features\edge_features.parquet` exists
- or pass explicit `--node-features-path` and `--edge-features-path` values

If scoring fails because columns are missing:

- inspect the new CSV header
- either rename the columns to match the training config
- or pass explicit column override flags

If scoring works but all rows are predicted as fraud or all as non-fraud:

- inspect `metrics.json` from training
- review class imbalance and threshold choice
- retrain with a larger and more realistic historical dataset
