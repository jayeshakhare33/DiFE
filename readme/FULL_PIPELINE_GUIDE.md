# Full Pipeline Guide: Git Clone to Inference

This guide walks through every step to go from a fresh clone of the repository to running fraud inference on new, unseen transactions. No prior setup is assumed.

---

## System Requirements

- **Python**: 3.10 or 3.11 (3.12+ is not supported by DGL 2.2.1)
- **OS**: Windows / Linux / macOS
- **RAM**: Minimum 8 GB recommended for the `basic` profile (100k transactions)
- **Disk**: ~2 GB for dependencies and generated data

---

## Step 1: Clone the Repository

```powershell
git clone <your-repository-url>
cd DiFE-main
```

---

## Step 2: Create and Activate the Virtual Environment

```powershell
# Create the virtual environment
python -m venv .venv

# Activate it (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Or if using CMD instead of PowerShell
.\.venv\Scripts\activate.bat
```

---

## Step 3: Install Dependencies

```powershell
# Use the venv's pip to install pinned dependencies
# NOTE: torchdata==0.7.1 and dgl==2.2.1 are pinned - do not upgrade them
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> **Important:** Always invoke scripts using `& .\.venv\Scripts\python.exe` to ensure the correct environment is used. Do NOT use a plain `python` command unless you have activated the venv.

---

## Step 4: Generate the Dataset

The project ships with a synthetic fraud data generator that produces a logically consistent dataset with realistic fraud ring patterns (account takeovers, money mules, structuring, cross-border layering).

```powershell
# From the root DiFE-main directory
# Use 'basic' for ~15k users / 100k transactions (recommended for first run)
& .\.venv\Scripts\python.exe storage\data_gen.py --profile basic
```

**Available profiles:**

| Profile  | Users  | Transactions | Fraud Rate |
|----------|--------|--------------|------------|
| `basic`  | 8,000  | 100,000      | 2.2%       |
| `medium` | 16,000 | 250,000      | 2.0%       |
| `million`| 60,000 | 1,000,000    | 1.8%       |

**Output:** The following files will be written to `storage/india_fraud_data_explainable/`:
- `transactions.csv` — One row per transaction with sender, receiver, amount, timestamp, and fraud label
- `users.csv` — User account profiles and risk attributes
- `locations.csv` — City/country reference data

---

## Step 5: Extract Graph Features

The model **does not** compute graph features during training. Features must be pre-computed and saved to disk as Parquet files before training can begin.

> **Why?** Graph statistics like PageRank, centrality, and neighborhood aggregation are expensive to compute. Pre-computing them means you can iterate on model hyperparameters quickly without recomputing features each time.

```powershell
# Run this EVERY TIME you generate a new dataset
# It reads transactions.csv and writes to data/features/
& .\.venv\Scripts\python.exe scripts\temp_extract.py
```

> **Note:** For large profiles (`medium`, `million`), this step can take 10–60 minutes depending on your machine, because it computes NetworkX-based graph centrality measures across all users.

**Output:** The following files will be written to `data/features/`:
- `node_features.parquet` — One row per user, ~50 graph and behavioral features
- `edge_features.parquet` — One row per transaction, ~12 edge-level features

---

## Step 6: Train the Model

Train the GraphSAGE edge classification model. It automatically:
- Splits data **temporally** (oldest 70% = train, middle 15% = val, newest 15% = test)
- Applies class-imbalance weighting for fraud minority class
- Triggers early stopping when validation PR-AUC stops improving

```powershell
& .\.venv\Scripts\python.exe scripts\train_edge_model.py `
  --epochs 40 `
  --patience 10 `
  --output-dir model\edge_fraud_local
```

**Expected training output:**
```
INFO - Epoch 001 | loss=1.3421 | val_pr_auc=0.2105 | val_roc_auc=0.8901 | val_f1=0.3100
INFO - Epoch 002 | loss=1.1234 | val_pr_auc=0.2380 | ...
...
INFO - Early stopping at epoch 22
INFO - Final test metrics | threshold=0.45 precision=0.41 recall=0.87 f1=0.56 ...
```

**Advanced options:**
```powershell
& .\.venv\Scripts\python.exe scripts\train_edge_model.py `
  --transactions-path storage\india_fraud_data_explainable\transactions.csv `
  --hidden-dim 256 `
  --num-layers 3 `
  --lr 0.0005 `
  --epochs 60 `
  --patience 15 `
  --output-dir model\edge_fraud_local
```

**Output:** The following files are written to `model/edge_fraud_local/`:

| File | Contents |
|---|---|
| `model.pt` | Trained PyTorch model weights |
| `metadata.json` | Feature column names, scalers, and optimal threshold |
| `metrics.json` | Final precision, recall, F1, PR-AUC, ROC-AUC per split |
| `training_history.json` | Epoch-by-epoch loss and validation metrics |

---

## Step 7: Prepare New Transaction Data for Inference

Create a CSV file containing the new transactions you want to score. You do **not** need to know whether they are fraud — that is what the model will tell you.

**Minimum required columns:**

| Column | Type | Example |
|---|---|---|
| `sender_id` | string | `U0004231` |
| `receiver_id` | string | `U0011782` |
| `amount` | float | `4500.00` |
| `timestamp` | unix ms or datetime | `1700000100000` |

**Optional but improves accuracy:**

| Column | Example |
|---|---|
| `mode` | `upi`, `neft`, `card` |
| `status` | `success`, `failed` |
| `currency` | `INR` |
| `sender_country` | `IN` |
| `receiver_country` | `IN` |
| `is_cross_border` | `0` or `1` |
| `device_id` | `d_U0001_0_abc123ef` |
| `geo_distance_km` | `312.5` |

**Example `new_transactions.csv`:**
```csv
transaction_id,sender_id,receiver_id,amount,timestamp,mode,status
TXN_A001,U0004231,U0011782,4500.00,1700000100000,upi,success
TXN_A002,U0009988,U0001234,87000.00,1700000200000,neft,success
TXN_A003,U0000050,U0003399,150.00,1700000300000,card,failed
```

---

## Step 8: Run Inference (Score New Transactions)

The scoring script builds a full transaction graph, generates node embeddings for all users, and outputs a fraud probability for every row in your new CSV.

### Best Practice: Score with Historical Context

Providing the full historical transaction file as context allows the model to build a richer graph for your new users' neighbors.

```powershell
& .\.venv\Scripts\python.exe scripts\score_transactions_csv.py `
  --model-dir model\edge_fraud_local `
  --history-transactions-path storage\india_fraud_data_explainable\transactions.csv `
  --new-transactions-path path\to\new_transactions.csv `
  --output-path data\predictions\scored_output.csv
```

### Minimal: Score Without History

Works, but will produce less reliable results for users with few transactions in the new file.

```powershell
& .\.venv\Scripts\python.exe scripts\score_transactions_csv.py `
  --model-dir model\edge_fraud_local `
  --new-transactions-path path\to\new_transactions.csv `
  --output-path data\predictions\scored_output.csv
```

---

## Step 9: Interpret the Output

The output CSV at `data/predictions/scored_output.csv` contains your original transaction rows plus three new columns appended on the right:

| New Column | Type | Meaning |
|---|---|---|
| `fraud_probability` | float (0.0–1.0) | Model's confidence that this transaction is fraudulent |
| `predicted_is_fraud` | int (0 or 1) | `1` if `fraud_probability >= model_threshold`, else `0` |
| `model_threshold` | float | The cutoff learned during training (e.g. `0.45`) |

**Example output:**
```csv
transaction_id,sender_id,receiver_id,amount,...,fraud_probability,predicted_is_fraud,model_threshold
TXN_A001,U0004231,U0011782,4500.00,...,0.09,0,0.45
TXN_A002,U0009988,U0001234,87000.00,...,0.91,1,0.45
TXN_A003,U0000050,U0003399,150.00,...,0.22,0,0.45
```

`TXN_A002` is flagged as **fraud** because `0.91 >= 0.45`.

---

## Cheat Sheet: Full Command Sequence

```powershell
# 1. Clone and enter the repo
git clone <your-repo-url>
cd DiFE-main

# 2. Set up environment
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Generate dataset
& .\.venv\Scripts\python.exe storage\data_gen.py --profile basic

# 4. Extract graph features
& .\.venv\Scripts\python.exe scripts\temp_extract.py

# 5. Train model
& .\.venv\Scripts\python.exe scripts\train_edge_model.py `
  --epochs 40 --patience 10 --output-dir model\edge_fraud_local

# 6. Score new transactions
& .\.venv\Scripts\python.exe scripts\score_transactions_csv.py `
  --model-dir model\edge_fraud_local `
  --history-transactions-path storage\india_fraud_data_explainable\transactions.csv `
  --new-transactions-path path\to\new_transactions.csv `
  --output-path data\predictions\scored_output.csv
```

---

## Troubleshooting

**`ValueError: Node feature row count X does not match graph node count Y`**  
Your feature parquet files are out of sync with the current dataset. Re-run Step 4 (feature extraction) after any data generation step.

**`FileNotFoundError: transactions.csv not found`**  
Always run scripts from the **root `DiFE-main` directory**, not from inside a subdirectory like `scripts/`.

**DGL import fails with `torchdata.datapipes` error**  
The virtual environment has an incompatible version of `torchdata`. Reinstall with the exact pinned version:
```powershell
& .\.venv\Scripts\python.exe -m pip install torchdata==0.7.1
```

**All predictions are `0` (no fraud flagged)**  
The model threshold may be too high for your dataset distribution. Check `model/edge_fraud_local/metrics.json` for class balance and consider retraining with a larger dataset using `--profile medium`.
