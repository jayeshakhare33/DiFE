# Standalone Edge Model: End-to-End Training & Testing Guide

This guide provides the exact commands and steps required to generate data, extract features, train the GraphSAGE fraud detection model, and test it on new transactions.

All commands assume you are running them from the root directory of the project (`DiFE-main`).

---

## 1. Environment Setup

Before running the scripts, ensure your Python virtual environment is activated and dependencies are installed.

```powershell
# Create a virtual environment (if you haven't already)
python -m venv .venv

# Install dependencies (requires torchdata==0.7.1 and dgl==2.2.1)
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 2. Generate the Dataset

We use a synthetic data generator that creates logically consistent fraud rings (account takeovers, money mules, structuring). 

```powershell
# Generate the default 'basic' profile (~15k users, 100k transactions)
& .\.venv\Scripts\python.exe storage\data_gen.py

# OR generate a specific profile (options: basic, medium, million)
& .\.venv\Scripts\python.exe storage\data_gen.py --profile basic
```
*Output: `users.csv`, `transactions.csv`, and `locations.csv` will be saved to `storage/india_fraud_data_explainable/`.*

---

## 3. Extract Graph Features

The training model **does not** compute graph math (centrality, PageRank, etc.) on the fly. You must extract features into `.parquet` files before training.

```powershell
# Run the local extraction script
& .\.venv\Scripts\python.exe scripts\temp_extract.py
```
*(Note: If you are using the full distributed architecture, use `scripts\extract_all_features.py` instead to utilize Neo4j and Redis).*

*Output: `node_features.parquet` and `edge_features.parquet` will be saved to `data/features/`.*

---

## 4. Train the Model

Train the GraphSAGE Edge Classifier. The script automatically handles splitting the data temporally (oldest for training, newest for testing) to prevent data leakage.

```powershell
& .\.venv\Scripts\python.exe scripts\train_edge_model.py `
  --epochs 40 `
  --patience 10 `
  --output-dir model\edge_fraud_local
```

### Advanced Training Options:
If you want to train on a different file or tweak the neural network:
```powershell
& .\.venv\Scripts\python.exe scripts\train_edge_model.py `
  --transactions-path "path/to/custom_transactions.csv" `
  --hidden-dim 256 `
  --num-layers 3 `
  --lr 0.001 `
  --output-dir model\custom_model
```

*Output: The trained weights (`model.pt`), scaling variables (`metadata.json`), and results (`metrics.json`) will be saved in `model/edge_fraud_local/`.*

---

## 5. Score / Test the Model (Inference)

To test the model on new data, use the scoring script. For the best accuracy, you should provide a "historical" transaction file so the model can build the graph context, alongside the "new" file containing the rows you actually want to score.

```powershell
& .\.venv\Scripts\python.exe scripts\score_transactions_csv.py `
  --model-dir model\edge_fraud_local `
  --history-transactions-path storage\india_fraud_data_explainable\transactions.csv `
  --new-transactions-path path\to\your\new_test_transactions.csv `
  --output-path data\predictions\scored_transactions.csv
```

### Understanding the Output
The output CSV (`scored_transactions.csv`) will append three new columns to your original data:
1. `fraud_probability`: A score from 0.0 to 1.0.
2. `model_threshold`: The optimal cutoff learned during training.
3. `predicted_is_fraud`: `1` if the probability is >= the threshold, otherwise `0`.
