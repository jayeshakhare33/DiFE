# Graph Fraud Detection Model - Complete Analysis

## Overview
This project implements a **Heterogeneous Graph Neural Network (GNN)** for fraud detection using the IEEE-CIS Fraud Detection dataset. The model treats fraud detection as a **node classification task** on a heterogeneous graph where transactions are nodes connected to various entities (cards, addresses, devices, etc.).

---

## Dataset Structure

### Source Data (IEEE-CIS Fraud Detection)
The project uses two main CSV files from the IEEE-CIS Fraud Detection competition:

1. **`train_transaction.csv`** - Transaction data
2. **`train_identity.csv`** - Identity/device information linked to transactions

### Data Processing Pipeline

The data goes through the following transformation:

1. **Feature Extraction** → Creates `features.csv` (node features for transactions)
2. **Edge Creation** → Creates multiple `relation_*_edgelist.csv` files (graph edges)
3. **Label Extraction** → Creates `tags.csv` (fraud labels)
4. **Test Split** → Creates `test.csv` (test transaction IDs)

---

## Graph Structure

### Node Types

1. **`target` (TransactionID)** - The main node type we want to classify
   - These are the transaction nodes
   - Each transaction has a feature vector
   - Labels: 0 (legitimate) or 1 (fraud)

2. **Entity Nodes** - Created from identity columns:
   - `card1`, `card2`, `card3`, `card4`, `card5`, `card6` - Card information
   - `ProductCD` - Product code
   - `addr1`, `addr2` - Address information
   - `P_emaildomain`, `R_emaildomain` - Email domains (purchaser/recipient)
   - `id_01` through `id_38` - Identity features (network, device info)
   - `DeviceType`, `DeviceInfo` - Device information

### Edge Types

Edges connect transactions to entities. Each identity column creates a relation type:
- `TransactionID <-> card1`
- `TransactionID <-> card2`
- `TransactionID <-> ProductCD`
- `TransactionID <-> addr1`
- `TransactionID <-> P_emaildomain`
- `TransactionID <-> id_01` through `id_38`
- `TransactionID <-> DeviceType`
- etc.

**Self-loops**: Transactions also have self-relations (`self_relation`) to themselves.

### Graph Statistics
- **Total Nodes**: 726,345
- **Total Edges**: 19,518,802
- **Node Types**: Multiple (target + all entity types)
- **Edge Types**: ~50+ different relation types

---

## Node Features (Transaction Features)

The features for transaction nodes (`target` nodes) are extracted from the transaction table and include:

### 1. **Transaction Amount**
- `TransactionAmt` - Log10 transformed transaction amount

### 2. **Distance Features**
- `dist1`, `dist2` - Distance metrics

### 3. **Counting Features (C1-C14)**
- Count-based features (e.g., number of cards associated, number of addresses associated)
- These are masked features from Vesta

### 4. **Timedelta Features (D1-D15)**
- Time-based features (e.g., days between transactions)
- These are also masked features

### 5. **Match Features (M1-M9)**
- Categorical match indicators (e.g., name matches card, address matches, etc.)
- Converted to dummy/one-hot encoded features
- Examples: name on card matches address, email matches, etc.

### 6. **Vesta Engineered Features (V1-V339)**
- Rich engineered features from Vesta
- Include ranking, counting, and other entity relations
- These are numerical features

### Feature Processing
- **Categorical features** (M1-M9): One-hot encoded using `pd.get_dummies()`
- **TransactionAmt**: Log10 transformation applied
- **Missing values**: Filled with 0
- **Normalization**: Features are normalized (mean=0, std=1) during training

### Final Feature Vector
- **Total dimensions**: ~360 features (TransactionID + TransactionAmt + dist1-2 + C1-C14 + D1-D15 + M1-M9 (one-hot) + V1-V339)
- Stored in `features.csv` format: `TransactionID,feature1,feature2,...,featureN`

---

## Model Architecture

### HeteroRGCN (Heterogeneous Relational Graph Convolutional Network)

The model uses a **Relational Graph Convolutional Network** designed for heterogeneous graphs:

#### Architecture Components:

1. **Node Embeddings** (for non-target nodes):
   - Learnable embeddings for entity nodes (cards, addresses, devices, etc.)
   - Initialized with Xavier uniform initialization
   - Embedding size: 360 (matches input feature size)

2. **HeteroRGCN Layers**:
   - Multiple layers of relational graph convolution
   - Each layer handles different edge types separately
   - Uses relation-specific weight matrices: `W_r` for each relation type `r`
   - Message passing: Mean aggregation of neighbor messages
   - Activation: LeakyReLU between layers

3. **Output Layer**:
   - Final linear layer: `hidden_size → 2` (binary classification)
   - Outputs logits for fraud (1) vs. legitimate (0)

#### Model Hyperparameters (defaults):
- **Hidden size**: 16
- **Number of layers**: 3
- **Embedding size**: 360
- **Learning rate**: 0.01
- **Weight decay**: 5e-4
- **Epochs**: 700
- **Optimizer**: Adam

#### Forward Pass:
1. Get embeddings for entity nodes (from learned embeddings)
2. Get features for target nodes (from input features)
3. Pass through multiple HeteroRGCN layers with message passing
4. Apply LeakyReLU activation between layers
5. Final linear layer outputs logits for target nodes

---

## Training Process

### Data Split
- **Training**: 80% of transactions
- **Test**: 20% of transactions
- **Fraud rate**: ~3.5% (highly imbalanced)

### Training Procedure:
1. **Graph Construction**: Build heterogeneous graph from edge lists
2. **Feature Normalization**: Normalize transaction features
3. **Label Loading**: Load fraud labels (0/1)
4. **Full Graph Training**: Train on entire graph (no mini-batching)
5. **Loss Function**: CrossEntropyLoss
6. **Evaluation**: F1 score during training, full metrics on test set

### Key Training Features:
- **Full graph training**: Uses entire graph at once (not mini-batched)
- **Test masking**: Test transactions are masked during training
- **Self-loops**: Transactions have self-relations for self-information

---

## Features Used for Classification

### Direct Node Features (Transaction Features):
1. **Transaction Amount** (log-transformed)
2. **Counting features** (C1-C14): Number of associated entities
3. **Timedelta features** (D1-D15): Time-based patterns
4. **Match features** (M1-M9): Matching indicators
5. **Vesta features** (V1-V339): Engineered rich features

### Graph Structure Features (Learned via GNN):
The GNN learns to aggregate information from:
1. **Card connections**: Transactions sharing same cards
2. **Address connections**: Transactions from same addresses
3. **Email domain connections**: Transactions with same email domains
4. **Device connections**: Transactions from same devices
5. **Identity connections**: Transactions with same identity features (IP, ISP, Proxy, etc.)
6. **Product connections**: Transactions for same products

### How GNN Uses Graph Structure:
- **Message Passing**: Each transaction node receives messages from its connected entities
- **Neighbor Aggregation**: Information from cards, addresses, devices, etc. is aggregated
- **Multi-hop Reasoning**: Through multiple layers, the model can reason about:
  - Transactions sharing the same card
  - Cards used from multiple addresses
  - Devices used with multiple cards
  - Complex fraud patterns across the network

---

## Classification Logic

The model classifies a transaction as fraud based on:

1. **Direct Features**: The transaction's own features (amount, counts, matches, etc.)
2. **Local Neighborhood**: Entities directly connected (cards, addresses, devices)
3. **Extended Neighborhood**: Entities connected through shared cards/addresses (multi-hop)
4. **Pattern Detection**: Learned patterns of fraudulent behavior in the graph

### Example Fraud Pattern:
- Transaction T1 uses Card C1 from Address A1
- Transaction T2 uses Card C1 from Address A2 (different address)
- Transaction T3 uses Card C2 from Address A1 (same address, different card)
- If T1 is fraud, the model learns that T2 and T3 might also be suspicious due to shared entities

---

## Model Performance

### Results:
- **Precision**: 0.86 (priority metric - avoids false positives)
- **ROC-AUC**: 0.92
- **F1 Score**: Calculated during training

### Confusion Matrix:
```
                    Labels Positive  Labels Negative
Predicted Positive        1435             240
Predicted Negative        2629           113804
```

### Trade-offs:
- **Precision prioritized**: To avoid misclassifying legitimate transactions as fraud (better user experience)
- **High recall less critical**: Missing some fraud is acceptable if precision is high

---

## File Structure

### Input Files (after data processing):
- `data/features.csv` - Transaction node features
- `data/tags.csv` - Transaction labels (TransactionID, isFraud)
- `data/test.csv` - Test transaction IDs
- `data/relation_*_edgelist.csv` - Edge lists for each relation type

### Model Files:
- `model/model.pth` - Trained model weights
- `model/metadata.pkl` - Graph metadata (node types, edge types, normalization params)
- `model/*.csv` - Learned embeddings for entity nodes

### Output Files:
- `output/results.txt` - Training metrics per epoch
- `output/roc_curve.png` - ROC curve visualization
- `output/pr_curve.png` - Precision-Recall curve

---

## Key Insights

1. **Heterogeneous Graph**: The model leverages multiple types of relationships, not just one
2. **Entity Embeddings**: Non-transaction entities get learned embeddings that capture their role in fraud patterns
3. **Graph Structure Matters**: The connections between transactions and entities are as important as the features themselves
4. **Multi-hop Reasoning**: The model can detect complex fraud patterns through multi-layer message passing
5. **Feature Engineering**: Both raw features (Vesta features) and graph structure contribute to classification

---

## Summary

This model combines:
- **Rich transaction features** (360+ dimensions)
- **Graph structure** (19M+ edges connecting transactions to entities)
- **Heterogeneous GNN** (HeteroRGCN) to learn from both

The key innovation is treating fraud detection as a graph problem where transactions are connected through shared entities (cards, addresses, devices), allowing the model to detect suspicious patterns that wouldn't be visible from transaction features alone.


