# Graph Fraud Detection - Architecture Summary

## Data Flow Diagram

```
IEEE-CIS Dataset
├── train_transaction.csv
│   ├── TransactionID (target nodes)
│   ├── TransactionAmt, C1-C14, D1-D15, M1-M9, V1-V339 (features)
│   └── card1-6, ProductCD, addr1-2, P_emaildomain, R_emaildomain (identity)
│
└── train_identity.csv
    ├── TransactionID (link to transactions)
    └── id_01-id_38, DeviceType, DeviceInfo (identity features)
         │
         │ Data Processing (10_data_loader.ipynb)
         │
         ▼
    ┌─────────────────────────────────────┐
    │   Processed Graph Data              │
    ├─────────────────────────────────────┤
    │ features.csv                        │
    │   └─ TransactionID, feat1...featN   │
    │                                      │
    │ tags.csv                            │
    │   └─ TransactionID, isFraud          │
    │                                      │
    │ test.csv                            │
    │   └─ Test TransactionIDs             │
    │                                      │
    │ relation_*_edgelist.csv            │
    │   ├─ relation_card1_edgelist.csv   │
    │   ├─ relation_addr1_edgelist.csv   │
    │   ├─ relation_P_emaildomain_...     │
    │   ├─ relation_id_01_edgelist.csv   │
    │   └─ ... (50+ edge types)          │
    └─────────────────────────────────────┘
         │
         │ Graph Construction (construct_graph)
         │
         ▼
    ┌─────────────────────────────────────┐
    │   Heterogeneous Graph (DGL)        │
    ├─────────────────────────────────────┤
    │                                     │
    │  Node Types:                        │
    │  • target (transactions)            │
    │  • card1, card2, ..., card6         │
    │  • ProductCD                        │
    │  • addr1, addr2                     │
    │  • P_emaildomain, R_emaildomain     │
    │  • id_01, id_02, ..., id_38         │
    │  • DeviceType, DeviceInfo           │
    │                                     │
    │  Edge Types:                        │
    │  • target <-> card1                 │
    │  • target <-> card2                 │
    │  • target <-> addr1                 │
    │  • target <-> P_emaildomain         │
    │  • target <-> id_01                 │
    │  • ... (50+ relation types)         │
    │  • target <-> target (self_relation)│
    │                                     │
    │  Features:                          │
    │  • target nodes: 360-dim features   │
    │  • entity nodes: learned embeddings │
    └─────────────────────────────────────┘
         │
         │ Model Training (train.py)
         │
         ▼
    ┌─────────────────────────────────────┐
    │   HeteroRGCN Model                 │
    ├─────────────────────────────────────┤
    │                                     │
    │  Input:                             │
    │  • Graph structure                  │
    │  • Transaction features (360-dim)   │
    │                                     │
    │  Architecture:                      │
    │  • Embedding layer (entity nodes)   │
    │  • HeteroRGCN Layer 1               │
    │  • LeakyReLU                        │
    │  • HeteroRGCN Layer 2               │
    │  • LeakyReLU                        │
    │  • HeteroRGCN Layer 3               │
    │  • Output Layer (Linear)            │
    │                                     │
    │  Output:                            │
    │  • Logits [fraud, legitimate]       │
    │                                     │
    └─────────────────────────────────────┘
         │
         │ Prediction
         │
         ▼
    ┌─────────────────────────────────────┐
    │   Fraud Classification             │
    ├─────────────────────────────────────┤
    │  • Fraud (1) or Legitimate (0)     │
    │  • Probability scores               │
    └─────────────────────────────────────┘
```

## Model Architecture Details

### HeteroRGCN Layer Structure

```
┌─────────────────────────────────────────────┐
│  HeteroRGCN Layer                          │
├─────────────────────────────────────────────┤
│                                             │
│  For each edge type r:                      │
│    W_r: Linear(in_size → out_size)          │
│                                             │
│  Message Passing:                          │
│    h'_r = W_r * h_src                      │
│                                             │
│  Aggregation:                              │
│    h_dst = mean(neighbors' messages)       │
│                                             │
└─────────────────────────────────────────────┘
```

### Full Model Architecture

```
Input Features (360-dim)
         │
         ▼
┌────────────────────┐
│  Entity Embeddings │  (learned, 360-dim)
│  • card1, card2... │
│  • addr1, addr2... │
│  • id_01, id_02... │
└────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  HeteroRGCN Layer 1                 │
│  • Input: 360-dim                   │
│  • Output: 16-dim (hidden_size)     │
│  • Relation-specific transformations │
└─────────────────────────────────────┘
         │
         ▼
┌────────────────────┐
│  LeakyReLU         │
└────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  HeteroRGCN Layer 2                 │
│  • Input: 16-dim                    │
│  • Output: 16-dim                   │
└─────────────────────────────────────┘
         │
         ▼
┌────────────────────┐
│  LeakyReLU         │
└────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  HeteroRGCN Layer 3                 │
│  • Input: 16-dim                    │
│  • Output: 16-dim                   │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Output Layer (Linear)               │
│  • Input: 16-dim                     │
│  • Output: 2-dim (fraud/legitimate)  │
└─────────────────────────────────────┘
         │
         ▼
    [fraud, legitimate]
```

## Feature Categories

### Transaction Node Features (360 dimensions)

```
┌─────────────────────────────────────────────┐
│  Transaction Features                       │
├─────────────────────────────────────────────┤
│                                             │
│  1. Transaction Amount (1)                  │
│     • TransactionAmt (log10 transformed)    │
│                                             │
│  2. Distance Features (2)                   │
│     • dist1, dist2                          │
│                                             │
│  3. Counting Features (14)                   │
│     • C1, C2, ..., C14                     │
│     • Number of cards/addresses associated  │
│                                             │
│  4. Timedelta Features (15)                  │
│     • D1, D2, ..., D15                     │
│     • Days between transactions             │
│                                             │
│  5. Match Features (9 → one-hot)             │
│     • M1, M2, ..., M9                      │
│     • Name/card/address/email matches       │
│     • Converted to dummy variables          │
│                                             │
│  6. Vesta Features (339)                    │
│     • V1, V2, ..., V339                    │
│     • Engineered rich features              │
│     • Rankings, counts, entity relations    │
│                                             │
└─────────────────────────────────────────────┘
```

## Graph Structure Example

```
Example Graph Structure:

    [Transaction T1] ──card1──> [Card C1]
         │                        │
         │                        │
    addr1│                        │card1
         │                        │
         ▼                        ▼
    [Address A1] <──addr1── [Transaction T2]
         │                        │
         │                        │
    addr1│                        │P_emaildomain
         │                        │
         ▼                        ▼
    [Transaction T3] ──P_emaildomain──> [Email E1]
         │
         │
    id_01│
         │
         ▼
    [Identity I1] <──id_01── [Transaction T4]
```

## Training Process

```
┌─────────────────────────────────────────────┐
│  Training Loop                              │
├─────────────────────────────────────────────┤
│                                             │
│  1. Load graph structure                    │
│  2. Load transaction features               │
│  3. Normalize features (mean=0, std=1)      │
│  4. Load labels (fraud/legitimate)          │
│  5. Create train/test masks                 │
│                                             │
│  For each epoch (700 epochs):               │
│    a. Forward pass:                         │
│       • Get entity embeddings               │
│       • Get transaction features            │
│       • Pass through HeteroRGCN layers      │
│       • Get predictions                     │
│                                             │
│    b. Compute loss:                        │
│       • CrossEntropyLoss                    │
│                                             │
│    c. Backward pass:                       │
│       • Compute gradients                   │
│       • Update parameters (Adam)          │
│                                             │
│    d. Evaluate:                             │
│       • Compute F1 score                    │
│                                             │
│  6. Evaluate on test set:                  │
│     • Precision, Recall, F1                │
│     • ROC-AUC, PR-AUC                      │
│     • Confusion Matrix                     │
│                                             │
└─────────────────────────────────────────────┘
```

## Key Components Summary

### 1. Data Processing
- **Input**: Raw CSV files (transaction + identity)
- **Output**: Graph structure (nodes, edges, features)
- **Key Files**: `features.csv`, `relation_*_edgelist.csv`, `tags.csv`

### 2. Graph Construction
- **Library**: DGL (Deep Graph Library)
- **Type**: Heterogeneous graph
- **Nodes**: Transactions (target) + Entities (cards, addresses, etc.)
- **Edges**: Relations between transactions and entities

### 3. Model Architecture
- **Type**: HeteroRGCN (Heterogeneous Relational GCN)
- **Layers**: 3 HeteroRGCN layers + 1 output layer
- **Hidden Size**: 16
- **Input Features**: 360 dimensions
- **Output**: 2 classes (fraud/legitimate)

### 4. Training
- **Optimizer**: Adam (lr=0.01)
- **Loss**: CrossEntropyLoss
- **Epochs**: 700
- **Evaluation**: F1 score, Precision, Recall, ROC-AUC

### 5. Features Used
- **Direct Features**: Transaction amount, counts, time deltas, matches, Vesta features
- **Graph Features**: Learned from connections to cards, addresses, devices, identities
- **Combined**: Both feature-based and structure-based information


