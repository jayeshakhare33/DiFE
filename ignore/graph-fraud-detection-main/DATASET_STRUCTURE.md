# Dataset Structure - Detailed Explanation

## Overview
This document provides a comprehensive explanation of the dataset structure used in the graph fraud detection model, from raw data to processed graph format.

---

## 1. Raw Dataset Structure

### Source: IEEE-CIS Fraud Detection Competition

The project uses two main CSV files from Kaggle's IEEE-CIS Fraud Detection competition:

### A. `train_transaction.csv`
This is the main transaction table containing transaction-level data.

**Key Columns:**

#### Transaction Identifiers:
- **`TransactionID`** (Primary Key): Unique identifier for each transaction
  - Range: ~2987000 to ~3577539
  - Total: ~590,540 transactions
  - This becomes the `target` node type in the graph

#### Transaction Amount:
- **`TransactionAmt`**: Payment amount in USD
  - Used as a feature (log10 transformed)
  - Example values: 68.5, 29.0, 99.99, etc.

#### Product Information:
- **`ProductCD`**: Product code (categorical)
  - Examples: 'W', 'R', 'C', 'H', 'S'
  - Becomes an entity node type in the graph

#### Card Information (Identity Columns):
- **`card1`**: Card identifier (numeric)
- **`card2`**: Card identifier (numeric)
- **`card3`**: Card identifier (numeric, often 150.0)
- **`card4`**: Card type (categorical, e.g., 'visa', 'mastercard')
- **`card5`**: Card identifier (numeric)
- **`card6`**: Card type (categorical, e.g., 'debit', 'credit')
  - These become entity node types in the graph
  - Each unique card value becomes a node

#### Address Information (Identity Columns):
- **`addr1`**: Address identifier (numeric)
- **`addr2`**: Address identifier (numeric)
  - These become entity node types in the graph

#### Email Domains (Identity Columns):
- **`P_emaildomain`**: Purchaser email domain (categorical)
  - Examples: 'gmail.com', 'yahoo.com', 'hotmail.com', etc.
- **`R_emaildomain`**: Recipient email domain (categorical)
  - These become entity node types in the graph

#### Distance Features:
- **`dist1`**: Distance metric 1 (numeric)
- **`dist2`**: Distance metric 2 (numeric)
  - Used as transaction features

#### Counting Features (C1-C14):
- **`C1`, `C2`, ..., `C14`**: Count-based features
  - Masked features from Vesta
  - Examples: number of cards associated, number of addresses associated
  - Used as transaction features

#### Timedelta Features (D1-D15):
- **`D1`, `D2`, ..., `D15`**: Time-based features
  - Masked features from Vesta
  - Examples: days between transactions, time since last transaction
  - Used as transaction features

#### Match Features (M1-M9):
- **`M1`, `M2`, ..., `M9`**: Categorical match indicators
  - Examples: 
    - Name on card matches address
    - Email matches billing address
    - Card matches address
  - Values: 'T' (True), 'F' (False), or NaN
  - Converted to one-hot encoded features

#### Vesta Engineered Features (V1-V339):
- **`V1`, `V2`, ..., `V339`**: Rich engineered features from Vesta
  - Include rankings, counts, and entity relations
  - All numerical features
  - Used as transaction features

#### Transaction Time:
- **`TransactionDT`**: Timedelta from reference datetime (not actual timestamp)
  - Used for temporal analysis but not directly as a feature

#### Label:
- **`isFraud`**: Binary label (0 = legitimate, 1 = fraud)
  - Used for training and evaluation
  - Fraud rate: ~3.5% (highly imbalanced)

### B. `train_identity.csv`
This table contains identity/device information linked to transactions.

**Key Columns:**

#### Link:
- **`TransactionID`**: Links to `train_transaction.csv`
  - Not all transactions have identity data (left join)

#### Device Information:
- **`DeviceType`**: Type of device (categorical)
  - Examples: 'mobile', 'desktop'
  - Becomes an entity node type

- **`DeviceInfo`**: Device information (categorical)
  - Examples: 'iOS Device', 'MacOS', 'Windows', etc.
  - Becomes an entity node type

#### Identity Features (id_01 through id_38):
- **`id_01`, `id_02`, ..., `id_38`**: Network and digital signature features
  - Masked features for privacy
  - Include:
    - Network connection information (IP, ISP, Proxy, etc.)
    - Digital signature (UA/browser/os/version, etc.)
  - Can be numeric or categorical
  - Each becomes an entity node type in the graph

**Example Identity Data:**
```
TransactionID, id_01, id_02, ..., DeviceType, DeviceInfo
2987000, -10.0, 129080.0, ..., mobile, iOS Device
2987001, -5.0, 110477.0, ..., desktop, MacOS
```

---

## 2. Data Processing Pipeline

### Step 1: Data Loading
```python
transaction_df = pd.read_csv('./ieee-data/train_transaction.csv')
identity_df = pd.read_csv('./ieee-data/train_identity.csv')
```

### Step 2: Train/Test Split
- **Training**: 80% of transactions (first 80%)
- **Test**: 20% of transactions (last 20%)
- Test IDs saved to `data/test.csv`

### Step 3: Feature Extraction

#### Identity Columns (become graph edges):
```python
id_cols = ['card1','card2','card3','card4','card5','card6',
           'ProductCD','addr1','addr2','P_emaildomain','R_emaildomain']
```

#### Categorical Columns (become one-hot features):
```python
cat_cols = ['M1','M2','M3','M4','M5','M6','M7','M8','M9']
```

#### Feature Columns (used as node features):
All columns except:
- `isFraud` (label)
- `TransactionDT` (time, not used as feature)
- Identity columns (used for edges, not features)

#### Feature Processing:
1. **One-hot encoding**: Categorical features (M1-M9) converted to dummy variables
2. **Log transformation**: `TransactionAmt` → `log10(TransactionAmt)`
3. **Missing values**: Filled with 0
4. **Output**: `data/features.csv`
   - Format: `TransactionID,feature1,feature2,...,featureN`
   - No header row
   - ~360 features per transaction

### Step 4: Edge Creation

For each identity column, create an edge list:

```python
# Merge transaction and identity data
full_identity_df = transaction_df[id_cols + ['TransactionID']].merge(
    identity_df, on='TransactionID', how='left'
)

# Create edges for each identity column
for etype in edge_types:  # card1, card2, ..., id_38, DeviceType, etc.
    edgelist = full_identity_df[['TransactionID', etype]].dropna()
    edgelist.to_csv(f'data/relation_{etype}_edgelist.csv', 
                    index=False, header=True)
```

**Edge List Format:**
Each `relation_*_edgelist.csv` file contains:
- Header row: `TransactionID,<entity_type>` (e.g., `TransactionID,card1`)
- Data rows: `transaction_id,entity_value`
- Example:
  ```
  TransactionID,card1
  2987000,13926
  2987001,2755
  2987002,4663
  ```

### Step 5: Label Extraction
```python
transaction_df[['TransactionID', 'isFraud']].to_csv('data/tags.csv', index=False)
```

**Format:**
```
TransactionID,isFraud
2987000,0
2987001,0
2987002,1
```

---

## 3. Processed Graph Structure

### Graph Construction

The processed data is used to build a **heterogeneous graph** using DGL:

### Node Types:

1. **`target`** (TransactionID nodes)
   - One node per transaction
   - ~590,540 nodes
   - Each has a 360-dimensional feature vector
   - Labels: 0 (legitimate) or 1 (fraud)

2. **Entity Nodes** (one type per identity column)
   - **Card nodes**: `card1`, `card2`, `card3`, `card4`, `card5`, `card6`
   - **Product nodes**: `ProductCD`
   - **Address nodes**: `addr1`, `addr2`
   - **Email nodes**: `P_emaildomain`, `R_emaildomain`
   - **Identity nodes**: `id_01` through `id_38`
   - **Device nodes**: `DeviceType`, `DeviceInfo`
   - Each unique value becomes a node
   - No features (learn embeddings during training)

### Edge Types:

Each identity column creates **two edge types** (bidirectional):

1. **Forward edge**: `target → entity_type`
   - Example: `TransactionID → card1`
   - Edge type name: `target<>card1`

2. **Reverse edge**: `entity_type → target`
   - Example: `card1 → TransactionID`
   - Edge type name: `card1<>target`

3. **Self-loop**: `target → target`
   - Edge type name: `self_relation`
   - Every transaction has a self-loop

**Total Edge Types**: ~50+ (2 per identity column + 1 self-loop)

### Graph Statistics:
- **Total Nodes**: 726,345
  - ~590,540 transaction nodes
  - ~135,805 entity nodes (cards, addresses, devices, etc.)
- **Total Edges**: 19,518,802
  - Includes bidirectional edges and self-loops

---

## 4. Example Data Flow

### Example Transaction:
```
TransactionID: 2987000
TransactionAmt: 68.5
card1: 13926
card2: NaN
ProductCD: 'W'
addr1: 315
P_emaildomain: 'gmail.com'
M1: 'T'
M2: 'F'
C1: 1
D1: 30
V1: 0.5
V2: 0.3
...
isFraud: 0
```

### After Processing:

**Features (features.csv):**
```
2987000,1.836,0.5,0.3,1,30,0.5,0.3,...,0,1,0,0,...
```
- TransactionID: 2987000
- TransactionAmt (log10): log10(68.5) = 1.836
- dist1, dist2, C1, D1, V1, V2, ... (numerical features)
- M1='T' → [1,0], M2='F' → [0,1], ... (one-hot encoded)

**Edges (relation_card1_edgelist.csv):**
```
TransactionID,card1
2987000,13926
```

**Graph Representation:**
```
Transaction Node (2987000)
  ├─→ Card Node (13926) [via card1 edge]
  ├─→ Product Node ('W') [via ProductCD edge]
  ├─→ Address Node (315) [via addr1 edge]
  ├─→ Email Node ('gmail.com') [via P_emaildomain edge]
  └─→ Transaction Node (2987000) [via self_relation edge]
```

---

## 5. Key Insights

1. **Sparse Identity Data**: Not all transactions have identity data (left join)
   - Missing values are dropped when creating edges
   - This is why some edge lists have fewer rows than transactions

2. **Multiple Entity Types**: Each identity column creates a separate node type
   - Allows the model to learn different patterns for different entity types
   - Cards vs. addresses vs. devices are treated differently

3. **Bidirectional Edges**: Each relation has forward and reverse edges
   - Allows information to flow both ways
   - Transaction → Entity and Entity → Transaction

4. **Feature vs. Structure**: 
   - **Features**: Transaction attributes (amount, counts, matches, Vesta features)
   - **Structure**: Connections to entities (cards, addresses, devices)
   - Model uses both for classification

5. **High Dimensionality**: 
   - 360 features per transaction
   - 50+ edge types
   - 726K+ nodes
   - 19M+ edges
   - This is why GNNs are powerful - they can handle this complexity

---

## Summary

The dataset transforms from:
- **Raw**: 2 CSV files (transaction + identity tables)
- **Processed**: Multiple CSV files (features, edges, labels)
- **Graph**: Heterogeneous graph with multiple node and edge types
- **Model Input**: Graph structure + node features → Fraud classification

The key innovation is treating fraud detection as a graph problem where transactions are connected through shared entities, allowing the model to detect suspicious patterns that wouldn't be visible from transaction features alone.


