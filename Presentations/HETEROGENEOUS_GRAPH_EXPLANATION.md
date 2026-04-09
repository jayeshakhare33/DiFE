# Heterogeneous Graphs Explained for Transaction Fraud Detection

## What is a Heterogeneous Graph?

### Simple Definition

A **heterogeneous graph** (also called a **heterogeneous network** or **multi-typed graph**) is a graph where:
- **Nodes can be of different types** (e.g., transactions, cards, addresses, devices)
- **Edges can be of different types** (e.g., "transaction uses card", "transaction from address", "card belongs to device")

### Comparison: Homogeneous vs. Heterogeneous

#### Homogeneous Graph (Simple)
```
All nodes are the same type:
[User] --friends--> [User] --friends--> [User]
     \                                    /
      \--friends-------------------------/
```
- **Example**: Social network where all nodes are "users" and all edges are "friendship"
- **One node type**: User
- **One edge type**: Friendship

#### Heterogeneous Graph (Complex)
```
Different node types connected by different edge types:
[Transaction] --uses--> [Card] --belongs_to--> [Device]
     |                      |
     |--from--> [Address]   |--used_at--> [IP]
     |
     |--product--> [Product]
```
- **Multiple node types**: Transaction, Card, Address, Device, IP, Product
- **Multiple edge types**: "uses", "from", "product", "belongs_to", "used_at"

---

## Why Heterogeneous Graphs for Fraud Detection?

### The Problem with Traditional Approaches

**Traditional ML**: Treats each transaction independently
```
Transaction 1: [features] → Fraud/Not Fraud
Transaction 2: [features] → Fraud/Not Fraud
Transaction 3: [features] → Fraud/Not Fraud
```
- **Missing**: Relationships between transactions
- **Missing**: Shared entities (cards, addresses, devices)
- **Missing**: Network patterns (fraud rings)

### The Power of Heterogeneous Graphs

**Graph-based**: Captures relationships and patterns
```
Transaction 1 --uses--> Card A
Transaction 2 --uses--> Card A  ← Same card!
Transaction 3 --uses--> Card B
Transaction 1 --from--> Address X
Transaction 2 --from--> Address X  ← Same address!
```
- **Captures**: Shared entities reveal fraud patterns
- **Detects**: Fraud rings (multiple cards, same address/device)
- **Identifies**: Compromised entities (card used by many transactions)

---

## Your Transaction Fraud Detection Graph Structure

### Node Types in Your Graph

Based on the IEEE-CIS Fraud Detection dataset, your graph contains:

#### 1. **Target Nodes** (Transactions)
- **Type**: `target` (or `TransactionID`)
- **Purpose**: These are the nodes you want to classify (fraud/not fraud)
- **Features**: 
  - Transaction amount
  - Transaction date/time
  - Vesta engineered features (V1-V339)
  - Counting features (C1-C14)
  - Timedelta features (D1-D15)
  - Match features (M1-M9)
  - Distance features (dist1, dist2)
- **Count**: ~590,540 transactions

#### 2. **Card Nodes**
- **Types**: `card1`, `card2`, `card3`, `card4`, `card5`, `card6`
- **Purpose**: Represent payment card information
- **Features**: None (learned embeddings)
- **Example**: Card number, card type (Visa, Mastercard), card category

#### 3. **Address Nodes**
- **Types**: `addr1`, `addr2`
- **Purpose**: Represent billing/shipping addresses
- **Features**: None (learned embeddings)
- **Example**: Billing address, shipping address

#### 4. **Email Domain Nodes**
- **Types**: `P_emaildomain`, `R_emaildomain`
- **Purpose**: Purchaser and recipient email domains
- **Features**: None (learned embeddings)
- **Example**: gmail.com, yahoo.com, suspicious-domain.com

#### 5. **Product Nodes**
- **Type**: `ProductCD`
- **Purpose**: Product codes for transactions
- **Features**: None (learned embeddings)
- **Example**: Product types (W, C, R, etc.)

#### 6. **Device Nodes**
- **Types**: `DeviceType`, `DeviceInfo`
- **Purpose**: Device information (from identity table)
- **Features**: None (learned embeddings)
- **Example**: Desktop, Mobile, Device fingerprints

#### 7. **Identity Nodes** (Network/System Info)
- **Types**: `id_01` through `id_38`
- **Purpose**: Network connection information, digital signatures
- **Features**: None (learned embeddings)
- **Example**: IP addresses, ISP, Proxy, Browser, OS, etc.

### Edge Types in Your Graph

Each relationship creates **two edge types** (forward and reverse):

#### Example Edge Types:

1. **Transaction → Card**
   - `(target, 'target<>card1', card1)`: Transaction uses card1
   - `(card1, 'card1<>target', target)`: Card1 is used by transaction (reverse)

2. **Transaction → Address**
   - `(target, 'target<>addr1', addr1)`: Transaction from address1
   - `(addr1, 'addr1<>target', target)`: Address1 has transaction (reverse)

3. **Transaction → Email**
   - `(target, 'target<>P_emaildomain', P_emaildomain)`: Transaction from email domain
   - `(P_emaildomain, 'P_emaildomain<>target', target)`: Email domain has transaction (reverse)

4. **Transaction → Product**
   - `(target, 'target<>ProductCD', ProductCD)`: Transaction for product
   - `(ProductCD, 'ProductCD<>target', target)`: Product has transaction (reverse)

5. **Transaction → Device**
   - `(target, 'target<>DeviceInfo', DeviceInfo)`: Transaction from device
   - `(DeviceInfo, 'DeviceInfo<>target', target)`: Device has transaction (reverse)

6. **Transaction → Identity**
   - `(target, 'target<>id_01', id_01)`: Transaction with identity feature
   - `(id_01, 'id_01<>target', target)`: Identity feature in transaction (reverse)

7. **Self-loops**
   - `(target, 'self_relation', target)`: Transaction to itself
   - **Purpose**: Allows transactions to learn from their own features

### Total Graph Statistics (From Your Dataset)

- **Total Nodes**: 726,345
- **Total Edges**: 19,518,802
- **Node Types**: ~50+ different types (target + all entity types)
- **Edge Types**: ~100+ different types (forward + reverse for each relationship)

---

## Visual Example of Your Graph

### Simple Example (3 Transactions)

```
                    ┌─────────────┐
                    │  Card_12345 │
                    └──────┬──────┘
                           │ uses
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐        ┌────▼────┐       ┌────▼────┐
   │Trans_001│        │Trans_002│       │Trans_003│
   └────┬────┘        └────┬────┘       └────┬────┘
        │                  │                  │
        │ from             │ from             │ from
        │                  │                  │
   ┌────▼────┐        ┌────▼────┐       ┌────▼────┐
   │Addr_789 │        │Addr_789 │       │Addr_456 │
   └─────────┘        └─────────┘       └─────────┘
        │                  │
        │                  │
   ┌────▼────┐        ┌────▼────┐
   │Device_A │        │Device_A │
   └─────────┘        └─────────┘
```

**What this reveals:**
- **Trans_001** and **Trans_002** share the same card, address, and device → **Suspicious!**
- **Trans_003** uses different address → **Less suspicious**
- **Pattern**: Multiple transactions sharing entities = potential fraud

### Real-World Fraud Pattern Example

```
Fraud Ring Detection:

Transaction_1 --uses--> Card_A
Transaction_2 --uses--> Card_B
Transaction_3 --uses--> Card_C
Transaction_4 --uses--> Card_D
    │            │            │            │
    │            │            │            │
    └──from──> Address_X <──from──┘            │
         │                                      │
         └──from───────────────────────────────┘
```

**What this reveals:**
- **4 different cards** (Card_A, B, C, D)
- **All from same address** (Address_X)
- **Pattern**: Multiple cards from one address = **Fraud Ring!**

---

## How the Graph is Constructed

### Step 1: Identify Entity Columns

From your transaction data, these columns become **node types**:
- `card1`, `card2`, `card3`, `card4`, `card5`, `card6` → Card nodes
- `addr1`, `addr2` → Address nodes
- `P_emaildomain`, `R_emaildomain` → Email nodes
- `ProductCD` → Product nodes
- `DeviceType`, `DeviceInfo` → Device nodes
- `id_01` through `id_38` → Identity nodes

### Step 2: Create Edges

For each transaction row:
```python
TransactionID = 2987000
card1 = 13926
addr1 = 123
ProductCD = 'W'
DeviceInfo = 'device_xyz'

# Creates edges:
(TransactionID, 'target<>card1', card1)
(TransactionID, 'target<>addr1', addr1)
(TransactionID, 'target<>ProductCD', ProductCD)
(TransactionID, 'target<>DeviceInfo', DeviceInfo)

# Plus reverse edges:
(card1, 'card1<>target', TransactionID)
(addr1, 'addr1<>target', TransactionID)
# etc.
```

### Step 3: Build HeteroData Object

```python
data = HeteroData()

# Add edge indices for each edge type
data[('target', 'target<>card1', 'card1')].edge_index = edge_tensor
data[('card1', 'card1<>target', 'target')].edge_index = reverse_edge_tensor

# Add node features for target nodes
data['target'].x = transaction_features

# Other nodes get learned embeddings (no initial features)
```

---

## Why This Structure Works for Fraud Detection

### 1. **Entity Sharing Detection**
- **Problem**: Same card used by multiple transactions
- **Solution**: Graph connects transactions through shared card nodes
- **Detection**: High connectivity = suspicious

### 2. **Fraud Ring Detection**
- **Problem**: Multiple cards from same address/device
- **Solution**: Graph shows clustering patterns
- **Detection**: Dense subgraphs = fraud rings

### 3. **Compromised Entity Detection**
- **Problem**: Card/address/device used in many fraud transactions
- **Solution**: Graph centrality measures identify compromised entities
- **Detection**: High degree centrality = compromised entity

### 4. **Pattern Recognition**
- **Problem**: Complex fraud patterns across multiple entities
- **Solution**: Graph neural network learns from graph structure
- **Detection**: Learned patterns = fraud indicators

---

## Graph Neural Network Processing

### How the Model Uses the Graph

1. **Message Passing**: Information flows along edges
   ```
   Transaction → Card → Other Transactions using same card
   ```

2. **Neighbor Aggregation**: Each transaction learns from its neighbors
   ```
   Transaction_1 learns from:
   - Card it uses
   - Address it's from
   - Device it's on
   - Other transactions sharing these entities
   ```

3. **Multi-hop Learning**: Information propagates multiple steps
   ```
   Transaction_1 → Card_A → Transaction_2 → Address_X → Transaction_3
   ```
   Transaction_1 can learn about Transaction_3 through shared entities!

---

## Key Characteristics of Your Graph

### 1. **Highly Heterogeneous**
- **50+ node types**: Transactions + all entity types
- **100+ edge types**: Forward + reverse for each relationship
- **Complex structure**: Many different types of relationships

### 2. **Large Scale**
- **726K nodes**: Mix of transactions and entities
- **19.5M edges**: Dense connections
- **Sparse per type**: Each edge type is relatively sparse

### 3. **Bipartite-like Structure**
- Most edges connect **target** (transactions) to **entity** nodes
- Few edges between entity nodes directly
- Structure: Transaction ↔ Entity ↔ Transaction

### 4. **Dynamic Nature**
- New transactions added over time
- New entities (cards, addresses) appear
- Graph structure evolves

---

## Summary

### What is a Heterogeneous Graph?
- Graph with **multiple node types** and **multiple edge types**
- More complex than simple graphs
- Better for real-world data with diverse entities

### Your Transaction Fraud Graph:
- **Node Types**: Transactions (target) + Cards + Addresses + Devices + Emails + Products + Identity features
- **Edge Types**: Transaction ↔ Entity relationships (bidirectional)
- **Purpose**: Detect fraud by analyzing relationships and patterns
- **Size**: 726K nodes, 19.5M edges

### Why It Works:
- **Captures relationships**: Shared entities reveal fraud
- **Detects patterns**: Fraud rings, compromised entities
- **Learns from structure**: Graph neural networks learn from connections
- **Improves accuracy**: Better than treating transactions independently

This heterogeneous graph structure is what makes your fraud detection system powerful - it can see the "big picture" of how transactions, cards, addresses, and devices are all connected!

