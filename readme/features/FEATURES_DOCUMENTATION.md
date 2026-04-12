# Features Documentation

This document provides a comprehensive list of all features calculated in the GNN Fraud Detection system, organized by category with their applications and use cases.

## Overview

The system calculates **62 total features**:
- **50 Node Features** (User-level features)
- **12 Edge Features** (Transaction-level features)

These features are extracted from transaction data and graph structure to enable effective fraud detection using Graph Neural Networks.

---

## Node Features (User-Level)

Node features are calculated for each user/transaction node in the graph. These features capture user behavior, network position, temporal patterns, and fraud risk indicators.

### A. Transaction Statistics Features (15 features)

These features capture the transaction activity and financial patterns of each user.

| # | Feature Name | Description | Application |
|---|--------------|-------------|-------------|
| 1 | `total_transactions_sent` | Total number of outgoing transactions | Identifies highly active senders; fraudsters often have unusual transaction volumes |
| 2 | `total_transactions_received` | Total number of incoming transactions | Identifies users receiving many transactions; may indicate money laundering patterns |
| 3 | `avg_transaction_amount_sent` | Average amount of sent transactions | Detects users with unusually high or low average amounts |
| 4 | `avg_transaction_amount_received` | Average amount of received transactions | Identifies recipients with suspicious average amounts |
| 5 | `max_transaction_amount_sent` | Maximum transaction amount sent | Flags users making unusually large transactions |
| 6 | `max_transaction_amount_received` | Maximum transaction amount received | Identifies recipients of large transactions |
| 7 | `std_transaction_amount_sent` | Standard deviation of sent amounts | Measures transaction amount variability; high variance may indicate suspicious behavior |
| 8 | `std_transaction_amount_received` | Standard deviation of received amounts | Identifies inconsistent receiving patterns |
| 9 | `total_amount_sent` | Total sum of all sent transactions | Tracks total outflow; useful for detecting large-scale fraud |
| 10 | `total_amount_received` | Total sum of all received transactions | Tracks total inflow; identifies potential money laundering |
| 11 | `net_flow` | Difference between received and sent amounts | Identifies users with significant net gains/losses |
| 12 | `transaction_frequency` | Transactions per day | Detects burst activity or suspiciously high frequency |
| 13 | `unique_receivers` | Number of unique recipients | Identifies users sending to many different accounts (potential fraud rings) |
| 14 | `unique_senders` | Number of unique senders | Identifies users receiving from many sources (potential money laundering) |
| 15 | `avg_time_between_transactions` | Average time gap between transactions | Detects rapid-fire transactions or unusual timing patterns |

**Use Cases:**
- Detecting money laundering networks
- Identifying account takeover patterns
- Flagging unusual transaction volumes
- Detecting coordinated fraud attacks

---

### B. Graph Topology Features (12 features)

These features capture the position and importance of each user in the transaction network.

| # | Feature Name | Description | Application |
|---|--------------|-------------|-------------|
| 16 | `in_degree` | Number of incoming edges | Users with high in-degree are central recipients |
| 17 | `out_degree` | Number of outgoing edges | Users with high out-degree are active senders |
| 18 | `total_degree` | Sum of in-degree and out-degree | Overall connectivity in the network |
| 19 | `degree_centrality` | Normalized degree (0-1 scale) | Measures local importance; fraudsters may have unusual centrality |
| 20 | `betweenness_centrality` | Fraction of shortest paths passing through node | Identifies bridge nodes connecting different parts of network |
| 21 | `closeness_centrality` | Average distance to all other nodes | Measures how quickly a user can reach others; useful for fraud propagation |
| 22 | `pagerank_score` | PageRank importance score | Identifies influential nodes in the network |
| 23 | `clustering_coefficient` | Density of local neighborhood | Measures how tightly connected neighbors are; fraud rings often have high clustering |
| 24 | `katz_centrality` | Centrality based on number of paths | Captures influence through multiple hops |
| 25 | `eigenvector_centrality` | Centrality based on connections to important nodes | Identifies users connected to influential accounts |
| 26 | `average_neighbor_degree` | Average degree of connected neighbors | Detects users connected to highly active accounts |
| 27 | `triangles_count` | Number of triangles involving the node | Identifies tightly-knit groups; fraud rings often form triangles |

**Use Cases:**
- Identifying key players in fraud networks
- Detecting fraud rings and coordinated attacks
- Finding bridge accounts connecting legitimate and fraudulent users
- Understanding network structure for fraud propagation

---

### C. Temporal Features (10 features)

These features capture time-based patterns in user behavior.

| # | Feature Name | Description | Application |
|---|--------------|-------------|-------------|
| 28 | `account_age_days` | Days between first and last transaction | New accounts are higher risk; very old accounts may be compromised |
| 29 | `first_transaction_timestamp` | Timestamp of first transaction | Identifies new users or account creation time |
| 30 | `last_transaction_timestamp` | Timestamp of most recent transaction | Detects dormant accounts that suddenly become active |
| 31 | `time_since_last_transaction` | Days since last transaction | Identifies recently inactive accounts (potential takeover) |
| 32 | `transactions_last_24h` | Transaction count in last 24 hours | Detects burst activity or rapid-fire fraud attempts |
| 33 | `transactions_last_7d` | Transaction count in last 7 days | Measures recent activity levels |
| 34 | `transactions_last_30d` | Transaction count in last 30 days | Tracks medium-term activity patterns |
| 35 | `hour_of_day_mode` | Most common hour for transactions | Identifies unusual timing patterns (e.g., transactions at 3 AM) |
| 36 | `day_of_week_mode` | Most common day of week | Detects deviations from normal patterns |
| 37 | `transaction_time_variance` | Variance in transaction timestamps | High variance indicates irregular patterns; low variance may indicate automation |

**Use Cases:**
- Detecting account takeover (sudden activity after dormancy)
- Identifying automated fraud (consistent timing patterns)
- Flagging unusual transaction times
- Detecting burst attacks (many transactions in short time)

---

### D. Behavioral Features (8 features)

These features capture behavioral patterns that may indicate fraud.

| # | Feature Name | Description | Application |
|---|--------------|-------------|-------------|
| 38 | `round_amount_ratio` | Ratio of round-number amounts (e.g., $100.00) | Fraudsters often use round numbers; legitimate transactions are more varied |
| 39 | `threshold_amount_ratio` | Ratio of common threshold amounts (100, 500, 1000, etc.) | Detects structured transactions designed to avoid detection |
| 40 | `transaction_mode_diversity` | Diversity of transaction methods used | Low diversity may indicate automated fraud; high diversity may indicate account compromise |
| 41 | `failed_transaction_ratio` | Ratio of failed transactions | High failure rate may indicate testing or brute-force attacks |
| 42 | `reversal_ratio` | Ratio of reversed transactions | Frequent reversals may indicate fraud or disputes |
| 43 | `cross_border_ratio` | Ratio of cross-border transactions | International transactions have higher fraud risk |
| 44 | `high_risk_country_ratio` | Ratio of transactions involving high-risk countries | Flags transactions with known fraud-prone regions |
| 45 | `burst_score` | Measure of rapid transaction bursts | Detects coordinated attacks or testing patterns |

**Use Cases:**
- Detecting structured fraud (round amounts, thresholds)
- Identifying testing/brute-force patterns (high failure rates)
- Flagging international fraud rings
- Detecting account compromise (unusual transaction modes)

---

### E. Fraud Propagation Features (5 features)

These features measure proximity and connection to known fraudulent accounts.

| # | Feature Name | Description | Application |
|---|--------------|-------------|-------------|
| 46 | `connected_to_fraud_count` | Number of direct connections to fraud nodes | Users directly connected to fraud are high risk |
| 47 | `fraud_propagation_score` | Weighted score based on distance to fraud nodes | Measures risk from nearby fraud; closer = higher risk |
| 48 | `distance_to_nearest_fraud` | Shortest path length to nearest fraud node | Users close to fraud are more likely to be fraudulent |
| 49 | `common_neighbors_with_frauds` | Number of shared neighbors with fraud nodes | Identifies users in same network neighborhood as fraud |
| 50 | `fraud_cluster_membership` | Binary indicator if user is in fraud cluster | Direct fraud label (if available) |

**Use Cases:**
- Guilt-by-association detection
- Identifying fraud rings and networks
- Risk scoring based on network proximity
- Detecting fraud propagation patterns

---

## Edge Features (Transaction-Level)

Edge features are calculated for each transaction (edge) in the graph. These features capture transaction-specific characteristics and relationships.

| # | Feature Name | Description | Application |
|---|--------------|-------------|-------------|
| 1 | `amount` | Transaction amount | Direct indicator; unusually large amounts are suspicious |
| 2 | `timestamp` | Transaction timestamp | Enables temporal analysis and pattern detection |
| 3 | `hour_of_day` | Hour of day (0-23) | Detects transactions at unusual hours |
| 4 | `day_of_week` | Day of week (0-6) | Identifies weekend or weekday patterns |
| 5 | `is_weekend` | Binary weekend indicator | Weekend transactions may have different fraud patterns |
| 6 | `transaction_mode` | Payment method or transaction type | Different modes have different risk levels |
| 7 | `time_since_last_between_users` | Time since last transaction between same user pair | Detects rapid back-and-forth transactions (potential fraud) |
| 8 | `amount_percentile_sender` | Percentile of amount relative to sender's history | Flags unusually large/small amounts for this sender |
| 9 | `amount_percentile_receiver` | Percentile of amount relative to receiver's history | Flags unusual amounts for this receiver |
| 10 | `is_reciprocal` | Binary indicator if reverse transaction exists | Reciprocal transactions may indicate money laundering |
| 11 | `reciprocal_time_gap` | Time between reciprocal transactions | Short gaps may indicate structured transactions |
| 12 | `geographic_distance` | Physical distance between sender and receiver | Large distances may indicate fraud or account compromise |

**Use Cases:**
- Real-time transaction scoring
- Detecting structured transactions (reciprocal, rapid)
- Flagging unusual amounts for specific users
- Identifying geographic anomalies

---

## Feature Storage

Features are stored in multiple formats for different use cases:

### Storage Backends

1. **Parquet Format** (Default for Training)
   - Location: `./data/features/`
   - Files: `node_features.parquet`, `edge_features.parquet`
   - Benefits: Fast bulk loading, compressed, columnar format
   - Used by: Training workers

2. **Redis Cache** (For API)
   - Location: Redis in-memory store
   - Benefits: Sub-millisecond access, fast lookups
   - Used by: API services for real-time inference

3. **CSV Format** (Legacy/Backup)
   - Location: `./data/features/`
   - Benefits: Human-readable, easy to inspect
   - Used by: Development and debugging

---

## Feature Engineering Pipeline

### 1. Graph Construction
- Transaction data is converted to a heterogeneous graph
- Nodes represent users/transactions and entities (cards, addresses, devices)
- Edges represent relationships between entities

### 2. Feature Extraction
- **Node Features**: Extracted using `FeatureExtractor.extract_all_features()`
  - Processes all 50 node features in batches
  - Computes graph topology features using NetworkX
  - Aggregates transaction statistics per user
  
- **Edge Features**: Extracted using `EdgeFeatureExtractor.extract_all_edge_features()`
  - Processes all 12 edge features
  - Computes relative features (percentiles, time gaps)
  - Extracts temporal patterns

### 3. Feature Storage
- Features are saved to the configured backend (Parquet/Redis/CSV)
- Node features indexed by node ID
- Edge features indexed by edge ID

### 4. Feature Loading
- Training: Loads features from Parquet for bulk processing
- Inference: Loads features from Redis for fast real-time access
- Fallback: Can load from Parquet if Redis unavailable

---

## Feature Applications by Use Case

### Real-Time Fraud Detection
**Primary Features:**
- Transaction amount and percentiles
- Time-based features (hour, day, time gaps)
- Behavioral features (round amounts, thresholds)
- Geographic distance

**Why:** Fast computation, directly applicable to individual transactions

### Network-Based Detection
**Primary Features:**
- All graph topology features (centrality, clustering)
- Fraud propagation features
- Transaction statistics (unique senders/receivers)
- Network position indicators

**Why:** Captures relationships and network structure that individual transactions miss

### Account Takeover Detection
**Primary Features:**
- Temporal features (time since last transaction, burst activity)
- Behavioral features (transaction mode diversity)
- Transaction statistics (sudden changes in patterns)

**Why:** Identifies sudden changes in account behavior

### Money Laundering Detection
**Primary Features:**
- Transaction statistics (net flow, unique senders/receivers)
- Reciprocal transaction features
- Graph topology (betweenness, clustering)
- Behavioral features (structured amounts)

**Why:** Detects structured patterns and network flows typical of money laundering

### Fraud Ring Detection
**Primary Features:**
- Graph topology (clustering coefficient, triangles)
- Fraud propagation features
- Common neighbors with frauds
- Transaction statistics (unique connections)

**Why:** Identifies tightly-knit groups and coordinated attacks

---

## Feature Statistics

### Node Features Summary
- **Total:** 50 features
- **Categories:** 5 (Transaction Statistics, Graph Topology, Temporal, Behavioral, Fraud Propagation)
- **Computation Complexity:** 
  - Fast: Transaction statistics, basic topology
  - Moderate: Centrality measures (PageRank, Katz)
  - Slow: Betweenness, closeness (for large graphs)

### Edge Features Summary
- **Total:** 12 features
- **Categories:** Transaction attributes, temporal patterns, relationship features
- **Computation Complexity:** All features are fast to compute

---

## Performance Considerations

### Feature Computation Time
- **Node Features:** 
  - Transaction statistics: O(E) where E = number of edges
  - Graph topology: O(V²) to O(V³) for some centrality measures
  - Temporal/Behavioral: O(E)
  - Fraud propagation: O(V × F) where F = number of fraud nodes

- **Edge Features:** O(E) - linear in number of edges

### Optimization Strategies
1. **Caching:** Features are cached in Parquet/Redis after computation
2. **Lazy Computation:** Some expensive features (betweenness, closeness) are skipped for very large graphs
3. **Sampling:** Large graphs use sampling for expensive centrality measures
4. **Parallel Processing:** Graph construction uses parallel workers

---

## Feature Updates

Features are computed during the `build` phase:
```bash
python main.py --mode build
```

To recompute features:
1. Delete existing feature files
2. Run build mode again
3. Features will be regenerated from current graph structure

---

## References

- Feature extraction code: `feature_engineering/feature_extractor.py`
- Edge feature extraction: `feature_engineering/feature_extractor.py` (EdgeFeatureExtractor class)
- Storage backends: `storage/storage_backend.py`
- Main orchestration: `main.py`

---

## Notes

- Some features (e.g., betweenness centrality, closeness centrality) may be set to zero for very large graphs (>5000 nodes) due to computational constraints
- Fraud propagation features require fraud labels to be available in the graph
- Edge features require transaction data with timestamps and amounts
- All features are normalized/scaled during model training

