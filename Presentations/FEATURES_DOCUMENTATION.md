# Complete Feature Extraction Documentation

This document lists all features extracted from graphs and transaction data, organized by category and their specific use cases in fraud detection.

## Feature Categories Overview

1. **Graph-Based Features** - Features derived from graph structure and relationships
2. **Temporal Features** - Time-based patterns and cyclical features
3. **Statistical Features** - Distribution characteristics and statistical measures
4. **Risk-Based Features** - Pattern-based risk indicators
5. **Advanced Features** - Graph centrality, pattern matching, and cross-features

---

## 1. Graph-Based Features

### 1.1 GraphNeighborAggregator Features

**Extractor**: `GraphNeighborAggregator`  
**Type**: Graph-based  
**Source**: Heterogeneous graph structure

#### Features Extracted:

For each edge type connecting to target (transaction) nodes, the following aggregations are computed from neighbor node features:

| Feature | Description | Use Case |
|---------|-------------|----------|
| `neighbor_mean_{i}` | Mean of neighbor features for each feature dimension `i` | Captures average behavior of connected entities (cards, addresses, devices) |
| `neighbor_std_{i}` | Standard deviation of neighbor features | Measures variability in connected entities - high std may indicate suspicious activity |
| `neighbor_max_{i}` | Maximum value among neighbors | Identifies extreme values in connected entities |
| `neighbor_min_{i}` | Minimum value among neighbors | Identifies minimum values in connected entities |
| `neighbor_sum_{i}` | Sum of neighbor features | Captures total activity/volume from connected entities |
| `neighbor_count_{i}` | Number of neighbors (repeated for each feature dim) | Measures connectivity - high connectivity may indicate fraud rings |

**Example Use Cases:**
- **Card sharing detection**: If multiple transactions share the same card, neighbor aggregations capture this pattern
- **Address clustering**: Transactions from same address show similar neighbor patterns
- **Device fingerprinting**: Multiple cards using same device creates distinct neighbor signatures
- **Fraud ring detection**: Highly connected nodes (many neighbors) may indicate organized fraud

**Edge Types Processed:**
- Transaction → Card relationships
- Transaction → Address relationships  
- Transaction → Device relationships
- Transaction → Email domain relationships
- Transaction → Product relationships
- All reverse relationships (Card → Transaction, etc.)

---

### 1.2 GraphCentralityExtractor Features

**Extractor**: `GraphCentralityExtractor`  
**Type**: Graph-based  
**Source**: Graph structure analysis

#### Features Extracted:

| Feature | Description | Use Case |
|---------|-------------|----------|
| `degree_centrality` | Total number of connections (edges) for each transaction node | High degree = highly connected transaction, may indicate fraud ring participation |
| `normalized_degree_centrality` | Degree centrality normalized by maximum degree in graph | Relative importance of node in graph structure |

**Use Cases:**
- **Hub detection**: Transactions with very high degree centrality are hubs - often fraud indicators
- **Isolation detection**: Transactions with very low degree centrality may be legitimate isolated transactions
- **Network analysis**: Identifies key nodes in fraud networks
- **Anomaly detection**: Unusual centrality patterns compared to normal transactions

---

## 2. Temporal Features

### 2.1 TemporalFeatureExtractor Features

**Extractor**: `TemporalFeatureExtractor`  
**Type**: Temporal  
**Source**: TransactionDT (transaction datetime)

#### Features Extracted:

| Feature | Description | Use Case |
|---------|-------------|----------|
| `hour_sin` | Sine encoding of hour of day (cyclical) | Captures time-of-day patterns - fraud often occurs at specific hours |
| `hour_cos` | Cosine encoding of hour of day (cyclical) | Complements sine for complete cyclical representation |
| `day_sin` | Sine encoding of day of week (cyclical) | Captures day-of-week patterns - weekend fraud patterns differ |
| `day_cos` | Cosine encoding of day of week (cyclical) | Complements sine for complete cyclical representation |
| `time_diff` | Time difference since last transaction (in seconds) | Rapid successive transactions may indicate fraud |
| `time_diff_log` | Log-transformed time difference | Normalizes time differences, captures both rapid and slow patterns |

**Use Cases:**
- **Velocity detection**: Rapid transactions (`time_diff` < threshold) indicate potential fraud
- **Time-based patterns**: Fraud often occurs during off-hours or specific days
- **Behavioral analysis**: Normal users have predictable temporal patterns
- **Anomaly detection**: Transactions at unusual times are suspicious

**Why Cyclical Encoding?**
- Standard encoding (0-23 for hours) treats hour 23 and hour 0 as far apart
- Cyclical encoding (sin/cos) treats them as adjacent, preserving temporal relationships

---

## 3. Statistical Features

### 3.1 StatisticalFeatureExtractor Features

**Extractor**: `StatisticalFeatureExtractor`  
**Type**: Statistical  
**Source**: All numerical columns in transaction data

#### Features Extracted:

For each numerical column (up to 50 columns), the following statistics are computed:

| Feature | Description | Use Case |
|---------|-------------|----------|
| `stat_mean_{i}` | Mean value across all transactions for column `i` | Baseline distribution characteristics |
| `stat_std_{i}` | Standard deviation for column `i` | Variability measure - high std indicates diverse values |
| `stat_min_{i}` | Minimum value for column `i` | Lower bound of distribution |
| `stat_max_{i}` | Maximum value for column `i` | Upper bound of distribution |
| `stat_p25_{i}` | 25th percentile for column `i` | Lower quartile |
| `stat_p50_{i}` | 50th percentile (median) for column `i` | Central tendency |
| `stat_p75_{i}` | 75th percentile for column `i` | Upper quartile |
| `stat_skew_{i}` | Skewness of distribution for column `i` | Asymmetry - positive skew = right tail, negative = left tail |
| `stat_kurt_{i}` | Kurtosis of distribution for column `i` | Tail heaviness - high kurtosis = heavy tails (outliers) |

**Use Cases:**
- **Distribution analysis**: Understanding normal vs. abnormal value distributions
- **Outlier detection**: High kurtosis indicates presence of outliers
- **Anomaly scoring**: Transactions far from distribution center are suspicious
- **Feature normalization**: Statistical measures help normalize features

**Columns Processed:**
- TransactionAmt, C1-C14, D1-D15, V1-V339, dist1, dist2, etc.

---

## 4. Risk-Based Features

### 4.1 RiskScoreExtractor Features

**Extractor**: `RiskScoreExtractor`  
**Type**: Risk-based  
**Source**: Transaction data columns

#### Features Extracted:

| Feature | Description | Use Case |
|---------|-------------|----------|
| `amt_log` | Log-transformed transaction amount | Normalizes amount distribution, reduces impact of outliers |
| `amt_zscore` | Z-score of transaction amount (standardized) | Identifies unusually large or small amounts (potential fraud) |
| `card_velocity` | Frequency of card usage (log-transformed) | High velocity = card used frequently, may indicate fraud |
| `addr_freq` | Frequency of address usage (log-transformed) | Multiple transactions from same address may indicate fraud |
| `email_freq` | Frequency of email domain usage (log-transformed) | Suspicious email domains used frequently |
| `product_freq` | Frequency of product code usage (log-transformed) | Unusual product code patterns |

**Use Cases:**
- **Amount anomalies**: `amt_zscore` flags transactions with unusual amounts
- **Card velocity**: Rapid card usage (`card_velocity` high) indicates potential card testing
- **Entity frequency**: Entities (addresses, emails) used frequently may be compromised
- **Pattern detection**: Frequent use of specific products/emails may indicate fraud schemes

---

### 4.2 PatternMatchingExtractor Features

**Extractor**: `PatternMatchingExtractor`  
**Type**: Risk-based  
**Source**: Transaction data with temporal and entity information

#### Features Extracted:

| Feature | Description | Use Case |
|---------|-------------|----------|
| `rapid_transactions` | Binary flag: 1 if transaction occurs within 1 hour of previous transaction with same card | Detects card testing attacks - rapid successive transactions |
| `round_amounts` | Binary flag: 1 if transaction amount is a round number (divisible by 100) | Round amounts often used in testing/fraud |
| `small_amounts` | Binary flag: 1 if transaction amount < $1 | Small amounts used for testing before large fraud |
| `addr_card_diversity` | Number of unique cards per address (log-transformed) | High diversity = multiple cards from same address (suspicious) |
| `device_sharing` | Number of unique cards per device (log-transformed) | Multiple cards on same device may indicate fraud |

**Use Cases:**
- **Card testing detection**: `rapid_transactions` flags rapid-fire testing
- **Testing patterns**: `round_amounts` and `small_amounts` detect testing behavior
- **Account takeover**: `addr_card_diversity` detects compromised addresses
- **Device fraud**: `device_sharing` detects device-based fraud schemes

**Pattern Examples:**
- **Card testing**: Multiple small/round amounts in rapid succession
- **Account takeover**: Multiple cards from same address
- **Device fraud**: Multiple cards using same device

---

## 5. Cross-Feature Interactions

### 5.1 CrossFeatureExtractor Features

**Extractor**: `CrossFeatureExtractor`  
**Type**: Statistical  
**Source**: Multiple transaction columns combined

#### Features Extracted:

| Feature | Description | Use Case |
|---------|-------------|----------|
| `amt_product_interaction` | Transaction amount × Product code (log-transformed) | Captures product-specific amount patterns - some products have typical amounts |
| `amt_card_interaction` | Transaction amount × Card type (log-transformed) | Different card types have different typical amounts |
| `dist_addr_interaction` | Distance × Address (log-transformed) | Geographic patterns - distance from address may indicate fraud |

**Use Cases:**
- **Product-specific fraud**: Certain products may have fraud patterns at specific amounts
- **Card type analysis**: Different card types (credit/debit) have different fraud patterns
- **Geographic anomalies**: Large distances from registered address are suspicious
- **Feature interactions**: Captures non-linear relationships between features

**Why Interactions Matter:**
- Individual features may not show fraud patterns
- Combinations reveal complex fraud schemes
- Example: Small amount + specific product + rapid timing = testing pattern

---

## Feature Summary by Category

### Graph Features (Neighbor Aggregations)
- **Count**: ~6 aggregations × number of edge types × feature dimensions
- **Purpose**: Capture local graph structure and entity relationships
- **Key Insight**: Fraud often involves multiple entities (cards, addresses, devices)

### Graph Features (Centrality)
- **Count**: 2 features (degree, normalized degree)
- **Purpose**: Identify important nodes in fraud networks
- **Key Insight**: Fraud rings create highly connected subgraphs

### Temporal Features
- **Count**: 6 features
- **Purpose**: Capture time-based fraud patterns
- **Key Insight**: Fraud has temporal signatures (off-hours, rapid transactions)

### Statistical Features
- **Count**: 9 statistics × 50 columns = 450 features
- **Purpose**: Understand distribution characteristics
- **Key Insight**: Fraud creates distribution anomalies

### Risk Features (Basic)
- **Count**: 6 features
- **Purpose**: Entity frequency and amount anomalies
- **Key Insight**: Frequent entity usage and unusual amounts indicate fraud

### Risk Features (Pattern Matching)
- **Count**: 5 features
- **Purpose**: Detect specific fraud patterns
- **Key Insight**: Fraud has recognizable patterns (testing, sharing, etc.)

### Cross-Feature Interactions
- **Count**: 3 features
- **Purpose**: Capture feature interactions
- **Key Insight**: Fraud patterns emerge from feature combinations

---

## Total Feature Count

**Approximate Total**: 
- Graph neighbor aggregations: Variable (depends on graph structure)
- Graph centrality: 2
- Temporal: 6
- Statistical: ~450 (9 × 50 columns)
- Risk (basic): 6
- Risk (pattern): 5
- Cross-features: 3

**Total**: ~470+ features (varies with graph structure)

---

## Feature Usage in Fraud Detection

### 1. **Graph Features → Network Analysis**
- Detect fraud rings and organized fraud
- Identify compromised entities (cards, addresses)
- Track entity relationships

### 2. **Temporal Features → Behavioral Analysis**
- Detect velocity-based fraud
- Identify time-based anomalies
- Pattern recognition in transaction timing

### 3. **Statistical Features → Distribution Analysis**
- Identify outliers and anomalies
- Understand normal vs. abnormal distributions
- Feature normalization

### 4. **Risk Features → Pattern Detection**
- Entity frequency analysis
- Amount anomaly detection
- Specific fraud pattern recognition

### 5. **Cross-Features → Complex Pattern Detection**
- Multi-dimensional fraud patterns
- Feature interaction analysis
- Non-linear relationship capture

---

## Feature Engineering Workflow

1. **Graph Construction**: Build heterogeneous graph from transaction data
2. **Neighbor Aggregation**: For each transaction, aggregate neighbor features
3. **Centrality Computation**: Calculate graph centrality measures
4. **Temporal Extraction**: Extract time-based features
5. **Statistical Analysis**: Compute distribution statistics
6. **Risk Scoring**: Calculate risk indicators
7. **Pattern Matching**: Detect known fraud patterns
8. **Cross-Feature Creation**: Generate feature interactions
9. **Feature Combination**: Concatenate all features
10. **Normalization**: Normalize features for model input

---

## Best Practices

### Feature Selection
- Use all features initially
- Remove low-variance features
- Use feature importance from model

### Feature Scaling
- Normalize all features before training
- Use mean/std normalization
- Handle missing values appropriately

### Feature Caching
- Cache extracted features for faster iteration
- Use feature versioning for experiments
- Store feature metadata

### Distributed Processing
- Extract features in parallel
- Use multiple workers for large datasets
- Cache intermediate results

---

## Example: How Features Detect Fraud

**Scenario**: Card testing attack

1. **Temporal Features**: `rapid_transactions = 1` (multiple transactions in short time)
2. **Risk Features**: `small_amounts = 1` (testing with small amounts)
3. **Risk Features**: `round_amounts = 1` (round number testing)
4. **Graph Features**: High `neighbor_count` (same card used multiple times)
5. **Risk Features**: High `card_velocity` (card used frequently)
6. **Pattern Features**: Combination of above flags fraud

**Result**: Model combines these features to detect card testing pattern

---

## Conclusion

The feature engineering system extracts **470+ features** across 7 categories, providing comprehensive coverage of:
- Graph structure and relationships
- Temporal patterns
- Statistical distributions
- Risk indicators
- Fraud patterns
- Feature interactions

These features enable the model to detect various fraud types including:
- Card testing
- Account takeover
- Fraud rings
- Velocity-based fraud
- Amount anomalies
- Entity compromise
- Pattern-based fraud



