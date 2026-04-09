# Distributed Feature Engineering for Fraud Detection

This document describes the distributed feature engineering framework that enables parallel, decentralized feature extraction for improved fraud detection accuracy.

## Overview

The distributed feature engineering system allows you to:
1. **Extract features in parallel** across multiple workers/processes
2. **Use advanced feature extractors** for better fraud detection accuracy
3. **Distribute feature engineering** across multiple machines/nodes
4. **Cache and reuse** extracted features for faster iteration

## Architecture

### Components

1. **Feature Extractors** (`gnn/feature_engineering.py`)
   - Base class for all feature extractors
   - Modular, parallelizable components
   - Types: Graph-based, Temporal, Statistical, Risk-based

2. **Distributed Pipeline** (`gnn/distributed_feature_pipeline.py`)
   - Manages parallel feature extraction
   - Handles caching and result combination
   - Supports batch processing

3. **Advanced Features** (`gnn/advanced_features.py`)
   - Graph centrality features
   - Pattern matching features
   - Cross-feature interactions
   - Risk scoring features

## Usage

### 1. Standalone Feature Extraction

Extract features independently before training:

```bash
python distributed_feature_extraction.py \
    --transaction-data ./ieee-data/train_transaction.csv \
    --identity-data ./ieee-data/train_identity.csv \
    --training-dir ./data \
    --output-dir ./features \
    --n-workers 4 \
    --extractors graph_neighbor_aggregator temporal_features risk_score_features
```

### 2. Integration with Training

Enable distributed features in training:

```bash
python train.py \
    --use-distributed-features \
    --feature-dir ./features \
    --enhance-features \
    --n-feature-workers 4
```

### 3. Programmatic Usage

```python
from gnn.distributed_feature_pipeline import DistributedFeaturePipeline
from gnn.feature_engineering import initialize_default_registry
from gnn.advanced_features import GraphCentralityExtractor, PatternMatchingExtractor
import pandas as pd

# Load data
transaction_df = pd.read_csv('train_transaction.csv')

# Initialize registry with advanced extractors
registry = initialize_default_registry()
registry.register(GraphCentralityExtractor())
registry.register(PatternMatchingExtractor())

# Create pipeline
pipeline = DistributedFeaturePipeline(
    registry=registry,
    n_workers=4,
    output_dir='./features'
)

# Extract features
features = pipeline.process_transaction_data(
    transaction_df=transaction_df,
    graph_data=graph,  # Optional HeteroData object
    use_cache=True
)
```

## Available Feature Extractors

### 1. GraphNeighborAggregator
- Aggregates neighbor features (mean, std, max, min, sum, count)
- Captures local graph structure
- **Type**: Graph-based

### 2. TemporalFeatureExtractor
- Time-based features (hour, day, time differences)
- Cyclical encoding for temporal patterns
- **Type**: Temporal

### 3. StatisticalFeatureExtractor
- Statistical measures (mean, std, percentiles, skewness, kurtosis)
- Distribution characteristics
- **Type**: Statistical

### 4. RiskScoreExtractor
- Risk scoring based on patterns
- Transaction amount anomalies
- Entity frequency analysis
- **Type**: Risk-based

### 5. GraphCentralityExtractor (Advanced)
- Graph centrality measures
- Degree centrality
- Node importance metrics
- **Type**: Graph-based

### 6. PatternMatchingExtractor (Advanced)
- Rapid transaction detection
- Unusual amount patterns
- Device/card sharing patterns
- **Type**: Risk-based

### 7. CrossFeatureExtractor (Advanced)
- Feature interactions
- Amount × Product interactions
- Cross-feature combinations
- **Type**: Statistical

## Distributed Processing

### Local Multiprocessing

The system uses Python's `multiprocessing` for local parallelization:

```python
pipeline = DistributedFeaturePipeline(n_workers=8)  # Use 8 CPU cores
```

### Distributed Computing

For distributed computing across multiple machines, you can:

1. **Use a job queue** (e.g., Celery, RQ):
   ```python
   # Each worker processes a subset of extractors
   # Results are combined at the end
   ```

2. **Use Dask or Ray**:
   ```python
   import dask
   # Distribute feature extraction across cluster
   ```

3. **Manual distribution**:
   - Split data by time windows or entity groups
   - Process each split independently
   - Combine results

## Performance Optimization

### Caching

Features are automatically cached to avoid recomputation:

```python
# First run: extracts and caches
features = pipeline.process_transaction_data(df, use_cache=True)

# Subsequent runs: loads from cache
features = pipeline.process_transaction_data(df, use_cache=True)
```

### Batch Processing

Process multiple batches in parallel:

```python
batches = [batch1, batch2, batch3, ...]
results = pipeline.process_batch(batches)
```

### Selective Extraction

Use only specific extractors:

```python
features = pipeline.process_transaction_data(
    df,
    extractor_names=['temporal_features', 'risk_score_features']
)
```

## Feature Engineering Workflow

### Recommended Workflow

1. **Pre-extract features** (can be done on separate machines):
   ```bash
   python distributed_feature_extraction.py --transaction-data ... --output-dir ./features
   ```

2. **Train with enhanced features**:
   ```bash
   python train.py --use-distributed-features --feature-dir ./features
   ```

3. **Iterate and improve**:
   - Add new extractors
   - Tune existing extractors
   - Re-extract and retrain

## Adding Custom Extractors

Create a new extractor:

```python
from gnn.feature_engineering import FeatureExtractor

class MyCustomExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__("my_custom_extractor", "custom_type")
    
    def extract(self, data: Dict[str, Any], **kwargs) -> np.ndarray:
        # Your feature extraction logic
        return features
    
    def get_feature_names(self) -> List[str]:
        return ['feature1', 'feature2', ...]

# Register it
registry.register(MyCustomExtractor())
```

## Accuracy Improvements

The distributed feature engineering system improves accuracy by:

1. **More comprehensive features**: Graph-based, temporal, and statistical features
2. **Better pattern detection**: Advanced pattern matching extractors
3. **Feature interactions**: Cross-feature extractors capture complex relationships
4. **Scalability**: Process larger datasets with parallelization

## Troubleshooting

### Memory Issues
- Reduce `n_workers` if running out of memory
- Process data in batches
- Use feature selection to reduce dimensionality

### Slow Performance
- Enable caching (`use_cache=True`)
- Use pre-extracted features
- Reduce number of extractors
- Optimize extractor implementations

### Feature Shape Mismatches
- Ensure all extractors return features with same number of rows
- Check that data alignment is correct
- Use `target_length` parameter in `combine_features`

## Examples

See `distributed_feature_extraction.py` for a complete example of standalone feature extraction.

See `train.py` for integration with the training pipeline.

