# Feature Storage Options Analysis

## Current State
- **Node Features**: 20 nodes × 50 features = 1,000 feature values
- **Edge Features**: 156 edges × 12 features = 1,872 feature values
- **Current Format**: CSV files
- **Storage Location**: `./data/features/`

## Available Storage Backends

Your system already supports three storage backends:

### 1. **CSV Backend** (Current)
**Pros:**
- ✅ Human-readable and easy to inspect
- ✅ No additional dependencies
- ✅ Works with any text editor/spreadsheet
- ✅ Easy to version control (git-friendly)
- ✅ Simple debugging

**Cons:**
- ❌ Slow read/write for large datasets
- ❌ No compression (larger file sizes)
- ❌ No columnar optimization
- ❌ Poor performance with millions of rows
- ❌ Type information can be lost

**Best For:**
- Small datasets (< 100K rows)
- Development and debugging
- One-time analysis
- When human readability is critical

**File Size Estimate:**
- Node features: ~50-100 KB
- Edge features: ~50-100 KB

---

### 2. **Parquet Backend** ⭐ **RECOMMENDED**
**Pros:**
- ✅ **10-100x faster** read/write than CSV
- ✅ **50-90% smaller** file sizes (columnar compression)
- ✅ Preserves data types (no type inference needed)
- ✅ Columnar format (read only needed columns)
- ✅ Industry standard for data science
- ✅ Works seamlessly with pandas, Dask, Spark
- ✅ Supports schema evolution
- ✅ Excellent for GNN training (batch loading)

**Cons:**
- ❌ Not human-readable (binary format)
- ❌ Requires `pyarrow` or `fastparquet` dependency
- ❌ Slightly more complex than CSV

**Best For:**
- **Production environments** ⭐
- Large datasets (100K+ rows)
- Frequent read/write operations
- Training pipelines
- When performance matters

**File Size Estimate:**
- Node features: ~10-20 KB (compressed)
- Edge features: ~10-20 KB (compressed)

**Performance:**
- Read: ~10-50ms (vs 100-500ms for CSV)
- Write: ~20-100ms (vs 200-1000ms for CSV)

---

### 3. **Redis Backend**
**Pros:**
- ✅ **Extremely fast** in-memory access (microseconds)
- ✅ Perfect for real-time inference
- ✅ Supports concurrent access
- ✅ Can serve multiple processes
- ✅ Built-in expiration/TTL support
- ✅ Good for hot data caching

**Cons:**
- ❌ **Volatile** (data lost on restart unless persisted)
- ❌ Limited by RAM size
- ❌ Requires Redis server running
- ❌ More complex setup
- ❌ Not ideal for large datasets (>GB)
- ❌ Network latency for remote access

**Best For:**
- Real-time inference API
- Hot data caching
- When sub-millisecond access is needed
- Small to medium datasets that fit in memory

**Memory Estimate:**
- Node features: ~50-100 KB
- Edge features: ~50-100 KB
- Total: ~100-200 KB (very small)

---

## Recommendations by Use Case

### 🎯 **For GNN Training (Current Priority)**
**Recommended: Parquet**

**Why:**
1. Training requires loading full feature sets multiple times
2. Parquet is 10-100x faster than CSV for bulk reads
3. Smaller files = faster I/O = faster training iterations
4. Columnar format allows selective column loading
5. Standard format for ML pipelines

**Migration:**
```python
# In your code, change:
feature_store = FeatureStore(backend_type='parquet', base_dir='./data/features')
```

---

### 🚀 **For Production Inference**
**Recommended: Hybrid Approach (Parquet + Redis)**

**Strategy:**
1. **Parquet** for persistent storage and batch loading
2. **Redis** for hot data caching during inference

**Why:**
- Parquet: Reliable, fast bulk loading
- Redis: Sub-millisecond access for real-time predictions
- Load features into Redis at startup, fallback to Parquet if cache miss

---

### 🔬 **For Development/Debugging**
**Recommended: CSV (Current)**

**Why:**
- Easy to inspect with text editors
- Quick to modify for testing
- No dependencies
- Good for small datasets

---

## Performance Comparison

| Metric | CSV | Parquet | Redis |
|--------|-----|---------|-------|
| **Read Speed** | 1x (baseline) | 10-100x faster | 1000x faster |
| **Write Speed** | 1x (baseline) | 10-50x faster | 1000x faster |
| **File Size** | 1x (baseline) | 50-90% smaller | N/A (memory) |
| **Human Readable** | ✅ Yes | ❌ No | ❌ No |
| **Type Safety** | ⚠️ Partial | ✅ Yes | ✅ Yes |
| **Scalability** | ⚠️ Poor (>1M rows) | ✅ Excellent | ⚠️ Limited by RAM |
| **Setup Complexity** | ✅ Simple | ✅ Simple | ⚠️ Medium |

---

## Migration Guide

### Option 1: Switch to Parquet (Recommended)

**Step 1: Update config.yaml**
```yaml
storage:
  backend: parquet  # Change from 'csv' to 'parquet'
  base_dir: ./data/features
```

**Step 2: Convert existing CSV files**
```python
from storage import FeatureStore
import pandas as pd

# Load existing CSV features
csv_store = FeatureStore(backend_type='csv', base_dir='./data/features')
node_features = csv_store.load_features('node_features')
edge_features = csv_store.load_features('edge_features')

# Save to Parquet
parquet_store = FeatureStore(backend_type='parquet', base_dir='./data/features')
parquet_store.save_features(node_features, 'node_features')
parquet_store.save_features(edge_features, 'edge_features')
```

**Step 3: Verify**
```python
# Test loading
features = parquet_store.load_features('node_features')
print(f"Loaded {len(features)} nodes with {len(features.columns)} features")
```

---

### Option 2: Use Redis for Inference (Optional)

**Step 1: Start Redis**
```bash
# Using Docker
docker-compose up -d redis

# Or install locally
# Windows: Download from redis.io
# Linux: sudo apt-get install redis-server
```

**Step 2: Load features into Redis at startup**
```python
from storage import FeatureStore

# Load from Parquet (persistent)
parquet_store = FeatureStore(backend_type='parquet')
node_features = parquet_store.load_features('node_features')

# Cache in Redis (fast access)
redis_store = FeatureStore(
    backend_type='redis',
    host='localhost',
    port=6379
)
redis_store.save_features(node_features, 'node_features')
```

---

## Best Practice: Hybrid Approach

For a production GNN system, use a **two-tier storage strategy**:

```
┌─────────────────────────────────────┐
│   Feature Generation Pipeline       │
│   (Extract → Transform → Store)     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Tier 1: Parquet (Persistent)      │
│   - Long-term storage               │
│   - Backup and versioning           │
│   - Batch training                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Tier 2: Redis (Cache)             │
│   - Hot data for inference          │
│   - Fast lookups                    │
│   - Auto-refresh from Parquet       │
└─────────────────────────────────────┘
```

**Implementation:**
```python
class HybridFeatureStore:
    def __init__(self):
        self.parquet_store = FeatureStore(backend_type='parquet')
        self.redis_store = FeatureStore(backend_type='redis')
    
    def get_features(self, key: str):
        # Try Redis first (fast)
        try:
            return self.redis_store.load_features(key)
        except:
            # Fallback to Parquet (reliable)
            features = self.parquet_store.load_features(key)
            # Cache in Redis for next time
            self.redis_store.save_features(features, key)
            return features
```

---

## Immediate Action Items

1. **✅ Switch to Parquet** (5 minutes)
   - Update `config.yaml`: `backend: parquet`
   - Convert existing CSV files (use migration script above)
   - Test loading features

2. **⏳ Consider Redis** (if doing real-time inference)
   - Set up Redis server
   - Implement caching layer
   - Load features at API startup

3. **📊 Monitor Performance**
   - Measure load times
   - Track file sizes
   - Optimize based on usage patterns

---

## Summary

**For your current needs (GNN training):**
- **Best Choice: Parquet** ⭐
- Fast, efficient, scalable
- Industry standard
- Easy migration from CSV

**For future (real-time inference):**
- **Add Redis** as a caching layer
- Keep Parquet as persistent storage
- Implement hybrid approach

**Keep CSV for:**
- Development and debugging
- Small test datasets
- Human inspection

---

## Questions to Consider

1. **Dataset Size**: Will you scale to millions of nodes/edges?
   - → Parquet is essential

2. **Access Pattern**: Bulk loading or random access?
   - Bulk → Parquet
   - Random → Redis

3. **Real-time Requirements**: Need sub-millisecond access?
   - → Redis caching

4. **Infrastructure**: Can you run Redis?
   - → If yes, use hybrid approach
   - → If no, Parquet alone is excellent

