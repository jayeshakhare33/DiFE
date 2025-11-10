# Distributed AI System for Fraud Detection using Graph Neural Networks

A scalable, distributed system for fraud detection using Graph Neural Networks (GNNs) with DGL and PyTorch Distributed.

## Features

- **Distributed Graph Construction**: Parallel graph building from transaction data
- **Graph-based Feature Engineering**: Extracts degree, centrality, clustering, and neighbor features
- **Distributed GNN Training**: Multi-process training using PyTorch Distributed and DGL
- **Feature Store**: Supports CSV, Redis, and Parquet backends
- **RESTful API**: FastAPI endpoints for inference and feature retrieval
- **Docker Deployment**: Containerized setup with docker-compose
- **Scalability**: Designed for large transaction graphs with fault tolerance

## Architecture

The system is organized into modular packages:

```
.
├── graph_processing/      # Graph construction and partitioning
├── feature_engineering/   # Feature extraction and embeddings
├── gnn_training/         # Distributed GNN training
├── api/                  # FastAPI endpoints
├── storage/              # Feature store backends
├── config.yaml           # Configuration file
├── main.py              # Main orchestration script
├── Dockerfile           # Docker image definition
└── docker-compose.yml   # Multi-container setup
```

## Installation

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- Redis (optional, for Redis backend)

### Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd GNN-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare data:
   - Place transaction data in `./ieee-data/train_transaction.csv`
   - Place identity data in `./ieee-data/train_identity.csv`
   - Ensure features are in `./data/features.csv`

## Usage

### Configuration

Edit `config.yaml` to configure:
- Model hyperparameters
- Training settings
- Distributed training parameters
- Storage backend
- Data paths

### Running Locally

1. **Build graph and extract features**:
```bash
python main.py --mode build
```

2. **Train model**:
```bash
python main.py --mode train
```

3. **Run full pipeline**:
```bash
python main.py --mode all
```

4. **Start API server**:
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Docker Deployment

1. **Build and start services**:
```bash
docker-compose up -d
```

2. **Start training (optional)**:
```bash
docker-compose --profile training up trainer
```

3. **View logs**:
```bash
docker-compose logs -f api
```

4. **Stop services**:
```bash
docker-compose down
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Predict Fraud
```bash
POST /predict
Content-Type: application/json

{
  "node_ids": ["2987000", "2987001", "2987002"]
}
```

### Get Features
```bash
GET /features?node_ids=2987000&node_ids=2987001
```

or

```bash
POST /features
Content-Type: application/json

{
  "node_ids": ["2987000", "2987001"],
  "feature_types": ["degree", "centrality"]
}
```

## Graph Structure

The system converts transaction data into a heterogeneous graph:

- **Nodes**: 
  - `target` nodes: Transactions (users)
  - Entity nodes: Cards, addresses, devices, etc.

- **Edges**: 
  - Transaction-to-entity relationships
  - Self-loops for transactions

## Feature Engineering

The system extracts various graph-based features:

- **Degree Features**: In-degree, out-degree, total degree per edge type
- **Centrality Features**: Degree centrality, betweenness, closeness, PageRank
- **Clustering Features**: Clustering coefficient
- **Neighbor Features**: Number of unique neighbors, average neighbor degree
- **Embeddings**: Node embeddings from trained GNN model

## Storage Backends

The feature store supports multiple backends:

- **CSV**: Simple file-based storage (default)
- **Redis**: In-memory storage for fast access
- **Parquet**: Efficient columnar storage

Configure in `config.yaml`:
```yaml
storage:
  backend: redis  # or 'csv' or 'parquet'
  redis:
    host: localhost
    port: 6379
```

## Distributed Training

The system supports distributed training using PyTorch Distributed:

1. Configure in `config.yaml`:
```yaml
distributed:
  world_size: 4
  backend: gloo  # or 'nccl' for GPU
```

2. Run distributed training:
```bash
python -m gnn_training.distributed_trainer
```

## Monitoring and Logging

- Logs are written to `./logs/` directory
- API logs are available via Docker Compose: `docker-compose logs -f api`
- Training metrics are saved to `./output/results.txt`

## Performance Considerations

- **Graph Partitioning**: Large graphs are partitioned for distributed processing
- **Feature Caching**: Features are cached in the feature store
- **Batch Processing**: Transactions are processed in batches
- **Fault Tolerance**: System handles failures gracefully with retries

## Troubleshooting

### Redis Connection Issues
- Ensure Redis is running: `docker-compose up -d redis`
- Check Redis connection: `redis-cli ping`

### Model Not Found
- Ensure model files exist in `./model/` directory
- Run training first: `python main.py --mode train`

### Out of Memory
- Reduce `world_size` in distributed config
- Reduce `chunk_size` in graph processing config
- Use smaller batch sizes

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Contact

[Add contact information here]










