# Quick Start Guide

## Prerequisites

- Python 3.9+
- Docker and Docker Compose (optional)
- Redis (optional, for Redis backend)

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Prepare data**:
   - Place transaction data in `./ieee-data/train_transaction.csv`
   - Place identity data in `./ieee-data/train_identity.csv`
   - Ensure features are in `./data/features.csv`

## Quick Start

### Option 1: Local Development

1. **Build graph and extract features**:
```bash
python main.py --mode build
```

2. **Train model**:
```bash
python main.py --mode train
```

3. **Start API server**:
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Option 2: Docker Deployment

1. **Start all services**:
```bash
docker-compose up -d
```

2. **Check API health**:
```bash
curl http://localhost:8000/health
```

3. **Make predictions**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"node_ids": ["2987000", "2987001"]}'
```

## Configuration

Edit `config.yaml` to configure:
- Model hyperparameters
- Training settings
- Storage backend
- Data paths

## API Usage

### Health Check
```bash
GET /health
```

### Predict Fraud
```bash
POST /predict
{
  "node_ids": ["2987000", "2987001"]
}
```

### Get Features
```bash
GET /features?node_ids=2987000&node_ids=2987001
```

## Troubleshooting

- **Redis connection issues**: Ensure Redis is running (`docker-compose up -d redis`)
- **Model not found**: Run training first (`python main.py --mode train`)
- **Out of memory**: Reduce `world_size` in `config.yaml`

## Next Steps

- See `README.md` for detailed documentation
- Check `config.yaml` for configuration options
- Review API documentation at `http://localhost:8000/docs` (when API is running)














