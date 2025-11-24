#!/bin/bash
# Deployment script for multi-container setup

set -e

echo "=========================================="
echo "Fraud Detection GNN - Multi-Container Deployment"
echo "=========================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if features exist
if [ ! -d "./data/features" ] || [ ! -f "./data/features/node_features.parquet" ]; then
    echo "⚠️  Warning: Features not found. Generating features first..."
    echo "   Run: python main.py --mode build"
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build Docker images
echo ""
echo "📦 Building Docker images..."
docker-compose build

# Start Redis first
echo ""
echo "🔴 Starting Redis..."
docker-compose up -d redis

# Wait for Redis to be healthy
echo "⏳ Waiting for Redis to be ready..."
timeout=30
while [ $timeout -gt 0 ]; do
    if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis is ready!"
        break
    fi
    sleep 1
    timeout=$((timeout - 1))
done

if [ $timeout -eq 0 ]; then
    echo "❌ Redis failed to start"
    exit 1
fi

# Load features into Redis
echo ""
echo "📥 Loading features into Redis..."
docker-compose up feature-loader

# Check if feature loading was successful
if [ $? -eq 0 ]; then
    echo "✅ Features loaded successfully!"
else
    echo "❌ Feature loading failed"
    exit 1
fi

# Start API services
echo ""
echo "🚀 Starting API services..."
docker-compose up -d api

# Wait for API to be ready
echo "⏳ Waiting for API to be ready..."
sleep 5

# Check API health
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is ready!"
else
    echo "⚠️  API may not be fully ready yet"
fi

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "Services:"
echo "  - Redis:      localhost:6379"
echo "  - API:         http://localhost:8000"
echo ""
echo "Useful commands:"
echo "  - View logs:    docker-compose logs -f"
echo "  - Stop all:     docker-compose down"
echo "  - Start training: docker-compose --profile training up"
echo ""

