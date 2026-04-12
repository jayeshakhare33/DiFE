# Dockerfile for Fraud Detection GNN System
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/data/features /app/model /app/output /app/logs /app/scripts

# Expose API port and distributed training port
EXPOSE 8000 12355

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FEATURE_STORE_BACKEND=parquet
ENV MODEL_PATH=/app/model/model.pth
ENV METADATA_PATH=/app/model/metadata.pkl
ENV DEVICE=cpu
ENV STORAGE_BASE_DIR=/app/data/features

# Run API server (default command, can be overridden)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

