#!/bin/bash
# Script to run the API server

# Set environment variables
export FEATURE_STORE_BACKEND=${FEATURE_STORE_BACKEND:-csv}
export MODEL_PATH=${MODEL_PATH:-./model/model.pth}
export METADATA_PATH=${METADATA_PATH:-./model/metadata.pkl}
export DEVICE=${DEVICE:-cpu}

# Run API server
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload














