#!/bin/bash
# Script to run distributed training

# Set environment variables
export WORLD_SIZE=${WORLD_SIZE:-4}
export BACKEND=${BACKEND:-gloo}
export DEVICE=${DEVICE:-cpu}

# Run training
python main.py --mode train











