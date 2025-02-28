#!/bin/bash
set -e

echo "Building Docker image with legacy builder..."
# Use DOCKER_BUILDKIT=0 to force using the legacy builder
export DOCKER_BUILDKIT=0
docker build -t videogen:test .

echo "Running test container..."
docker run --rm --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/output:/app/output" \
  videogen:test test

echo "Test completed successfully!" 