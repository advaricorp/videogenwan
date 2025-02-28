#!/bin/bash
set -e

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "docker-compose not found. Installing..."
    pip install docker-compose
fi

# Check if NVIDIA Container Toolkit is installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers not found. Please install NVIDIA drivers and NVIDIA Container Toolkit."
    exit 1
fi

echo "Creating necessary directories..."
mkdir -p models output

echo "Deploying video generation service..."
docker-compose up -d

echo "Checking service status..."
sleep 5
docker-compose ps

echo "Viewing logs..."
docker-compose logs -f