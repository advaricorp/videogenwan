#!/bin/bash
set -e

echo "Attempting to recover Docker build space..."

# Stop Docker service
echo "Stopping Docker service..."
sudo service docker stop || true

# Remove Docker data directory (WARNING: This removes ALL Docker data)
echo "Removing Docker data directory..."
sudo rm -rf /var/lib/docker || true

# Restart Docker service
echo "Restarting Docker service..."
sudo service docker start || true

# Check disk space
echo "Current disk space:"
df -h

echo "Docker space recovery complete!"
echo "Note: All Docker images, containers, and volumes have been removed." 