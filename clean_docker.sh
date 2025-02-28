#!/bin/bash
set -e

echo "Performing deep clean of Docker system..."

# Stop all running containers
echo "Stopping all running containers..."
docker stop $(docker ps -aq) 2>/dev/null || true

# Remove all containers
echo "Removing all containers..."
docker rm $(docker ps -aq) 2>/dev/null || true

# Remove all images
echo "Removing all Docker images..."
docker rmi $(docker images -q) --force 2>/dev/null || true

# Remove all volumes
echo "Removing all Docker volumes..."
docker volume rm $(docker volume ls -q) 2>/dev/null || true

# Remove all networks
echo "Removing all Docker networks..."
docker network rm $(docker network ls -q) 2>/dev/null || true

# Prune everything
echo "Pruning entire Docker system..."
docker system prune -a -f --volumes

# Check disk space
echo "Checking disk space..."
df -h

echo "Docker system has been completely cleaned!"
echo "You can now rebuild with a clean slate." 