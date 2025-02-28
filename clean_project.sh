#!/bin/bash
set -e

echo "Cleaning up project and Docker..."

# Remove any Docker-related temporary files in the project
echo "Removing Docker-related files in the project..."
find . -name "*.tar" -o -name "docker-*" -type f -delete 2>/dev/null || true

# Clean up any large temporary files
echo "Removing large temporary files..."
find . -path "./models" -prune -o -path "./venv" -prune -o -type f -size +100M -delete 2>/dev/null || true

# Force Docker system prune
echo "Forcing Docker system prune..."
docker system prune -a -f --volumes

# Check disk space
echo "Current disk space:"
df -h

echo "Cleanup complete!" 