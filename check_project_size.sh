#!/bin/bash

echo "Checking project directory size..."
du -h --max-depth=1 | sort -hr

echo -e "\nChecking for large files in the project directory:"
find . -type f -size +100M | xargs du -h | sort -hr

echo -e "\nChecking for Docker-related files:"
find . -name "*.tar" -o -name "docker-*" | xargs du -h 2>/dev/null || echo "No Docker files found"

echo -e "\nChecking total disk usage:"
df -h 