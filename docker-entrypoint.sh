#!/bin/bash
set -e

# Download models if requested and they don't exist
if [ "$DOWNLOAD_MODELS" = "true" ] && [ ! -d "/app/models/Wan2.1-T2V-1.3B" ] || [ -z "$(ls -A /app/models/Wan2.1-T2V-1.3B)" ]; then
    echo "Models not found. Running setup..."
    python test_wan_video.py setup
fi

# Run test script if requested
if [ "$1" = "test" ]; then
    echo "Running test script..."
    python test_wan_video.py
    exit $?
fi

# Execute the command passed to docker
exec "$@" 
