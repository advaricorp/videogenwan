#!/bin/bash

echo "Checking largest directories..."
du -h --max-depth=1 | sort -hr | head -10

echo -e "\nChecking largest files..."
find . -type f -not -path "*/\.*" -not -path "*/venv/*" -exec du -h {} \; | sort -hr | head -20

echo -e "\nTotal size of current directory:"
du -sh . 