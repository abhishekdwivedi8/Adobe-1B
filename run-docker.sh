#!/bin/bash

# Enhanced Document Intelligence System - Docker Runner
# Quick start script for Docker deployment

set -e

echo "=== Enhanced Document Intelligence System - Docker Runner ==="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Build the Docker image
echo "Building Docker image..."
docker build -t document-intelligence:latest .

# Create necessary directories if they don't exist
mkdir -p input output logs

# Check if input directory has PDF files
if [ -z "$(ls -A input/*.pdf 2>/dev/null)" ]; then
    echo "Warning: No PDF files found in input/ directory"
    echo "Please add PDF files to the input/ directory before running"
fi

# Run the container
echo "Starting container..."
docker run --rm \
    -v "$(pwd)/input:/app/input:ro" \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/logs:/app/logs" \
    --memory=2g \
    --cpus=2.0 \
    document-intelligence:latest

echo "=== Analysis completed. Check output/ directory for results ==="