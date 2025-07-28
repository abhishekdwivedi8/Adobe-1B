# Enhanced Document Intelligence System Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for CPU-only execution
ENV TORCH_DISABLE_CUDA=1
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV PYTHONPATH=/app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with CPU-only versions
RUN pip install --no-cache-dir -r requirements.txt

# Copy cached models to container (no internet download needed)
COPY models/ /app/models/

# Copy core system files
COPY core/ /app/core/
COPY run_enhanced_analysis.py /app/
COPY main.py /app/

# Copy configuration files
COPY config.py /app/
COPY config_optimized.py /app/
COPY lightweight_model_manager.py /app/

# Copy any additional system files that might be needed
COPY *.py /app/ 2>/dev/null || true

# Create necessary directories
RUN mkdir -p /app/input /app/output /app/logs

# Copy sample input files if they exist
COPY input/ /app/input/ 2>/dev/null || true

# Set permissions
RUN chmod +x main.py

# Expose port (if needed for API)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app'); from run_enhanced_analysis import main; print('OK')" || exit 1

# Default command - single entry point
CMD ["python", "main.py"]

# Labels for metadata
LABEL maintainer="Document Intelligence Team"
LABEL version="2.0"
LABEL description="Enhanced Document Intelligence System - Single Entry Point"
LABEL features="90%+ accuracy, title reconstruction, enhanced ranking"