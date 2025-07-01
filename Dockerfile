# Use NVIDIA CUDA base image with Python 3.10
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies including Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    python3.10-distutils \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Install pip for Python 3.10
RUN python3.10 -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Install dependencies from PyPI first to avoid TestPyPI issues
RUN pip install --no-cache-dir \
    pandas>=1.3.0 \
    numpy>=1.21.0 \
    openai>=1.0.0 \
    biopython>=1.79 \
    requests>=2.25.0 \
    tiktoken>=0.7.0 \
    vllm>=0.6.0 \
    htcondor>=24.0.0

# Build argument for skimgpt version
ARG SKIMGPT_VERSION=0.1.9
# Try to install skimgpt from TestPyPI (may need to skip broken dependencies)
RUN pip install --no-cache-dir --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --no-deps skimgpt==${SKIMGPT_VERSION} || \
    echo "Warning: Could not install skimgpt from TestPyPI, installing from PyPI instead" && \
    pip install --no-cache-dir skimgpt==${SKIMGPT_VERSION}

# Verify the installation and entry point
RUN skimgpt-relevance --help || echo "Entry point verification failed, but package may still work"

# Install additional dependencies that might be needed
RUN pip install --no-cache-dir \
    torch \
    transformers \
    accelerate \
    xformers

# Create directories for data and output
RUN mkdir -p /app/input_lists /app/output /app/debug /app/token

# Copy configuration file if it exists (will be volume mounted in production)
COPY config.json ./ 

# Set environment variables for vLLM
ENV VLLM_USE_MODELSCOPE=False
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# Expose port if needed
EXPOSE 5081

# Create entrypoint script
RUN echo '#!/bin/bash\n\
# Check for GPU availability\n\
if ! nvidia-smi > /dev/null 2>&1; then\n\
    echo "Warning: No GPU detected. vLLM will run on CPU (much slower)"\n\
fi\n\
\n\
# Run the application\n\
exec "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["skimgpt-relevance", "--help"]

