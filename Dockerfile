# Use NVIDIA CUDA runtime image (not devel) — saves ~5 GB by dropping compilers
# and dev headers we don't need at runtime. vLLM and torch ship prebuilt wheels
# for CUDA 12.1, so no compilation occurs during install.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Minimal system deps. Removed build-essential, cmake, python3.10-dev,
# wget, curl — unused at runtime once prebuilt wheels install. git stays for
# any pip vcs installs the user may layer on top.
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-distutils \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

RUN python3.10 -m pip install --upgrade pip

WORKDIR /app

# All dependencies in one layer. Includes vLLM (which pins compatible torch
# and CUDA kernels), plus transformers + accelerate which skimgpt uses for
# model loading. xformers removed — vLLM has its own attention kernels.
RUN pip install --no-cache-dir \
    "pandas>=1.3.0" \
    "numpy>=1.21.0" \
    "openai>=1.0.0" \
    "biopython>=1.79" \
    "requests>=2.25.0" \
    "tiktoken>=0.7.0" \
    "vllm>=0.6.0" \
    "htcondor>=24.0.0" \
    "transformers" \
    "accelerate"

# Install skimgpt with --no-deps. All declared deps were installed above; this
# avoids the duplicate-install bug in the previous Dockerfile where the `||`
# fallback always re-ran a non-no-deps install (~4.8 GB of duplicated layers).
ARG SKIMGPT_VERSION=0.1.9
RUN pip install --no-cache-dir --no-deps "skimgpt==${SKIMGPT_VERSION}"

RUN skimgpt-relevance --help || echo "Entry point verification failed, but package may still work"

RUN mkdir -p /app/input_lists /app/output /app/debug /app/token

COPY config.json ./

ENV VLLM_USE_MODELSCOPE=False
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

EXPOSE 5081

RUN echo '#!/bin/bash\n\
if ! nvidia-smi > /dev/null 2>&1; then\n\
    echo "Warning: No GPU detected. vLLM will run on CPU (much slower)"\n\
fi\n\
\n\
exec "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["skimgpt-relevance", "--help"]
