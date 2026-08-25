# Use NVIDIA CUDA runtime image (not devel) — saves ~5 GB by dropping compilers
# and dev headers we don't need at runtime. vLLM and torch ship prebuilt wheels,
# so no compilation occurs during install.
#
# CUDA 12.8 (cuDNN9) is required for Blackwell (sm_120, e.g. RTX PRO 6000
# Blackwell Server Edition) GPU support — a torch build against CUDA <=12.6
# only ships kernels up to sm_90 and fails with "no kernel image is available
# for execution on the device" on those cards. 12.1/cuDNN8 was previously
# pinned deliberately (see below); this bump replaces the base image project-
# wide, not just for one GPU generation, so re-verify this Dockerfile's pins
# after any future torch/vllm bump.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# CHTC's docker-universe jobs run as an arbitrary sandboxed UID with no
# matching /etc/passwd entry. torch._inductor.codecache computes its cache
# dir at import time via getpass.getuser(), which falls back to
# pwd.getpwuid(os.getuid()) and crashes (KeyError: getpwuid(): uid not found)
# when that UID isn't in passwd. getpass.getuser() checks LOGNAME/USER first
# and never touches pwd if either is set, so set them unconditionally.
ENV USER=vllm
ENV LOGNAME=vllm

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

# All dependencies in one layer. vLLM is PINNED to the version validated
# against this base image (resolves torch==2.9.1+cu128, confirmed to include
# sm_120 in its compiled kernels — see the arch-flags check below). Unbounded
# `vllm>=0.6.0` previously pulled vllm 0.21 + torch 2.11+cu130, which silently
# mismatched the CUDA 12.1 base image — caught after release; see v2.0.7
# commit message. Re-pin deliberately, the same way, if vllm is bumped again.
# xformers removed — vLLM has its own attention kernels.
RUN pip install --no-cache-dir \
    "pandas>=1.3.0" \
    "numpy>=1.21.0" \
    "openai>=1.0.0" \
    "biopython>=1.79" \
    "requests>=2.25.0" \
    "tiktoken>=0.7.0" \
    "htcondor>=24.0.0" \
    "vllm==0.14.1"

# Fail the build if a future base-image/registry change silently drops
# sm_120 support instead of crashing at inference time on the actual GPU.
RUN python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); flags = torch._C._cuda_getArchFlags(); print('arch_flags', flags); assert 'sm_120' in flags, 'sm_120 missing from compiled arch flags'"

# Install skimgpt with --no-deps. All declared deps were installed above; this
# avoids the duplicate-install bug in the previous Dockerfile where the `||`
# fallback always re-ran a non-no-deps install (~4.8 GB of duplicated layers).
ARG SKIMGPT_VERSION=2.2.1
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
