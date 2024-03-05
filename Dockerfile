# Use the NVIDIA CUDA 12.1.1 base image with Ubuntu 20.04
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Set noninteractive timezone to avoid tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install software properties common and other necessary tools
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl

# Add deadsnakes PPA for Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev

# Download and install the latest version of pip for Python 3.10
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Upgrade pip and setuptools
RUN python3.10 -m pip install --upgrade pip setuptools

# Set the working directory in the container
WORKDIR /app

# Create a Python virtual environment
RUN python3.10 -m venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy the requirements file into the container
COPY ./requirements.txt /app/requirements.txt

# Install the Python dependencies
RUN pip install -r requirements.txt
COPY --chmod=777 ./Mistral-7B-OpenOrca /app/Mistral-7B-OpenOrca
COPY --chmod=777 ./_model.py /opt/venv/lib/python3.10/site-packages/guidance/models/_model.py
