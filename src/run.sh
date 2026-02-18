#!/bin/bash

# Get km_output_file from environment variable
km_output_file=files.txt
config_file="config.json"
secrets_file="secrets.json"

export TRANSFORMERS_CACHE=$_CONDOR_SCRATCH_DIR/models
export HF_HOME=$_CONDOR_SCRATCH_DIR/models
export HF_DATASETS_CACHE=$_CONDOR_SCRATCH_DIR/datasets
export HF_MODULES_CACHE=$_CONDOR_SCRATCH_DIR/modules
export HF_METRICS_CACHE=$_CONDOR_SCRATCH_DIR/metrics

# --- Fix CUDA_VISIBLE_DEVICES before Python/CUDA initialization ---
# HTCondor may set CUDA_VISIBLE_DEVICES to GPU UUIDs (e.g. "GPU-abc123...").
# CUDA inside Docker containers often can't resolve UUIDs and needs numeric IDs.
# Convert UUIDs to numeric device IDs using nvidia-smi before any Python import
# touches CUDA.
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Original CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

    # Check if the value contains GPU UUIDs (starts with "GPU-")
    if echo "$CUDA_VISIBLE_DEVICES" | grep -q "GPU-"; then
        if command -v nvidia-smi &> /dev/null; then
            converted=""
            IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"
            for dev in "${DEVICES[@]}"; do
                dev=$(echo "$dev" | xargs)  # trim whitespace
                if [[ "$dev" == GPU-* ]]; then
                    # Look up the numeric index for this UUID
                    idx=$(nvidia-smi --query-gpu=uuid,index --format=csv,noheader,nounits | \
                          grep "$dev" | head -1 | awk -F',' '{print $2}' | xargs)
                    if [ -n "$idx" ]; then
                        echo "Converted GPU UUID $dev -> device $idx"
                        converted="${converted:+$converted,}$idx"
                    else
                        echo "WARNING: Could not resolve UUID $dev, using 0"
                        converted="${converted:+$converted,}0"
                    fi
                else
                    converted="${converted:+$converted,}$dev"
                fi
            done
            export CUDA_VISIBLE_DEVICES="$converted"
            echo "Updated CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
        else
            # No nvidia-smi available; fall back to device 0
            echo "WARNING: nvidia-smi not found, setting CUDA_VISIBLE_DEVICES=0"
            export CUDA_VISIBLE_DEVICES=0
        fi
    fi
fi

# Execute relevance.py with the required arguments
python relevance.py --km_output "$km_output_file" --config "$config_file" --secrets "$secrets_file"
