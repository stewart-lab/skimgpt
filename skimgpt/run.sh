#!/bin/bash

km_output_file=files.txt
config_file="config.json"

export HF_HOME=$_CONDOR_SCRATCH_DIR/models

# --- Fix CUDA_VISIBLE_DEVICES before Python/CUDA initialization ---
# HTCondor may set CUDA_VISIBLE_DEVICES to a UUID rather than a numeric index:
# "GPU-abc123..." for a whole physical GPU, or "MIG-abc123..." on a node where
# the job was assigned a MIG partition instead. vLLM's own arg parsing rejects
# UUID-form CUDA_VISIBLE_DEVICES outright (even though CUDA/PyTorch resolve it
# fine), so this needs converting to a numeric ID before any Python import
# touches CUDA.
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Original CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

    # Trigger on any non-numeric value, not just "GPU-" -- MIG UUIDs and any
    # other future UUID form need the same handling.
    if ! [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9,[:space:]]+$ ]]; then
        if command -v nvidia-smi &> /dev/null; then
            # When exactly one GPU/MIG device is visible in this container --
            # the normal case for a single-GPU HTCondor job -- there's no
            # ambiguity: it's device 0, regardless of UUID prefix. This
            # sidesteps needing a UUID->index lookup for MIG instances (which
            # don't show up in `nvidia-smi --query-gpu`, only whole physical
            # GPUs do), since it's unnecessary whenever there's nothing else
            # the ID could refer to.
            device_count=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | wc -l)
            if [ "$device_count" -eq 1 ]; then
                echo "Exactly one GPU visible in this container; using device 0"
                export CUDA_VISIBLE_DEVICES=0
            else
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
                        echo "WARNING: Cannot resolve device identifier $dev among $device_count visible devices; using 0"
                        converted="${converted:+$converted,}0"
                    fi
                done
                export CUDA_VISIBLE_DEVICES="$converted"
            fi
            echo "Updated CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
        else
            # No nvidia-smi available; fall back to device 0
            echo "WARNING: nvidia-smi not found, setting CUDA_VISIBLE_DEVICES=0"
            export CUDA_VISIBLE_DEVICES=0
        fi
    fi
fi

# Execute relevance_chtc.py with the required arguments
python relevance_chtc.py --km_output "$km_output_file" --config "$config_file"
