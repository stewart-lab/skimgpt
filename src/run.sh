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
# Execute main.py with the required arguments
python relevance.py --km_output "$km_output_file" --config "$config_file" --secrets "$secrets_file"

