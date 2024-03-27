#!/bin/bash
echo "Running job on $(hostname)"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
# Navigate to the /app directory where main.py and the input files are expected to be
chmod +x main.py
# Find the .tsv file for the --km_output argument
km_output_file=$1
# Find the .json file for the --config argument
config_file="config.json"
export TRANSFORMERS_CACHE=$_CONDOR_SCRATCH_DIR/models
export HF_HOME=$_CONDOR_SCRATCH_DIR/models
export HF_DATASETS_CACHE=$_CONDOR_SCRATCH_DIR/datasets
export HF_MODULES_CACHE=$_CONDOR_SCRATCH_DIR/modules
export HF_METRICS_CACHE=$_CONDOR_SCRATCH_DIR/metrics
# Execute main.py with the required arguments
python relevance.py --km_output "$km_output_file" --config "$config_file"