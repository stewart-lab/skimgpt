#!/bin/bash
echo "Running job on `hostname`"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

# Navigate to the /app directory where main.py and the input files are expected to be
chmod +x main.py

# Find the .tsv file for the --km_output argument
# km_output_file=$(find . -name "*.tsv" -print -quit)
km_output_file="data.tsv"

# Find the .json file for the --config argument
# config_file=$(find . -name "*.json" -print -quit)
config_file="config.json"

# Specify the output file name for the --output_file argument
output_tsv="filtered.tsv"

output_cot="cot.tsv"
export TRANSFORMERS_CACHE=$_CONDOR_SCRATCH_DIR/models
export HF_HOME=$_CONDOR_SCRATCH_DIR/models
export HF_DATASETS_CACHE=$_CONDOR_SCRATCH_DIR/datasets
export HF_MODULES_CACHE=$_CONDOR_SCRATCH_DIR/modules
export HF_METRICS_CACHE=$_CONDOR_SCRATCH_DIR/metrics


# Execute main.py with the required arguments
python main.py --km_output "$km_output_file" --config "$config_file" --filtered_tsv_name "$output_file" --cot_tsv_name "$output_cot"