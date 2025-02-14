#!/bin/bash
echo "Running job on $(hostname)"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

# Debugging: Print all arguments received by run.sh
echo "Number of arguments received: $#"
echo "Arguments received: $*"
echo "First argument (\$1): '$1'"
echo "Second argument (\$2): '$2'" # Just in case something unexpected is happening

echo "--- Network Connectivity Test (using curl) ---"
date

# Function to test URL and check HTTP status
test_url() {
  URL="$1"
  NAME="$2"
  echo "--- Testing ${NAME} (URL: ${URL}) ---"
  echo "--- Raw curl -Is output for ${NAME}: ---"
  curl -Is --connect-timeout 10 --max-time 30 "${URL}" 2>&1 # Redirect stderr to stdout to capture all output
  echo "--- End raw curl output for ${NAME} ---"
}

# Test Google (basic internet)
test_url "http://google.com" "Google (HTTP)"

# Test PubMed E-utilities (HTTPS)
test_url "https://eutils.ncbi.nlm.nih.gov" "PubMed E-utilities (HTTPS)"

# Test Hugging Face (HTTPS)
test_url "https://huggingface.co" "Hugging Face (HTTPS)"

echo "--- End Network Connectivity Test ---"


# Debugging: Try printing standard HTCondor environment variables
echo "--- HTCondor Environment Variables ---"
echo "_CONDOR_ITEM (env var): $_CONDOR_ITEM" # Common env var for Item
echo "_CONDOR_JOBID (env var): $_CONDOR_JOBID"
echo "_CONDOR_CLUSTERID (env var): $_CONDOR_CLUSTERID"
echo "_CONDOR_PROCID (env var): $_CONDOR_PROCID"
echo "_CONDOR_JOBAD_RAW (env var): $_CONDOR_JOBAD_RAW" # Raw JobAd (might be large)
echo "--- End of HTCondor Environment Variables ---"

# Get km_output_file from environment variable
km_output_file=files.txt
config_file="config.json"

export TRANSFORMERS_CACHE=$_CONDOR_SCRATCH_DIR/models
export HF_HOME=$_CONDOR_SCRATCH_DIR/models
export HF_DATASETS_CACHE=$_CONDOR_SCRATCH_DIR/datasets
export HF_MODULES_CACHE=$_CONDOR_SCRATCH_DIR/modules
export HF_METRICS_CACHE=$_CONDOR_SCRATCH_DIR/metrics
# Execute main.py with the required arguments
python relevance.py --km_output "$km_output_file" --config "$config_file"

