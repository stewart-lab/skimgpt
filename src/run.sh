#!/bin/bash
echo "Running job on $(hostname)"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

# Debugging: Print all arguments received by run.sh
echo "Number of arguments received: $#"
echo "Arguments received: $*"
echo "First argument (\$1): '$1'"
echo "Second argument (\$2): '$2'" # Just in case something unexpected is happening
exec 2>> run_$(Cluster)_$(Process).err

echo "--- Network Connectivity Test (using curl) ---"
date

# Function to test URL and check HTTP status
test_url() {
  URL="$1"
  NAME="$2"
  echo "--- Testing ${NAME} (URL: ${URL}) ---"
  if status_code=$(curl -Is --connect-timeout 10 --max-time 30 "${URL}" -o /dev/null | head -n 1 | awk '{print $2}'); then
    if [[ "$status_code" =~ ^(2|3)[0-9]{2}$ ]]; then # Check if status code starts with 2 or 3 (success)
      echo "--- ${NAME} connectivity test: SUCCESS, Status Code: ${status_code} ---"
    else
      echo "--- ${NAME} connectivity test: WARNING, Unexpected Status Code: ${status_code} ---"
    fi
  else
    echo "--- ${NAME} connectivity test: FAILED ---"
    echo "--- FATAL ERROR: Connectivity to ${NAME} failed. Exiting script. ---"
    exit 1
  fi
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
echo "Item (from macro): $(Item)"       # Macro, might not work directly as env var
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

