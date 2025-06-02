#!/bin/bash
# HTCondor job execution script for skimgpt relevance analysis
# 
# REQUIREMENTS:
# - The skimgpt package must be installed in the execution environment
# - Install with: pip install /path/to/skimgpt/ or pip install skimgpt
# - Or include package installation in HTCondor submit file
#
# This script uses the skimgpt-relevance entry point instead of copying source files

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
secrets_file="secrets.json"

export TRANSFORMERS_CACHE=$_CONDOR_SCRATCH_DIR/models
export HF_HOME=$_CONDOR_SCRATCH_DIR/models
export HF_DATASETS_CACHE=$_CONDOR_SCRATCH_DIR/datasets
export HF_MODULES_CACHE=$_CONDOR_SCRATCH_DIR/modules
export HF_METRICS_CACHE=$_CONDOR_SCRATCH_DIR/metrics

# Set GPU mode for skimgpt package
export SKIMGPT_GPU_MODE=true

# Verify skimgpt package installation and entry point availability
echo "--- Verifying skimgpt package installation ---"
python -c "import src.relevance; import src.utils; print('✅ Core skimgpt modules imported successfully')" 2>&1
if [ $? -ne 0 ]; then
    echo "WARNING: skimgpt package modules not found. Attempting installation..."
    
    # Look for setup.py in parent directories (common HTCondor pattern)
    if [ -f "../setup.py" ]; then
        echo "Found setup.py in parent directory, installing package..."
        cd ..
        pip install -e . --user
        cd -
    elif [ -f "../../setup.py" ]; then
        echo "Found setup.py in grandparent directory, installing package..."
        cd ../..
        pip install -e . --user
        cd -
    else
        echo "ERROR: setup.py not found and skimgpt package not installed"
        exit 1
    fi
    
    # Verify installation after attempt
    python -c "import src.relevance; import src.utils; print('✅ Core skimgpt modules imported successfully')" 2>&1
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install skimgpt package"
        exit 1
    fi
    echo "Successfully installed skimgpt package"
    
    # Update PATH to include user-installed entry points
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check if entry point is available
echo "Checking skimgpt-relevance entry point availability..."
which skimgpt-relevance 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: skimgpt-relevance entry point not found. Package may not be properly installed."
    exit 1
fi

# Execute using the skimgpt package entry point
echo "Executing skimgpt-relevance with arguments: --km_output '$km_output_file' --config '$config_file' --secrets '$secrets_file'"
skimgpt-relevance --km_output "$km_output_file" --config "$config_file" --secrets "$secrets_file"

