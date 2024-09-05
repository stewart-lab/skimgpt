#!/bin/bash

# Array of test leakage types
leakage_types=("neutral" "negative" "positive" "empty")

# Array of hypotheses
hypotheses=(
    "Treatment with {b_term} will worsen {a_term}."
    "Treatment with {b_term} will improve {a_term} patient outcomes."
    "Treatment with {b_term} will have no effect on {a_term} patient outcomes."
)

# Function to update the config file
update_config() {
    local leakage_type="$1"
    local hypothesis="$2"
    
    # Use jq to update the config file
    jq --arg leakage "$leakage_type" --arg hyp "$hypothesis" '
        .abstract_filter.TEST_LEAKAGE_TYPE = $leakage |
        .KM_hypothesis = $hyp
    ' /w5home/jfreeman/kmGPT/config.json > /w5home/jfreeman/kmGPT/config.json.tmp && mv /w5home/jfreeman/kmGPT/config.json.tmp /w5home/jfreeman/kmGPT/config.json
}

# Main loop
for hypothesis in "${hypotheses[@]}"; do
    for leakage_type in "${leakage_types[@]}"; do
        echo "Running analysis with leakage type: $leakage_type and hypothesis: $hypothesis"
        
        # Update the config file
        update_config "$leakage_type" "$hypothesis"
        
        # Run the analysis
        cd /w5home/jfreeman/kmGPT/src
        python main.py
        cd ..
        
        echo "Analysis completed for leakage type: $leakage_type and hypothesis: $hypothesis"
        echo "----------------------------------------"
    done
done

echo "All analyses completed."