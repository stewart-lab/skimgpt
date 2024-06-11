#!/bin/bash

# Check if the parent directory is supplied as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <parent_dir>"
    exit 1
fi

# Define the parent directory from the argument
parent_dir="$1"

# Navigate to the parent directory
cd "$parent_dir" || { echo "Failed to navigate to the directory: $parent_dir"; exit 1; }

# Check if the Python script exists in the current directory
if [[ ! -f "eval_JSON_results.py" ]]; then
    echo "Python script 'eval_JSON_results.py' not found in the current directory."
    exit 1
fi

# Loop through each child directory in the parent directory
for child_dir in */; do
    # Ensure the path is a directory
    if [[ -d "$child_dir" ]]; then
        echo "Processing directory: $child_dir"
        # Run the Python script with the directory as an argument
        python eval_JSON_results.py "${parent_dir}/${child_dir}"
    fi
done

# Run the plot.py script with the parent_dir as an argument
python plot.py "$parent_dir"

echo "All directories have been processed."
