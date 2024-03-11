#!/bin/bash

# Navigate to the /app directory where main.py and the input files are expected to be
chmod +x main.py

# Find the .tsv file for the --km_output argument
km_output_file=$(find . -name "*.tsv" -print -quit)

# Find the .json file for the --config argument
config_file=$(find . -name "*.json" -print -quit)

# Specify the output file name for the --output_file argument
output_file="filtered_output.tsv"

# Execute main.py with the required arguments
python main.py --km_output "$km_output_file" --config "$config_file" --output_file "$output_file"
