#!/bin/bash

# Ask the user if they would like to conduct their analysis using abstract filtering
read -p "Would you like to conduct your analysis using abstract filtering (Note: Need to configure CHTC key under config file)? [y/N] " answer

# Convert the answer to lower case
answer=$(echo "$answer" | tr '[:upper:]' '[:lower:]')

# Change directory to src
cd src

# Decision based on the user's answer
if [[ "$answer" == "y" ]]; then
    echo "Running CHTC analysis..."
    # Run chtc.py
    python chtc.py
else
    echo "Running abstract comprehension analysis..."
    # Run abstract_comprehension.py
    python abstract_comprehension.py
fi
