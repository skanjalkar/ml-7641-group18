#!/bin/bash

# Get the upper limit for pgn_number from the user
read -p "Enter the upper limit for pgn_number: " upper_limit
read -p "Do you want to use Feature Extraction (True/False): " feature_extraction

if ! [[ "$upper_limit" =~ ^[0-9]+$ ]]; then
    echo "Error: Please enter a valid number for upper limit"
    exit 1
fi

if ! [[ "$feature_extraction" =~ ^(True|False)$ ]]; then
    echo "Error: Please enter True or False for feature extraction"
    exit 1
fi

# Loop from 1 to the user-defined upper limit
for i in $(seq 1 "$upper_limit"); do
    # Run each process in the background
    python3 src/parser/parser.py --stockfish_custom=True --pgn_number="$i" --feature="$feature_extraction"&
done

# Wait for all background processes to complete
wait

echo "All tasks completed."
