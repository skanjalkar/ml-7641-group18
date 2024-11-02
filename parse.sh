#!/bin/bash

# Get the upper limit for pgn_number from the user
read -p "Enter the upper limit for pgn_number: " upper_limit

# Loop from 1 to the user-defined upper limit
for i in $(seq 1 "$upper_limit"); do
    # Run each process in the background
    python3 src/parser/parser.py --stockfish_custom=True --pgn_number="$i" &
done

# Wait for all background processes to complete
wait

echo "All tasks completed."
