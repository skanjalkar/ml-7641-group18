#!/bin/bash

# Parent directory containing the subdirectories like 1000-1100, 1100-1200, etc.
TARGET_DIR="./data/bins/"

# Find and delete all .npy files in the subdirectories
find "$TARGET_DIR" -type f -name "*.npy" -delete

echo "All .npy files have been deleted from $TARGET_DIR and its subdirectories."
