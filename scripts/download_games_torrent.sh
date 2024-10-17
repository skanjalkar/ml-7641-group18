#!/bin/bash

# Base URL for downloading files
base_url="https://database.lichess.org/standard/"

# List of files to download
#"lichess_db_standard_rated_2024-09.pgn.zst"

# if you want to test this script
# Try this file first
# 200 mb only
# "lichess_db_standard_rated_2014-01.pgn.zst"
# "lichess_db_standard_rated_2014-02.pgn.zst"

# If you need more games, you can download the following files
#"lichess_db_standard_rated_2024-06.pgn.zst"
#"lichess_db_standard_rated_2024-05.pgn.zst"
#"lichess_db_standard_rated_2024-04.pgn.zst"
#"lichess_db_standard_rated_2024-03.pgn.zst"
#"lichess_db_standard_rated_2024-02.pgn.zst"
#"lichess_db_standard_rated_2024-01.pgn.zst"

# "lichess_db_standard_rated_2024-08.pgn.zst"
# "lichess_db_standard_rated_2024-07.pgn.zst"

files=(
    "lichess_db_standard_rated_2024-08.pgn.zst"
    "lichess_db_standard_rated_2024-07.pgn.zst"
)

# Directory where .pgn files will be moved
destination_dir="./data/pgn-data"

# Check if zstd is installed
if ! command -v zstd &> /dev/null; then
    echo "zstd is not installed. Please install it to proceed."
    exit 1
fi

# Create the destination directory if it doesn't exist
mkdir -p "$destination_dir"

# Function to download, decompress, and move a file
process_file() {
    local file=$1
    echo "Downloading $file..."
    wget "${base_url}${file}" -O "$file"

    # Check if the download was successful
    if [[ -f "$file" ]]; then
        echo "Decompressing $file..."
        unzstd "$file" -o "${file%.zst}"

        # Check if decompression was successful
        if [[ -f "${file%.zst}" ]]; then
            echo "Decompression completed: ${file%.zst}"

            # Move the .pgn file to the destination directory
            mv "${file%.zst}" "$destination_dir/"
            echo "Moved ${file%.zst} to $destination_dir."

            # Optionally, remove the original .zst file
            rm "$file"
        else
            echo "Failed to decompress $file."
        fi
    else
        echo "Failed to download $file."
    fi
}

# Run downloads in parallel using background jobs
for file in "${files[@]}"; do
    process_file "$file" &
done

# Wait for all background jobs to finish
wait

echo "All files processed."
