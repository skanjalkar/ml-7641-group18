import os
import requests
import zstandard as zstd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Base URL for downloading files
base_url = "https://database.lichess.org/standard/"

# Updated list of files to match the ones you have
files = [
    "lichess_db_standard_rated_2024-08.pgn.zst",
    "lichess_db_standard_rated_2024-07.pgn.zst",
]

# files = [
#     "lichess_db_standard_rated_2013-01.pgn.zst"
# ]

# Corrected destination directory
destination_dir = Path("data/pgn-data")

# Create the destination directory if it doesn't exist
destination_dir.mkdir(parents=True, exist_ok=True)

def process_file(file, position):
    # Adjusted path to the .zst file
    local_zst_path = Path('scripts') / file
    local_pgn_path = destination_dir / local_zst_path.with_suffix('').name  # Remove .zst extension
    try:
        # Download the file
        url = base_url + file
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        # Save the .zst file with progress bar
        with open(local_zst_path, 'wb') as f, tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            desc=f"Downloading {file}",
            position=position*2,
            leave=False,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        # Decompress the .zst file to .pgn
        compressed_size = local_zst_path.stat().st_size

        with open(local_zst_path, 'rb') as ifh, open(local_pgn_path, 'wb') as ofh, tqdm(
            total=compressed_size,
            unit='iB',
            unit_scale=True,
            desc=f"Decompressing {file}",
            position=position*2 + 1,
            leave=False,
        ) as pbar:
            dctx = zstd.ZstdDecompressor()
            reader = dctx.stream_reader(ifh)
            while True:
                chunk = reader.read(8192)
                if not chunk:
                    break
                ofh.write(chunk)
                pbar.update(len(chunk))

        # Remove the original .zst file if desired
        # os.remove(local_zst_path)
    except Exception as e:
        print(f"Failed to process {file}: {e}")

def main():
    # Check if zstandard is installed
    try:
        import zstandard
    except ImportError:
        print("The 'zstandard' library is not installed. Please install it using 'pip install zstandard'")
        return

    max_workers = min(len(files), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for position, file in enumerate(files):
            futures.append(executor.submit(process_file, file, position))
        for future in futures:
            future.result()

    print("All files processed.")

if __name__ == "__main__":
    main()
