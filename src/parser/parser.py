import numpy as np
from pathlib import Path
from game_processor import GameProcessor
from config import PGN_FILE_PATH, DATA_DIR, CHUNK_SIZE
from chess_utils import get_bin, create_bins_folders
from collections import defaultdict

# Initialize buffers for each ELO range
X_buffers = defaultdict(list)
y_buffers = defaultdict(list)
z_buffers = defaultdict(list)

def save_chunk(bin_name, X_chunk, y_chunk, z_chunk):
    bin_dir = DATA_DIR / "bins" / bin_name
    bin_dir.mkdir(parents=True, exist_ok=True)

    chunk_id = np.random.randint(1000000)
    X_path = bin_dir / f"X_chunk_{chunk_id}.npy"
    y_path = bin_dir / f"y_chunk_{chunk_id}.npy"
    z_path = bin_dir / f"z_chunk_{chunk_id}.npy"

    np.save(X_path, np.array(X_chunk))
    np.save(y_path, np.array(y_chunk))
    np.save(z_path, np.array(z_chunk))
    print(f"Saved {bin_name} chunk {chunk_id} with {len(X_chunk)} positions")

def main():
    # down the line change the path from PGN_FILE to
    # path from file downloaded directly from lichess
    np.random.seed(69)
    create_bins_folders()
    # Get all the .pgn files in the directory of PGN_FILE_PATH
    for PGN_FILE in PGN_FILE_PATH.glob("*.pgn"):
        print(f"Processing {PGN_FILE} .......")
        processor = GameProcessor(PGN_FILE)

        for X, y, elo in processor.process_games():
            bin_name = get_bin(elo)
            if bin_name:
                X_buffers[bin_name].append(X)
                y_buffers[bin_name].append(y)
                z_buffers[bin_name].append(elo)

                # Check if this bin's buffer has reached CHUNK_SIZE
                if len(X_buffers[bin_name]) >= CHUNK_SIZE:
                    save_chunk(bin_name, X_buffers[bin_name], y_buffers[bin_name], z_buffers[bin_name])
                    # Reset this bin's buffers
                    X_buffers[bin_name] = []
                    y_buffers[bin_name] = []
                    z_buffers[bin_name] = []

    # Save remaining data in all buffers
    for bin_name in X_buffers:
        if len(X_buffers[bin_name]) > 0:
            save_chunk(bin_name, X_buffers[bin_name], y_buffers[bin_name], z_buffers[bin_name])

if __name__ == "__main__":
    main()
