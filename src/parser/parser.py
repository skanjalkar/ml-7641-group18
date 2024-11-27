import numpy as np
from pathlib import Path
from game_processor import GameProcessor
from config import PGN_FILE_PATH, DATA_DIR, CHUNK_SIZE, STOCKFISH_PATH, STOCKFISH_PATH_CUSTOM
from chess_utils import get_bin, create_bins_folders
from collections import defaultdict
import argparse

# Initialize buffers for each ELO range
X_buffers = defaultdict(list)  # Will store feature vectors instead of board arrays
y_buffers = defaultdict(list)  # Will store blunder labels

def save_chunk(bin_name, X_chunk, y_chunk, pgn_number):
    bin_dir = DATA_DIR / "bins" / bin_name
    bin_dir.mkdir(parents=True, exist_ok=True)

    chunk_id = np.random.randint(1000000)
    X_path = bin_dir / f"X_chunk_{chunk_id}_{pgn_number}.npy"
    y_path = bin_dir / f"y_chunk_{chunk_id}_{pgn_number}.npy"

    np.save(X_path, np.array(X_chunk))
    np.save(y_path, np.array(y_chunk))
    print(f"Saved {bin_name} chunk {chunk_id} with {len(X_chunk)} positions")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish_custom", type=bool, default=False)
    parser.add_argument("--pgn_number", type=int, default=1)

    args = parser.parse_args()
    stockfish_custom = args.stockfish_custom
    pgn_number = args.pgn_number

    PGN_FILE = PGN_FILE_PATH / f"{pgn_number}.pgn"
    # PGN_FILE = PGN_FILE_PATH / "test.pgn"

    np.random.seed(69)
    create_bins_folders()
    print(f"Processing {PGN_FILE} .......")

    processor = None
    if (stockfish_custom):
        processor = GameProcessor(PGN_FILE, STOCKFISH_PATH_CUSTOM)
        print("Using custom stockfish engine")
    else:
        processor = GameProcessor(PGN_FILE, STOCKFISH_PATH)
        print("Using default stockfish engine")

    # Process games and store features
    for features, is_blunder in processor.process_games():
        # Get the ELO from the features array (index 3 contains current_elo)
        elo = int(features[3])
        bin_name = get_bin(elo)

        if bin_name:
            X_buffers[bin_name].append(features)
            y_buffers[bin_name].append(is_blunder)

            # Check if this bin's buffer has reached CHUNK_SIZE
            if len(X_buffers[bin_name]) >= CHUNK_SIZE:
                save_chunk(bin_name, X_buffers[bin_name], y_buffers[bin_name], pgn_number)

                # Reset this bin's buffers
                X_buffers[bin_name] = []
                y_buffers[bin_name] = []

    # Save remaining data in all buffers
    for bin_name in X_buffers:
        if len(X_buffers[bin_name]) > 0:
            save_chunk(bin_name, X_buffers[bin_name], y_buffers[bin_name], pgn_number)

if __name__ == "__main__":
    main()
