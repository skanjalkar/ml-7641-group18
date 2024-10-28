import numpy as np
from pathlib import Path
from game_processor import GameProcessor
from config import PGN_FILE_PATH, DATA_DIR, CHUNK_SIZE, STOCKFISH_PATH, STOCKFISH_PATH_CUSTOM
from chess_utils import get_bin, create_bins_folders
from collections import defaultdict
import argparse
import threading
import chess.pgn
import io
from tqdm import tqdm
import concurrent.futures
from queue import Queue

def game_reader(pgn_file):
    games = []
    with open(pgn_file, 'r') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(str(game))
    return games

def worker(game_data, stockfish_path, global_blunder_count, lock):
    processor = GameProcessor(stockfish_path)
    results, blunder_count, not_blunder_count = processor.process_game(game_data)

    # Safely increment the global blunder count
    with lock:
        global_blunder_count[0] += blunder_count
        global_blunder_count[1] += not_blunder_count

    return results  # Return processed game results

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

def result_collector(results, X_buffers, y_buffers, z_buffers, lock):
    for result in results:
        if result is None:
            continue
        X, y, elo = result
        bin_name = get_bin(elo)
        if bin_name:
            with lock:
                if bin_name not in X_buffers:
                    X_buffers[bin_name] = []
                    y_buffers[bin_name] = []
                    z_buffers[bin_name] = []
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish_custom", action='store_true')
    args = parser.parse_args()
    stockfish_path = STOCKFISH_PATH_CUSTOM if args.stockfish_custom else STOCKFISH_PATH
    print(f"Using {'custom' if args.stockfish_custom else 'default'} stockfish engine")

    np.random.seed(69)
    create_bins_folders()

    X_buffers = defaultdict(list)
    y_buffers = defaultdict(list)
    z_buffers = defaultdict(list)
    lock = threading.Lock()  # Ensures thread-safe operations
    global_blunder_count = [0, 0]  # Use a list to allow mutable integer for threads

    # Gather all PGN files
    pgn_files = [PGN_FILE_PATH / f"{i}.pgn" for i in range(1, 65)]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        game_data_list = []
        for pgn_file in tqdm(pgn_files, desc="Reading games"):
            games = game_reader(pgn_file)
            game_data_list.extend(games)

        futures = [executor.submit(worker, game_data, stockfish_path, global_blunder_count, lock) for game_data in game_data_list]
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())

        result_collector(results, X_buffers, y_buffers, z_buffers, lock)

    # Save any remaining data in buffers
    for bin_name in X_buffers.keys():
        if X_buffers[bin_name]:
            save_chunk(bin_name, X_buffers[bin_name], y_buffers[bin_name], z_buffers[bin_name])

    print(f"Global blunder count: {global_blunder_count[0]}")
    print(f"Global not blunder count: {global_blunder_count[1]}")
    print("Processing complete.")

if __name__ == "__main__":
    main()
