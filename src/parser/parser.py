import numpy as np
from pathlib import Path
from game_processor import GameProcessor
from config import PGN_FILE, DATA_DIR
from chess_utils import get_bin, create_bins_folders

def main():
    # down the line change the path from PGN_FILE to
    # path from file downloaded directly from lichess
    np.random.seed(69)
    create_bins_folders()
    processor = GameProcessor(PGN_FILE)
    for data, elo in processor.process_games():
        bin = get_bin(elo)
        if bin:
            bin_dir = DATA_DIR / "bins" / bin
            bin_dir.mkdir(parents=True, exist_ok=True)
            file_path = bin_dir / f"{elo}-{np.random.randint(1000000)}.npy"
            np.save(file_path, data)

if __name__ == "__main__":
    main()
