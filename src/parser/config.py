from pathlib import Path

MIN_MOVES = 20
MOVE_SELECTION_PROBABILITY = 1/32
MIN_CLOCK_TIME = 30
DATA_DIR = Path("./data")
PGN_FILE_PATH = DATA_DIR / "pgn-data"
STOCKFISH_PATH = "/usr/games/stockfish"
STOCKFISH_PATH_CUSTOM = "../Stockfish/src/stockfish"
BLUNDER_THRESHOLD = 200
CHUNK_SIZE = 512
