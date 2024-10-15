import chess
import numpy as np
from typing import Dict, Optional

PIECE_DICT: Dict[str, int] = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
}

def board_to_array(board: chess.Board) -> np.ndarray:
    """Convert a chess board to a numpy array."""
    board_array = np.zeros((8, 8), dtype=np.int8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            board_array[row, col] = PIECE_DICT[piece.symbol()]
    return board_array

def get_bin(elo: int) -> Optional[str]:
    """Get the bin for the elo rating."""
    if 1000 <= elo < 2000:
        lower_bound = (elo // 100) * 100
        upper_bound = lower_bound + 100
        return f"{lower_bound}-{upper_bound}"
    return None
