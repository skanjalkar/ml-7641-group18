import chess
import numpy as np
from typing import Dict, Optional
import os

PIECE_DICT: Dict[str, int] = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
}

def create_8x8x17_board(board):
    """Convert chess board to 8x8x17 representation"""
    state = np.zeros((8, 8, 17), dtype=np.float32)

    # 12 channels for pieces (6 piece types × 2 colors)
    piece_channels = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
    }

    # Fill piece channels (0-11)
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(8*i + j)
            if piece is not None:
                state[i, j, piece_channels[piece.symbol()]] = 1

    # Castling rights (channels 12-15)
    state[:, :, 12] = float(board.has_kingside_castling_rights(chess.WHITE))
    state[:, :, 13] = float(board.has_queenside_castling_rights(chess.WHITE))
    state[:, :, 14] = float(board.has_kingside_castling_rights(chess.BLACK))
    state[:, :, 15] = float(board.has_queenside_castling_rights(chess.BLACK))

    # Active player (channel 16)
    state[:, :, 16] = float(board.turn)

    return state

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

def create_bins_folders():
    """Create the bin folders for the elo ratings.
    Path: data/bins/{elo-bin}
    """

    for lower_bound in range(1000, 2000, 100):
        upper_bound = lower_bound + 100
        bin_dir = f"{lower_bound}-{upper_bound}"
        os.makedirs(f"data/bins/{bin_dir}", exist_ok=True)

    print("Bins folders created successfully.")
    print("==================================")
