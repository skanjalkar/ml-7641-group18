import chess.pgn
import numpy as np
from typing import Iterator
from config import MIN_MOVES, MOVE_SELECTION_PROBABILITY, MIN_CLOCK_TIME
from chess_utils import board_to_array, get_bin
from time_utils import get_clock_time

class GameProcessor:
    def __init__(self, pgn_file: str):
        self.pgn_file = pgn_file

    def process_games(self) -> Iterator[tuple]:
        with open(self.pgn_file) as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                yield from self.process_game(game)

    def process_game(self, game: chess.pgn.Game) -> Iterator[tuple]:
        if "Bullet" in game.headers["Event"]:
            return

        board = game.board()
        move_count = 0

        for node in game.mainline():
            move = node.move
            comment = node.comment
            clock_time = get_clock_time(comment)

            if (move_count >= MIN_MOVES and
                np.random.random() < MOVE_SELECTION_PROBABILITY and
                clock_time is not None and
                clock_time >= MIN_CLOCK_TIME):

                elo = game.headers["WhiteElo"] if move_count % 2 == 0 else game.headers["BlackElo"]
                board_array = board_to_array(board)
                board.push(move)
                ground_truth_board = board_to_array(board)
                data = np.stack((board_array, ground_truth_board), axis=-1)

                yield data, int(elo)
            else:
                board.push(move)
            move_count += 1
