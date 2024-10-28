import chess.pgn
import numpy as np
from typing import Iterator

from stockfish.models import Stockfish
from config import MIN_MOVES, MOVE_SELECTION_PROBABILITY, MIN_CLOCK_TIME, BLUNDER_THRESHOLD, STOCKFISH_PATH
from chess_utils import board_to_array, get_bin, create_8x8x17_board
from time_utils import get_clock_time
import stockfish

class GameProcessor:
    def __init__(self, pgn_file: str, stockfish_path: str = STOCKFISH_PATH):
        self.pgn_file = pgn_file
        self.stockfish = self.stockfish_initialize(stockfish_path=stockfish_path)
        self.game_count = 0
        self.blunder_count = 0

    def stockfish_initialize(self, stockfish_path: str = STOCKFISH_PATH):
        stockfish = Stockfish(STOCKFISH_PATH)
        stockfish.update_engine_parameters({
                "Threads": 1,
                "Hash": 128,
                "Minimum Thinking Time": 10,
                "Skill Level": 0
            }
        )
        return stockfish

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
        self.game_count += 1
        print("Game count : ", self.game_count, "  Processing game....")
        move_count = 0

        for node in game.mainline():
            move = node.move
            comment = node.comment
            clock_time = get_clock_time(comment)

            if (move_count >= MIN_MOVES and
                clock_time is not None and
                clock_time >= MIN_CLOCK_TIME):

                # Get evaluation before move
                self.stockfish.set_fen_position(board.fen())
                eval_before = self.stockfish.get_evaluation()

                # Make the move and get evaluation after
                board.push(move)
                self.stockfish.set_fen_position(board.fen())
                eval_after = self.stockfish.get_evaluation()


                # Calculate evaluation difference
                eval_diff = abs(self.get_eval_value(eval_after) - self.get_eval_value(eval_before))
                is_blunder = eval_diff >= BLUNDER_THRESHOLD

                # Adjust sampling probability based on whether it's a blunder
                should_sample = (
                    np.random.random() < MOVE_SELECTION_PROBABILITY # normal sampling
                )

                if should_sample:
                    elo = game.headers["WhiteElo"] if move_count % 2 == 0 else game.headers["BlackElo"]
                    # if (is_blunder):
                    #     print("Yielding move that is a blunder")
                    # else:
                    #     print("Yielding move that is not a blunder")
                    board_array = create_8x8x17_board(board)
                    ground_truth = is_blunder
                    self.blunder_count += 1 if is_blunder else 0

                    # yield board_array, ground_truth, and elo
                    yield_tuple = (board_array, ground_truth, int(elo), self.blunder_count)
                    yield yield_tuple
            else:
                board.push(move)
            move_count += 1

    def get_eval_value(self, eval_dict):
            """Convert Stockfish evaluation to numerical value"""
            if eval_dict['type'] == 'cp':
                return eval_dict['value']
            elif eval_dict['type'] == 'mate':
                # Convert mate score to high centipawn value
                return 10000 * (1 if eval_dict['value'] > 0 else -1)
            return 0
            import re
            from typing import Optional

            def time_to_seconds(time_str: str) -> int:
                """Convert time string to seconds."""
                parts = time_str.split(':')
                if len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                elif len(parts) == 2:
                    return int(parts[0]) * 60 + int(parts[1])
                else:
                    return int(parts[0])
