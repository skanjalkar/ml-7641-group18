import chess.pgn
import numpy as np
from stockfish import Stockfish
from config import MIN_MOVES, MOVE_SELECTION_PROBABILITY, MIN_CLOCK_TIME, BLUNDER_THRESHOLD
from chess_utils import create_8x8x17_board
from time_utils import get_clock_time
import io

class GameProcessor:
    def __init__(self, stockfish_path):
        self.stockfish = Stockfish(path=stockfish_path)
        self.stockfish.update_engine_parameters({
            "Threads": 1,
            "Hash": 128,
            "Minimum Thinking Time": 0,
            "Skill Level": 20
        })
        self.blunder = 0
        self.not_blunder = 0

    def process_game(self, game_data):
        game = chess.pgn.read_game(io.StringIO(game_data))
        if game is None or "Bullet" in game.headers.get("Event", ""):
            return []

        board = game.board()
        move_count = 0
        results = []

        for node in game.mainline():
            move = node.move
            comment = node.comment
            clock_time = get_clock_time(comment)

            if (move_count >= MIN_MOVES and
                clock_time is not None and
                clock_time >= MIN_CLOCK_TIME):

                # Set up board position and get evaluation before and after move
                self.stockfish.set_fen_position(board.fen())
                eval_before = self.stockfish.get_evaluation()

                board.push(move)
                self.stockfish.set_fen_position(board.fen())
                eval_after = self.stockfish.get_evaluation()

                # Calculate evaluation difference and check for blunders
                eval_diff = abs(self.get_eval_value(eval_after) - self.get_eval_value(eval_before))
                is_blunder = eval_diff >= BLUNDER_THRESHOLD

                # Sampling based on move selection probability and blunder status
                if (
                    np.random.random() < MOVE_SELECTION_PROBABILITY or
                    (is_blunder and np.random.random() < 0.75)
                ):
                    elo = game.headers.get("WhiteElo" if move_count % 2 == 0 else "BlackElo", "1500")
                    try:
                        elo = int(elo)
                    except ValueError:
                        elo = 1500  # Default Elo if parsing fails

                    board_array = create_8x8x17_board(board)
                    ground_truth = is_blunder
                    results.append((board_array, ground_truth, elo))
                    self.blunder += is_blunder
                    self.not_blunder += not is_blunder
            else:
                board.push(move)
            move_count += 1

        return results, self.blunder, self.not_blunder

    @staticmethod
    def get_eval_value(eval_dict):
        """Convert Stockfish evaluation to numerical value"""
        if eval_dict['type'] == 'cp':
            return eval_dict['value']
        elif eval_dict['type'] == 'mate':
            # Convert mate score to high centipawn value
            return 10000 * (1 if eval_dict['value'] > 0 else -1)
        return 0
