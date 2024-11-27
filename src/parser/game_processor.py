import chess.pgn
import numpy as np
from typing import Iterator

from stockfish.models import Stockfish
from config import MIN_MOVES, MOVE_SELECTION_PROBABILITY, MIN_CLOCK_TIME, BLUNDER_THRESHOLD, STOCKFISH_PATH, BLUNDER_PROBABILITY
from chess_utils import board_to_array, get_bin, create_8x8x17_board
from time_utils import get_clock_time

# Game count :  1   Processing game....
# Features :  [ 3.000e+01  2.730e+02  2.950e+02  1.167e+03  2.000e+00  1.000e+00
#  2.300e+01  2.000e+00  3.500e+01  3.000e+00 -2.170e+02  0.000e+00
#  0.000e+00 -2.000e+00  1.000e+00  0.000e+00  0.000e+00  1.000e+00
#  1.000e+00 -1.000e+00  2.470e+02]

class GameProcessor:
    def __init__(self, pgn_file: str, stockfish_path: str = STOCKFISH_PATH):
        self.pgn_file = pgn_file
        self.stockfish = self.stockfish_initialize(stockfish_path=stockfish_path)
        self.game_count = 0


    def eco_to_number(self, eco: str) -> int:
        """Convert ECO code to numerical value
        A00 = 0, A01 = 1, ..., E99 = 499"""

        # https://en.wikipedia.org/wiki/Encyclopaedia_of_Chess_Openings#A
        # for future reference if something goes wrong ^
        if not eco or len(eco) != 3:
            return 0  # Default value for unknown/invalid ECO

        try:
            letter = eco[0]
            number = int(eco[1:])
            # Convert letter to base number (A=0, B=100, C=200, D=300, E=400)
            base = (ord(letter) - ord('A')) * 100
            return base + number
        except:
            return 0

    def stockfish_initialize(self, stockfish_path: str = STOCKFISH_PATH):
        stockfish = Stockfish(stockfish_path)
        stockfish.update_engine_parameters({
                "Threads": 1,
                "Hash": 128,
                "Minimum Thinking Time": 0,
                "Skill Level": 20
            }
        )
        return stockfish

    def process_games(self) -> Iterator[tuple]:
        with open(self.pgn_file) as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                # if elo not in range 1000-2000 for either white or black, skip the game
                if game is None:
                    break

                white_elo, black_elo = 0, 0
                if ("WhiteElo" in game.headers):
                    white_elo = int(game.headers["WhiteElo"])
                if ("BlackElo" in game.headers):
                    black_elo = int(game.headers["BlackElo"])

                if (white_elo < 1000 or white_elo > 2000 or black_elo < 1000 or black_elo > 2000):
                    continue

                if "Bullet" in game.headers["Event"]:
                    print("Skipping bullet game")
                    continue

                yield from self.process_game(game)

    def process_game(self, game: chess.pgn.Game) -> Iterator[tuple]:

        board = game.board()
        self.game_count += 1
        print("Game count : ", self.game_count, "  Processing game....")
        # print("Game Headers" , game.headers)
        move_count = 0
        current_position_blunders = 0
        TIME_CONTROL = game.headers["TimeControl"]
        # format = "time+increment"
        time = TIME_CONTROL.split("+")[0]
        increment = TIME_CONTROL.split("+")[1]
        time_white = int(time)
        time_black = int(time)
        elo_black = game.headers["BlackElo"]
        elo_white = game.headers["WhiteElo"]
        elo_diff = abs(int(elo_white) - int(elo_black))
        eco = game.headers["ECO"]
        eco_number = self.eco_to_number(eco)


        for node in game.mainline():
            move = node.move
            comment = node.comment
            clock_time = get_clock_time(comment)

            legal_moves_number = board.legal_moves.count()
            turn = float(board.turn)

            if (turn == 0):
                time_white -= int(clock_time)
            else:
                time_black -= int(clock_time)

            time_diff = abs(time_white - time_black)


            if (move_count >= MIN_MOVES):

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
                    np.random.random() < MOVE_SELECTION_PROBABILITY or
                    (is_blunder and np.random.random() < BLUNDER_PROBABILITY)
                )

                if should_sample:
                    piece_counts = {
                        'P': len(board.pieces(chess.PAWN, chess.WHITE)),
                        'p': len(board.pieces(chess.PAWN, chess.BLACK)),
                        'N': len(board.pieces(chess.KNIGHT, chess.WHITE)),
                        'n': len(board.pieces(chess.KNIGHT, chess.BLACK)),
                        'B': len(board.pieces(chess.BISHOP, chess.WHITE)),
                        'b': len(board.pieces(chess.BISHOP, chess.BLACK)),
                        'R': len(board.pieces(chess.ROOK, chess.WHITE)),
                        'r': len(board.pieces(chess.ROOK, chess.BLACK)),
                        'Q': len(board.pieces(chess.QUEEN, chess.WHITE)),
                        'q': len(board.pieces(chess.QUEEN, chess.BLACK)),
                    }

                    total_pieces = sum(piece_counts.values())
                    piece_mismatch = abs(
                        sum(v for k, v in piece_counts.items() if k.isupper()) -  # white pieces
                        sum(v for k, v in piece_counts.items() if k.islower())    # black pieces
                    )
                    queens_on_board = piece_counts['Q'] + piece_counts['q']
                    current_elo = elo_white if turn else elo_black
                    piece_type_advantages = {
                        'bishop_vs_knight': 1 if piece_counts['B'] > piece_counts['N'] else
                                            (-1 if piece_counts['B'] < piece_counts['N'] else 0),
                        'rook_vs_minor': 1 if piece_counts['R'] > (piece_counts['B'] + piece_counts['N']) else
                                        (-1 if piece_counts['R'] < (piece_counts['B'] + piece_counts['N']) else 0),
                        'queen_vs_rooks': 1 if piece_counts['Q'] > piece_counts['R'] else
                                        (-1 if piece_counts['Q'] < piece_counts['R'] else 0),
                    }

                    features = np.array([
                        legal_moves_number,          # 0: legal moves
                        clock_time,                  # 1: time
                        time_diff,                   # 2: time difference
                        current_elo,                 # 3: current player elo
                        elo_diff,                    # 4: elo difference
                        piece_mismatch,              # 5: overall piece mismatch
                        total_pieces,                # 6: total pieces
                        queens_on_board,             # 7: queens on board
                        move_count,                  # 8: move number
                        increment,                   # 9: time increment
                        self.get_eval_value(eval_before),                 # 10: evaluation
                        turn,                        # 11: turn (0 for black, 1 for white)
                        piece_counts['P'] - piece_counts['p'],  # 12: pawn mismatch
                        piece_counts['N'] - piece_counts['n'],  # 13: knight mismatch
                        piece_counts['B'] - piece_counts['b'],  # 14: bishop mismatch
                        piece_counts['R'] - piece_counts['r'],  # 15: rook mismatch
                        piece_counts['Q'] - piece_counts['q'],  # 16: queen mismatch
                        piece_type_advantages['bishop_vs_knight'],    # 17: bishop vs knight advantage
                        piece_type_advantages['rook_vs_minor'],       # 18: rook vs minor advantage
                        piece_type_advantages['queen_vs_rooks'],      # 19: queen vs rooks advantage
                        eco_number,                                   # 20: ECO number
                    ], dtype=np.float32)
                    # yield features, ground_truth
                    print("Features : ", features)
                    print("Is Blunder : ", is_blunder)

                    yield_tuple = (features, is_blunder)

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

        def time_to_seconds(time_str: str) -> int:
            """Convert time string to seconds."""
            parts = time_str.split(':')
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            else:
                return int(parts[0])
