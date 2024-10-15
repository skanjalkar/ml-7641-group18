import numpy as np
import chess
import chess.pgn
import re


np.random.seed(69)

# Format of headers.
"""
Headers(Event='Rated Bullet game',
Site='https://lichess.org/nQ1xYNSF',
Date='2024.08.01', Round='-',
White='kingskreamer', Black='mysteryvabs',
Result='1-0',
UTCDate='2024.08.01', UTCTime='00:00:09',
WhiteElo='2148', BlackElo='2155',
WhiteRatingDiff='+6', BlackRatingDiff='-6',
ECO='B10',
Opening='Caro-Kann Defense: Accelerated Panov Attack',
TimeControl='60+0',
Termination='Time forfeit')
"""

def board_to_array(board):
    """
    # dictionary to map chess pieces to integers
    # P: 1, N: 2, B: 3, R: 4, Q: 5, K: 6 -> white pieces
    # p: -1, n: -2, b: -3, r: -4, q: -5, k: -6 -> black pieces
    # 0 for empty square
    # 8x8 board
    Sample Board:
            [[ 0  4  0  0  6  0  0  4]
            [ 1  0  0  5  0  0  1  1]
            [ 0  0  0  0  0  2  0  0]
            [ 0  0 -2  0  1  0  0  0]
            [ 0  2  0  0  0  0  0  0]
            [ 0  0  0  0  0 -1  0  0]
            [-1 -1  0 -1 -3  0 -1 -1]
            [-4  0 -3 -6  0  0 -4  0]]
    """
    piece_dict = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
    }

    board_array = np.zeros((8, 8), dtype=np.int8)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            board_array[row, col] = piece_dict[piece.symbol()]

    return board_array

def get_bin(elo):
    """
    # gets the bin for the elo rating
    """
    if 1000 <= elo < 2000:
        lower_bound = (elo // 100) * 100
        upper_bound = lower_bound + 100
        return f"{lower_bound}-{upper_bound}"
    return None


def time_to_seconds(time_str):
    # Split the time string into hours, minutes, and seconds
    parts = time_str.split(':')

    # Convert to seconds
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    else:
        return int(parts[0])

def get_clock_time(comment):
    clock_match = re.search(r'\[%clk ([\d:]+)\]', comment)
    if clock_match:
        clock_time = clock_match.group(1)
        clock_time = time_to_seconds(clock_time)
        return clock_time
    return None

def main():
    pgn = open("../../data/pgn-data/test.pgn")
    file_count = 1
    while (True):
        game = chess.pgn.read_game(pgn)
        if game is None:
            break

        move_count = 0
        # only consider moves after 20 moves (10 on each side to remove theory)
        # then with a probablity of 1/32, consider the move and store
        # the board state in a npy file
        white_elo = game.headers["WhiteElo"]
        black_elo = game.headers["BlackElo"]
        game_mode = game.headers["Event"]
        if ("Bullet" in game_mode):
            continue # Skip bullet or Hyper Bullet games
        board = game.board()
        for move in game.mainline_moves():
            node = game.variation(0)
            for _ in range(board.fullmove_number - 1):
                node = node.variation(0)
            comment = node.comment

            clock_time = get_clock_time(comment)

            if move_count >= 20 and np.random.randint(32) == 0 and clock_time is not None and clock_time >= 30:
                elo = white_elo if move_count % 2 == 0 else black_elo
                board_array = board_to_array(board)
                board.push(move)
                ground_truth_board = board_to_array(board)
                data = np.stack((board_array, ground_truth_board), axis=-1)

                # store data as 8x8x2 array, where 8x8 is board
                # and 2 is the move
                bin = get_bin(int(elo))
                if (bin is not None):
                    # np.save(f"../../data/bins/{bin}/{elo}-{file_count}.npy", data)
                    file_count += 1
            else:
                board.push(move)
            move_count += 1


if __name__ == "__main__":
    main()
