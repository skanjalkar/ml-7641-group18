import numpy as np
import chess
import chess.pgn

np.random.seed(69)

def main():
    pgn = open("../../data/test.pgn")
    while (True):
        game = chess.pgn.read_game(pgn)
        if game is None:
            break

        move_count = 0
        # only consider moves after 10 moves
        # then with a probablity of 1/32, consider the move and store
        # the board state in a npy file
        for move in game.mainline_moves():
            print(move)
            board = game.board()
            board.push(move)
            print(board)
            break


if __name__ == "__main__":
    main()
