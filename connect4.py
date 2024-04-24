import numpy as np
import math

class Connect4Game:
    def __init__(self):
        self.board = self.create_board()
        self.game_over = False
        self.turn = 0

    def create_board(self):
        return np.zeros((6, 7))

    def drop_piece(self, col, piece):
        for r in range(6):
            if self.board[r][col] == 0:
                self.board[r][col] = piece
                break

    def is_valid_location(self, col):
        return self.board[5][col] == 0

    def get_next_open_row(self, col):
        for r in range(6):
            if self.board[r][col] == 0:
                return r

    def winning_move(self, piece):
        # Check horizontal locations for win
        for c in range(4):
            for r in range(6):
                if self.board[r][c] == piece and self.board[r][c+1] == piece and self.board[r][c+2] == piece and self.board[r][c+3] == piece:
                    return True

        # Check vertical locations for win
        for c in range(7):
            for r in range(3):
                if self.board[r][c] == piece and self.board[r+1][c] == piece and self.board[r+2][c] == piece and self.board[r+3][c] == piece:
                    return True

        # Check positively sloped diaganols
        for c in range(4):
            for r in range(3):
                if self.board[r][c] == piece and self.board[r+1][c+1] == piece and self.board[r+2][c+2] == piece and self.board[r+3][c+3] == piece:
                    return True

        # Check negatively sloped diaganols
        for c in range(4):
            for r in range(3, 6):
                if self.board[r][c] == piece and self.board[r-1][c+1] == piece and self.board[r-2][c+2] == piece and self.board[r-3][c+3] == piece:
                    return True

    def print_board(self):
        print(np.flip(self.board, 0))

    def play(self, col):
        if self.is_valid_location(col):
            row = self.get_next_open_row(col)
            self.drop_piece(col, self.turn + 1)

            if self.winning_move(self.turn + 1):
                self.game_over = True

            self.turn += 1
            self.turn %= 2
            return True
        else:
            return False
