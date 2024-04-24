import numpy as np
import math

ROW_COUNT = 6
COLUMN_COUNT = 7
PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4

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
        
    def evaluate_window(self, piece):
        score = 0
        opp_piece = PLAYER_PIECE
        if piece == PLAYER_PIECE:
            opp_piece = AI_PIECE

        if self.window.count(piece) == 4:
            score += 100
        elif self.window.count(piece) == 3 and self.window.count(EMPTY) == 1:
            score += 5
        elif self.window.count(piece) == 2 and self.window.count(EMPTY) == 2:
            score += 2

        if self.window.count(opp_piece) == 3 and self.window.count(EMPTY) == 1:
            score -= 4

        return score
    
    
    def score_position(self, piece):
        score = 0

        ## Score center column
        center_array = [int(i) for i in list(self.board[:, COLUMN_COUNT//2])]
        center_count = center_array.count(piece)
        score += center_count * 3

        ## Score Horizontal
        for r in range(ROW_COUNT):
            row_array = [int(i) for i in list(self.board[r,:])]
            for c in range(COLUMN_COUNT-3):
                window = row_array[c:c+WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        ## Score Vertical
        for c in range(COLUMN_COUNT):
            col_array = [int(i) for i in list(self.board[:,c])]
            for r in range(ROW_COUNT-3):
                window = col_array[r:r+WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        ## Score posiive sloped diagonal
        for r in range(ROW_COUNT-3):
            for c in range(COLUMN_COUNT-3):
                window = [self.board[r+i][c+i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        for r in range(ROW_COUNT-3):
            for c in range(COLUMN_COUNT-3):
                window = [self.board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        return score

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
