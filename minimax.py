import numpy as np
import random
import pygame
import math
from neuralNetwork import *
from finetuning import *
import sys


BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE/2 - 5)

myfont = pygame.font.SysFont("monospace", 75)

turn = random.randint(PLAYER, AI)

def create_board():
    board = np.zeros((ROW_COUNT,COLUMN_COUNT))
    return board

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[ROW_COUNT-1][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r

def print_board(board):
    print(np.flip(board, 0))

def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Check positively sloped diaganols
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # Check negatively sloped diaganols
    for c in range(COLUMN_COUNT-3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True

def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opp_piece = AI_PIECE

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4

    return score

def score_position(board, piece):
    score = 0

    ## Score center column
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    ## Score Horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r,:])]
        for c in range(COLUMN_COUNT-3):
            window = row_array[c:c+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    ## Score Vertical
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:,c])]
        for r in range(ROW_COUNT-3):
            window = col_array[r:r+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    ## Score posiive sloped diagonal
    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score

def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0

# Minimax with alpha-beta pruning
def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, 100000000000000)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -10000000000000)
            else: # Game is over, no more valid moves
                return (None, 0)
        else: # Depth is zero
            return (None, score_position(board, AI_PIECE))
    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value

    else: # Minimizing player
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value

# Minimax with neural network
def minimax_with_NN(board, depth, alpha, beta, maximizingPlayer, NN):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, 100000000000000)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -10000000000000)
            else: # Game is over, no more valid moves
                return (None, 0)
        else: # Depth is zero
            encoded_state = NN.encode_game_state_c138(board)
            return (None, NN.forward_propagation(encoded_state))

    if maximizingPlayer:
        column = None
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax_with_NN(b_copy, depth-1, alpha, beta, False, NN)[1]
            if new_score > alpha:
                alpha = new_score
                column = col
            if alpha >= beta:
                break
        return column, alpha

    else: # Minimizing player
        column = None
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax_with_NN(b_copy, depth-1, beta, beta, True, NN)[1]
            if new_score < beta:
                beta = new_score
                column = col
            if beta <= beta:
                break
        return column, beta


# Monte carlo trees search
def simulate_game(board):
    temp_board = board.copy()
    current_player = PLAYER_PIECE

    while not is_terminal_node(temp_board):
        if current_player == PLAYER_PIECE:
            col = random.choice(get_valid_locations(temp_board))
            row = get_next_open_row(temp_board, col)
            drop_piece(temp_board, row, col, PLAYER_PIECE)
            current_player = AI_PIECE
        else:
            col = random.choice(get_valid_locations(temp_board))
            row = get_next_open_row(temp_board, col)
            drop_piece(temp_board, row, col, AI_PIECE)
            current_player = PLAYER_PIECE

    if winning_move(temp_board, AI_PIECE):
        return 1
    elif winning_move(temp_board, PLAYER_PIECE):
        return 0
    else:
        return 0.5  # Draw

def monte_carlo_ai_move(board, simulations=1000):
    valid_locations = get_valid_locations(board)
    best_col = random.choice(valid_locations)
    best_score = -1

    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, AI_PIECE)

        total_score = 0
        for _ in range(simulations):
            score = simulate_game(temp_board)
            total_score += score

        average_score = total_score / simulations

        if average_score > best_score:
            best_score = average_score
            best_col = col

    return best_col

def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

# def pick_best_move(board, piece):
#     valid_locations = get_valid_locations(board)
#     best_score = -10000
#     best_col = random.choice(valid_locations)
#     for col in valid_locations:
#         row = get_next_open_row(board, col)
#         temp_board = board.copy()
#         drop_piece(temp_board, row, col, piece)
#         score = score_position(temp_board, piece)
#         if score > best_score:
#             best_score = score
#             best_col = col

#     return best_col

def simple_heuristic(board, piece):
    # print("in here")
    for col in range(COLUMN_COUNT):
        for row in range(ROW_COUNT-1, -1, -1):
            if board[row][col] == piece and is_valid_location(board, col):
                # print(board)
                # print("in here")
                return col
    return np.random.randint(0, COLUMN_COUNT)

def draw_board(board, screen):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
    
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):        
            if board[r][c] == PLAYER_PIECE:
                pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
            elif board[r][c] == AI_PIECE: 
                pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
    pygame.display.update()

def initialise_weights(input_size, hidden_layers_sizes, output_size):
    layer_sizes = [input_size] + hidden_layers_sizes + [output_size]
    weights = []
    biases = []
    # Initialize weights and biases for each layer
    for i in range(len(layer_sizes) - 1):
        weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i+1])
        bias_vector = np.zeros((1, layer_sizes[i+1]))
        weights.append(weight_matrix)
        biases.append(bias_vector)
    return weights, biases

def play_game(NN):
    game_over=False
    board = create_board()
    turn = random.randint(PLAYER, AI)
    while not game_over:
        if turn == PLAYER and not game_over:
            # col = np.random.randint(0, COLUMN_COUNT)
            # col, minimax_score = minimax(board, 5, -math.inf, math.inf, True)
            # col = monte_carlo_ai_move(board)
            col = simple_heuristic(board, PLAYER_PIECE)
            # print(col)
            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_PIECE)

                if winning_move(board, PLAYER_PIECE):
                    game_over = True
                    return 0
                if 0 not in board:
                    game_over = True
                    return 0.5

                turn += 1
                turn = turn % 2

        # # Ask for Player 2 Input
        if turn == AI and not game_over:        
            col, minimax_score = minimax_with_NN(board, 5, -math.inf, math.inf, True, NN)
            # score, col = NN.forward_propagation(board)
            

            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI_PIECE)

                if winning_move(board, AI_PIECE):
                    game_over = True
                    return 1
        
                if 0 not in board:
                    game_over = True
                    return 0.5

                turn += 1
                turn = turn % 2

def play_game_gui(NN):
    screen = pygame.display.set_mode(size)
    game_over=False
    board = create_board()
    draw_board(board, screen)
    pygame.display.update()

    myfont = pygame.font.SysFont("monospace", 75)
    turn = random.randint(PLAYER, AI)
    
    result = 0
    while not game_over:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        pygame.display.update()
        
        if turn == PLAYER and not game_over:
            # col = np.random.randint(0, COLUMN_COUNT)
            col = simple_heuristic(board, PLAYER_PIECE)
            print(col)
            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_PIECE)

                if winning_move(board, PLAYER_PIECE):
                    label = myfont.render("Player 1 wins!!", 1, RED)
                    screen.blit(label, (40,10))
                    print("player 1 wins")
                    game_over = True
                    # pygame.time.wait(3000)
                    result = 0
                if 0 not in board:
                    label = myfont.render("Draw", 1, BLUE)
                    print("Draw")
                    # print_board(board)
                    screen.blit(label, (40,10))
                    game_over = True
                    # pygame.time.wait(3000)
                    result = 0.5

                turn += 1
                turn = turn % 2

                draw_board(board, screen)
                pygame.time.wait(3000)


        # # Ask for Player 2 Input
        if turn == AI and not game_over:        
            col, _ = minimax_with_NN(board, 10, -math.inf, math.inf, True, NN)
            

            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI_PIECE)

                if winning_move(board, AI_PIECE):
                    label = myfont.render("Player 2 wins!!", 1, YELLOW)
                    screen.blit(label, (40,10))
                    game_over = True
                    # pygame.time.wait(3000)
                    result = 1
        
                if 0 not in board:
                    label = myfont.render("Draw", 1, BLUE)
                    screen.blit(label, (40,10))
                    game_over = True
                    # pygame.time.wait(3000)
                    result = 0.5

                draw_board(board,screen)
                pygame.time.wait(3000)

                turn += 1
                turn = turn % 2

        if game_over:
            print_board(board)
            # draw_board(board, screen)
            pygame.time.wait(3000)
            return result

# weights, biases = initialise_weights(INPUT_SIZE, HIDDEN_LAYERS_SIZES, OUTPUT_SIZE)
# nn = NeuralNetwork(INPUT_SIZE, HIDDEN_LAYERS_SIZES, OUTPUT_SIZE, weights, biases)
# play_game_gui(nn)