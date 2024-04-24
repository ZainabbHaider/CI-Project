from connect4 import Connect4Game
from neuralNetwork import NeuralNetwork
import numpy as np
import pygame
import sys
import math
# def evaluate_player(player):
#     # Simulate games and evaluate player performance
#     pass

# def genetic_algorithm(population_size, generations):
#     # Initialize population of neural networks
#     # Implement genetic algorithm for evolving the population
#     # Evaluate player fitness and evolve population
#     pass

# def main():
#     # Initialize population of players
#     # Train players using genetic algorithm
#     # Evaluate and test the best players
#     # Optionally, visualize game progress or neural network performance
#     pass

# if __name__ == "__main__":
#     main()


# Define colors
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Define game constants
ROW_COUNT = 6
COLUMN_COUNT = 7

# Weight and bias dimensions for the NN
first_layer_hidden_weights = (7, 6)
first_layer_hidden_bias = (1, 6)
second_layer_hidden_weights = (6, 40)
second_layer_hidden_bias = (1, 40)
third_layer_hidden_weights = (7, 10)
third_layer_hidden_bias = (1, 10)



# c4 = Connect4Game()

# Print the game board
def print_board(board):
    print(np.flip(board, 0))

# Draw the game board
def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == 1:
                pygame.draw.circle(screen, RED, (int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
            elif board[r][c] == 2:
                pygame.draw.circle(screen, YELLOW, (int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    pygame.display.update()

# Define the Evol_Player class
class Evol_Player:
    def __init__(self, number, first_layer_weights, first_layer_bias, second_layer_weights, second_layer_bias, third_layer_weights, third_layer_bias):
        self.number = number
        self.score = 0
        self.first_layer_weights = first_layer_weights
        self.first_layer_bias = first_layer_bias
        self.second_layer_weights = second_layer_weights
        self.second_layer_bias = second_layer_bias
        self.third_layer_weights = third_layer_weights
        self.third_layer_bias = third_layer_bias
        self.win = 0
        self.loss = 0
        self.draw = 0

    # Get the weights of the neural network
    def getWeights(self):
        return self.first_layer_weights, self.first_layer_bias, self.second_layer_weights, self.second_layer_bias, self.third_layer_weights, self.third_layer_bias

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Hyperbolic tangent activation function
def tanh(x):
    return np.tanh(x)

# Function to create an evolutionary player
def evolutionary_player(count):
    first_layer_weights = np.random.normal(0, scale=1.0, size=first_layer_hidden_weights)
    first_layer_bias = np.random.normal(0, scale=1.0, size=first_layer_hidden_bias)
    second_layer_weights = np.random.normal(0, scale=1.0, size=second_layer_hidden_weights)
    second_layer_bias = np.random.normal(0, scale=1.0, size=second_layer_hidden_bias)
    third_layer_weights = np.random.normal(0, scale=1.0, size=third_layer_hidden_weights)
    third_layer_bias = np.random.normal(0, scale=1.0, size=third_layer_hidden_bias)
    return Evol_Player(count, first_layer_weights, first_layer_bias, second_layer_weights, second_layer_bias, third_layer_weights, third_layer_bias)

# Function to predict using the neural network
def predict_nn(board, player):
    first_hidden_output = sigmoid(np.dot(board, player.first_layer_weights) + player.first_layer_bias)
    second_hidden_output = sigmoid(np.dot(first_hidden_output, player.second_layer_weights) + player.second_layer_bias)
    third_layer_output = sigmoid(np.dot(second_hidden_output, player.third_layer_weights) + player.third_layer_bias)
    
    # Find available columns
    available_columns = [col for col in range(COLUMN_COUNT) if board[5][col] == 0]
    
    # If no available columns, return -1 indicating no valid move
    if not available_columns:
        return -1
    
    # Predict output for each available column
    outputs = [third_layer_output[col] for col in available_columns]
    
    # Choose the column with the highest output
    chosen_column = available_columns[np.argmax(outputs)]
    
    return chosen_column




# Main function for running the game and evolution
# Main function for running the game and evolution
def main():
    pygame.init()

    global SQUARESIZE, width, height, RADIUS, screen

    SQUARESIZE = 100
    width = COLUMN_COUNT * SQUARESIZE
    height = (ROW_COUNT + 1) * SQUARESIZE
    RADIUS = int(SQUARESIZE / 2 - 5)

    size = (width, height)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Connect 4")
    
    c4 = Connect4Game()
    board = c4.board
    draw_board(board)
    pygame.display.update()

    myfont = pygame.font.SysFont("monospace", 75)

    population_size = 5
    num_generations = 2

    # Initialize population
    population = [evolutionary_player(count) for count in range(population_size)]

    for generation in range(num_generations):
        print("Generation:", generation)
        for player in population:
            c4 = Connect4Game()
            board = c4.board
            game_over = False
            turn = 0
            while not game_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()

                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if turn == (player.number + 1) % 2:
                            posx = event.pos[0]
                            col = int(math.floor(posx / SQUARESIZE))

                            if c4.is_valid_location(col):
                                row = c4.get_next_open_row(col)
                                c4.drop_piece(col, (player.number + 1) % 2 + 1)

                                if c4.winning_move((player.number + 1) % 2 + 1):
                                    player.loss += 1
                                    game_over = True
                                elif np.all(board != 0):
                                    player.draw += 1
                                    game_over = True

                                draw_board(board)
                                pygame.time.wait(1000)
                                turn += 1
                                turn %= 2

                if turn == player.number % 2:
                    # Use neural network to make move
                    print("NN Output:", predict_nn(board, player))  # Print NN output for debugging
                    col = predict_nn(board, player)
                    if c4.is_valid_location(col):
                        row = c4.get_next_open_row(col)
                        c4.drop_piece(col, player.number % 2 + 1)

                        if c4.winning_move(player.number % 2 + 1):
                            player.win += 1
                            game_over = True
                        elif np.all(board != 0):
                            player.draw += 1
                            game_over = True

                        draw_board(board)
                        pygame.time.wait(1000)
                        turn += 1
                        turn %= 2

            player.score = player.win - player.loss
            print("Player", player.number, "Score:", player.score)

        # Select parents for next generation
        parents = sorted(population, key=lambda x: x.score, reverse=True)[:int(population_size / 2)]

        # Crossover and mutation
        new_generation = []
        for i in range(len(parents) - 1):
            parent1, parent2 = parents[i], parents[i + 1]
            offspring1, offspring2 = create_offspring(parent1, parent2, len(population))
            new_generation.extend([offspring1, offspring2])

        population = new_generation

def create_offspring(parent1, parent2, count, mutation_rate=0.1):
    # Extract weights and biases from parents
    first_layer_weights1, first_layer_bias1, second_layer_weights1, second_layer_bias1, third_layer_weights1, third_layer_bias1 = parent1.getWeights()
    first_layer_weights2, first_layer_bias2, second_layer_weights2, second_layer_bias2, third_layer_weights2, third_layer_bias2 = parent2.getWeights()

    # Perform crossover
    crossover_point = np.random.randint(0, first_layer_weights1.shape[1])
    offspring1_weights = np.hstack((first_layer_weights1[:, :crossover_point], first_layer_weights2[:, crossover_point:]))
    offspring2_weights = np.hstack((first_layer_weights2[:, :crossover_point], first_layer_weights1[:, crossover_point:]))

    # Perform mutation
    if np.random.rand() < mutation_rate:
        mutation_point = np.random.randint(0, offspring1_weights.shape[1])
        offspring1_weights[:, mutation_point] += np.random.normal(0, scale=0.1, size=(offspring1_weights.shape[0],))
    
    if np.random.rand() < mutation_rate:
        mutation_point = np.random.randint(0, offspring2_weights.shape[1])
        offspring2_weights[:, mutation_point] += np.random.normal(0, scale=0.1, size=(offspring2_weights.shape[0],))

    # Create offspring players
    offspring1 = Evol_Player(count + 1, offspring1_weights, first_layer_bias1, second_layer_weights1, second_layer_bias1, third_layer_weights1, third_layer_bias1)
    offspring2 = Evol_Player(count + 2, offspring2_weights, first_layer_bias2, second_layer_weights2, second_layer_bias2, third_layer_weights2, third_layer_bias2)

    return offspring1, offspring2

if __name__ == "__main__":
    main()
