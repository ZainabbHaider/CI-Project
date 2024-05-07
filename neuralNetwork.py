import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers_sizes, output_size, weights, biases):
        self.input_size = input_size
        self.hidden_layers_sizes = hidden_layers_sizes
        self.output_size = output_size
        self.weights = weights
        self.biases = biases

    def forward_propagation(self, x):
        activations = [x]
        # Forward propagation through hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            activation = np.tanh(z)
            activations.append(activation)

        # Output layer
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        output = z  # No activation function for output layer
        activations.append(output)
        return output

    # def encode_game_state(self, board):
    #     # Flatten the board into a 1D array
    #     return np.array(board).flatten()

    def encode_game_state(self, board):
        # Initialize the encoded game state array
        encoded_state = []

        # Define players' colors
        player_colors = [1, 2]  # Assuming player 1 is represented by 1 and player 2 by 2

        for player_color in player_colors:
            # Calculate nearly completed rows of four discs
            for row in range(board.shape[0]):
                for col in range(board.shape[1] - 3):
                    row_values = board[row, col:col+4]
                    if np.count_nonzero(row_values == player_color) == 3 and 0 in row_values:
                        # Nearly completed row of four discs found
                        encoded_state.append(1)
                    else:
                        encoded_state.append(0)

            # Calculate nearly completed rows of two discs (vertical and diagonal)
            for row in range(board.shape[0] - 3):
                for col in range(board.shape[1]):
                    # Vertical
                    if np.count_nonzero(board[row:row+2, col] == player_color) == 1 and \
                            0 in board[row:row+2, col]:
                        encoded_state.append(1)
                    else:
                        encoded_state.append(0)

                    # Diagonal (down-right)
                    if col <= board.shape[1] - 4:
                        diagonal_values = [board[row+i, col+i] for i in range(4)]
                        if np.count_nonzero(diagonal_values == player_color) == 3 and 0 in diagonal_values:
                            encoded_state.append(1)
                        else:
                            encoded_state.append(0)

                    # Diagonal (down-left)
                    if col >= 3:
                        diagonal_values = [board[row+i, col-i] for i in range(4)]
                        if np.count_nonzero(diagonal_values == player_color) == 3 and 0 in diagonal_values:
                            encoded_state.append(1)
                        else:
                            encoded_state.append(0)

            # Calculate height of nearly completed rows of four discs
            for row in range(board.shape[0]):
                for col in range(board.shape[1] - 3):
                    row_values = board[row, col:col+4]
                    if np.count_nonzero(row_values == player_color) == 3 and 0 in row_values:
                        height = np.where(row_values == 0)[0][0]
                        encoded_state.append(height)
                    else:
                        encoded_state.append(0)

            # Calculate height of nearly completed rows of two discs (vertical and diagonal)
            for row in range(board.shape[0] - 3):
                for col in range(board.shape[1]):
                    # Vertical
                    if np.count_nonzero(board[row:row+2, col] == player_color) == 1 and \
                            0 in board[row:row+2, col]:
                        height = np.where(board[row:row+2, col] == 0)[0][0]
                        encoded_state.append(height)
                    else:
                        encoded_state.append(0)

                    # Diagonal (down-right)
                    if col <= board.shape[1] - 4:
                        diagonal_values = [board[row+i, col+i] for i in range(4)]
                        if np.count_nonzero(diagonal_values == player_color) == 3 and 0 in diagonal_values:
                            height = np.where(np.array(diagonal_values) == 0)[0][0]
                            encoded_state.append(height)
                        else:
                            encoded_state.append(0)

                    # Diagonal (down-left)
                    if col >= 3:
                        diagonal_values = [board[row+i, col-i] for i in range(4)]
                        if np.count_nonzero(diagonal_values == player_color) == 3 and 0 in diagonal_values:
                            height = np.where(np.array(diagonal_values) == 0)[0][0]
                            encoded_state.append(height)
                        else:
                            encoded_state.append(0)

        # Count total number of nearly completed rows of each player's color
        for player_color in player_colors:
            total_rows = 0
            for i in range(len(encoded_state)):
                if encoded_state[i] == 1 and encoded_state[i + len(player_colors * 21)] == player_color:
                    total_rows += 1
            encoded_state.append(total_rows)

        # Convert the encoded state into a 1D numpy array
        # print(np.concatenate((np.array(encoded_state), np.array(board).flatten())))
        return np.concatenate((np.array(encoded_state), np.array(board).flatten()))