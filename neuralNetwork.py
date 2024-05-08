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

    def encode_game_state_c42(self, board):
        return np.array(board).flatten()
    
    def encode_game_state_c138(self, board):
        encoded_board = np.array(board).flatten()
        def check_nearly_completed_rows(player):
            count = 0
            for row in range(6):
                for col in range(7):
                    # Horizontal nearly completed rows
                    if col <= 3 and np.sum(board[row, col:col+4] == player) == 3 and board[row, col+3] == 0:
                        count += 1
                    # Vertical nearly completed rows
                    if row <= 2 and np.sum(board[row:row+4, col] == player) == 3 and board[row+3, col] == 0:
                        count += 1
                    # Diagonal nearly completed rows (top-left to bottom-right)
                    if row <= 2 and col <= 3 and np.sum(np.diagonal(board[row:row+4, col:col+4]) == player) == 3 and board[row+3, col+3] == 0:
                        count += 1
                    # Diagonal nearly completed rows (bottom-left to top-right)
                    if row >= 3 and col <= 3 and np.sum(np.diagonal(np.flipud(board[row-3:row+1, col:col+4])) == player) == 3 and board[row-3, col+3] == 0:
                        count += 1
            return count

        def check_rows_of_two(player):
            count = 0
            for row in range(6):
                for col in range(7):
                    # Horizontal rows of two
                    if col <= 5 and np.sum(board[row, col:col+2] == player) == 2 and np.all(board[row, col:col+2] != 3-player):
                        count += 1
                    # Vertical rows of two
                    if row <= 4 and np.sum(board[row:row+2, col] == player) == 2 and np.all(board[row:row+2, col] != 3-player):
                        count += 1
                    # Diagonal rows of two (top-left to bottom-right)
                    if row <= 4 and col <= 5 and np.sum(np.diagonal(board[row:row+2, col:col+2]) == player) == 2 and np.all(np.diagonal(board[row:row+2, col:col+2]) != 3-player):
                        count += 1
                    # Diagonal rows of two (bottom-left to top-right)
                    if row >= 1 and col <= 5 and np.sum(np.diagonal(np.flipud(board[row-1:row+1, col:col+2])) == player) == 2 and np.all(np.diagonal(np.flipud(board[row-1:row+1, col:col+2])) != 3-player):
                        count += 1
            return count
        
        for player in [1, 2]:
            nearly_completed_rows_count = check_nearly_completed_rows(player)
            rows_of_two_count = check_rows_of_two(player)
            encoded_board = np.append(encoded_board, nearly_completed_rows_count)
            encoded_board = np.append(encoded_board, rows_of_two_count)
            
            # Count nearly completed rows of four discs in each column
            for col in range(7):
                col_slice = board[:, col]
                if np.sum(col_slice == player) == 3 and np.any(col_slice == 0):
                    encoded_board = np.append(encoded_board, 1)
                else:
                    encoded_board = np.append(encoded_board, 0)

            # Count empty fields in horizontal rows
            for row in range(6):
                if np.sum(board[row] == player) == 3 and np.any(board[row] == 0):
                    encoded_board = np.append(encoded_board, 1)
                else:
                    encoded_board = np.append(encoded_board, 0)

            # Count empty fields in diagonal rows
            for offset in range(-2, 4):
                diag_slice = np.diagonal(board, offset)
                for i in range(len(diag_slice) - 3):
                    if np.sum(diag_slice[i:i+4] == player) == 3 and np.any(diag_slice[i:i+4] == 0):
                        encoded_board = np.append(encoded_board, 1)
                    else:
                        encoded_board = np.append(encoded_board, 0)

            # Count empty fields in vertical rows
            for col in range(7):
                for row in range(3):
                    if np.sum(board[row:row+4, col] == player) == 3 and np.any(board[row:row+4, col] == 0):
                        encoded_board = np.append(encoded_board, 1)
                    else:
                        encoded_board = np.append(encoded_board, 0)
        return encoded_board