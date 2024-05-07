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

    def encode_game_state(self, board):
        # Flatten the board into a 1D array
        return np.array(board).flatten()