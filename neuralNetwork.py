import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers_sizes, output_size):
        self.input_size = input_size
        self.hidden_layers_sizes = hidden_layers_sizes
        self.output_size = output_size
        self.weights = []
        self.biases = []

        layer_sizes = [input_size] + hidden_layers_sizes + [output_size]

        # Initialize weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i+1])
            bias_vector = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, x):
        activations = [x]

        # Forward propagation through hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            activations.append(a)

        # Output layer
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        output = z  # No activation function for output layer
        activations.append(output)

        return output, activations

    def encode_game_state(self, board):
        # Flatten the board into a 1D array
        return np.array(board).flatten()

# Define the FNN architecture (input size, hidden layer sizes, output size)
input_size = 42  # 6 rows x 7 columns
hidden_layers_sizes = [64, 32]  # Example hidden layers with 64 and 32 neurons
output_size = 1  # Single output neuron for the score

# Create the neural network
nn = NeuralNetwork(input_size, hidden_layers_sizes, output_size)

# Example forward propagation with a dummy game state
dummy_board = np.zeros((6, 7))  # Example empty board
encoded_state = nn.encode_game_state(dummy_board)
output, activations = nn.forward_propagation(encoded_state)

print("Output score:", output)