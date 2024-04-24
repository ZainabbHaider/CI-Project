import numpy as np

class NeuralNetwork:
    def __init__(self, first_layer_hidden_weights, first_layer_hidden_bias, 
                 second_layer_hidden_weights, second_layer_hidden_bias,
                 third_layer_hidden_weights, third_layer_hidden_bias):
        self.first_layer_hidden_weights = first_layer_hidden_weights
        self.first_layer_hidden_bias = first_layer_hidden_bias
        self.second_layer_hidden_weights = second_layer_hidden_weights
        self.second_layer_hidden_bias = second_layer_hidden_bias
        self.third_layer_hidden_weights = third_layer_hidden_weights
        self.third_layer_hidden_bias = third_layer_hidden_bias

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def predict(self, board):
        first_hidden_output = self.tanh(np.dot(board, self.first_layer_hidden_weights) + self.first_layer_hidden_bias)
        second_hidden_output = self.tanh(np.dot(first_hidden_output, self.second_layer_hidden_weights) + self.second_layer_hidden_bias)
        third_layer_output = self.tanh(np.dot(second_hidden_output, self.third_layer_hidden_weights) + self.third_layer_hidden_bias)

        return np.sum(third_layer_output)
    
    def createNN(first_layer_weights, first_layer_bias, 
             second_layer_weights, second_layer_bias, 
             third_layer_weights, third_layer_bias):
        player = NeuralNetwork(first_layer_weights, first_layer_bias, 
                            second_layer_weights, second_layer_bias, 
                            third_layer_weights, third_layer_bias)
        return player

