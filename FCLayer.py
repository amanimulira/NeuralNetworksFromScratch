from fromScratch import *

class FullyConnectedLyaer(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        self.weights = np.random.rand(input_size, output_size) - 0.5 
        self.bias = np.random.rand(1, output_size) - 0.5 

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot( self.input, self.weights) + self.bias

        return self.output

    def backwards_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weight_error = np.dot(self.input.T, output_error)


        self.weights -= learning_rate * weight_error
        self.bias -= learning_rate * output_error

        return input_error

