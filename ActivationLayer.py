from fromScratch import *

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime) -> None:
        self.activation = activation
        self.activation_prime = activation_prime
    
        

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(input_data)

        return self.output

    def backwards_propagation(self, output_error, learning_rate):
        
        return self.activation_prime(self.input) * output_error