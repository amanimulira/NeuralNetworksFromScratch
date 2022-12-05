import numpy as np 

class Layer:
    def __init__(self) -> None:
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        raise NotImplementedError
    def backwards_propagation(self, output_error, learning_rate):
        raise NotImplementedError
