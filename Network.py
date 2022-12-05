from fromScratch import *

class Network(Layer):
    def __init__(self ):
        self.layers = []
    


    def add(self, layer ) -> None:

        self.layers.append(layer)

    def use(self, loss, loss_prime) -> None:

        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):

        samples = len(input_data)
        results = []

        for i in range(samples):
            output = input_data[i]

            for layer in self.layers:
                output = layer.forward_propagation(output)
            results.append(output)
            
        return results


    def fit(self, x_train, y_train, epochs, learning_rate):

        samples = len(x_train)
        for i in range(epochs):
            err = 0 
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                err += self.loss(y_train[j], output)

                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backwards_propagation(error, learning_rate)
                
                err /= samples
                print('epoch %d/%d   error=%f' % (i+1, epochs, err))



