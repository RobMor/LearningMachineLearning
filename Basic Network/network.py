import numpy as np
#from numpy import random, dot, multiply, transpose


class Network:
    def __init__(self, layer_array):
        self.layers = [np.random.randn(current, prev) for prev, current in zip(layer_array[:-1], layer_array[1:])]
        self.biases = [np.random.randn(1, current) for current in layer_array[1:]]

    def train(self, inputs, correct): # Should I split up the inputs
        for i in range(1, 100):
            self.gradient_descent(inputs, correct)

    def gradient_descent(self, inputs, correct):
        outputs = self.compute(inputs)
        
        # Back Propogation
        error = None
        for (layer, bias), layer_out in zip(reversed(self.layers), reversed(outputs)):
            if not error:
                error = np.multiply(-(correct - layer_out), Activation.d_sigmoid(layer_out))
            else:
                weight_delta = np.dot(error, np.transpose(layer_out))
                bias_delta = error
                error = np.multiply(np.dot(np.transpose(layer), error), Activation.d_sigmoid(layer_out))
    
    def compute(self, inputs):
        outputs = list()
        out = inputs
        for layer in self.layers:
            out = Activation.sigmoid(np.dot(out, layer)) # Maybe layer dot out
            outputs.append(out)

        return outputs

    def test(self, inputs):
        outputs = self.compute(inputs)
        return outputs[-1]


class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def d_sigmoid(x): # Not the actual derivative, just gives you the slope based on the actual value
        return x * (1 - x)

    @staticmethod
    def heaviside(x):
        return (x > 0).astype(int)