import numpy as np
from numpy import dot, random, array


class Perceptron:
    def __init__(self, num, rate=0.1):
        self.weights = random.uniform(-1, 1, 2)
        self.bias = random.uniform(-1, 1)
        self.num = num
        self.rate = rate

    def train(self, inputs, correct):
        errors = list()
        for i in range(0, self.num):
            out = self.test(inputs)
            error = correct - out
            if i % 100 == 0:
                errors.append(sum(error))
            self.weights += dot(inputs.T, error * self.rate)  # Can change the rate to train faster and with less data
            self.bias += sum(error) * self.rate  # This bias isn't doing ANYTHING
        return np.array(errors)

    def test(self, inputs):
        return heaviside(dot(inputs, self.weights) + self.bias)


def heaviside(x):
    return (x > 0).astype(int)
