import numpy as np
from numpy import array, dot, exp, random


class Perceptron:
    def __init__(self, num):
        random.seed(1337)  # Maybe experiment with removing this
        self.weights = random.sample(num)  # Weights from -1 to 1

    def sigmoid_func(self, x):
        return 1/(1+exp(-x))

    def d_sigmoid_func(self, x):
        return x * (1 - x)

    def compute(self, X):
        return self.sigmoid_func(dot(X, self.weights))

    def train(self, X, Y, num):
        for x in range(0, num):
            out = self.compute(X)  # vector - each row corresponds to one response
            error = Y - out.T  # vector - each row corresponds to the difference between output and real value
            self.weights += dot(X.T, error * self.d_sigmoid_func(out))


def check(result, correct):
    return abs(correct - result)


if __name__ == "__main__":
    # The boolean formula ((a and b) or (c and d))
    train_X = array([[1, 1, 0, 1], [0, 0, 0, 1], [0, 1, 1, 0], [1, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 1], [0, 0, 0, 1]])
    train_Y = np.transpose(array([1, 0, 0, 1, 0, 1, 0]))

    test_X = array([[0, 0, 1, 1], [1, 0, 1, 1], [0, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
    test_Y = array([1, 1, 0, 1, 1])

    net = Perceptron(4)

    net.train(train_X, train_Y, 10000)

    print(net.weights)
    print(net.compute(test_X))
    print(abs(test_Y - net.compute(test_X)))  # Not the best...
