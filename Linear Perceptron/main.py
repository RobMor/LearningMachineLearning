import numpy as np
import matplotlib.pyplot as plt
from numpy import random, array
from perceptron import Perceptron


def check(a, b, inputs):
    return int((a*inputs[0] + b) <= inputs[1])


def accuracy(results, correct):
    return np.sum(results == correct) / results.size


def create_set(size, a, b, point_range):
    points = list()
    correct = list()

    for i in range(0, size):
        point = random.uniform(-point_range, point_range, 2)
        points.append(point)
        correct.append(check(a, b, point))

    return array(points), array(correct)


def display(input, results, a, b, point_range):
    x = np.array(range(-point_range, point_range))
    y = a * x + b
    y2 = a * x

    axes = plt.gca()
    axes.set_xlim([-point_range, point_range])
    axes.set_ylim([-point_range, point_range])

    above = input[np.where(results == 1)]
    below = input[np.where(results == 0)]

    plt.scatter(below[:, 0], below[:, 1], c='blue')
    plt.scatter(above[:, 0], above[:, 1], c='orange')

    plt.plot(x, y, c='black')
    plt.plot(x, y2, c='red')

    plt.show()


def error_display(errors):
    plt.plot(errors)
    plt.show()


if __name__ == "__main__":

    # The goal is to create a perceptron that can identify if a point is above or below the line (a * x + b)

    a_range = 5
    b_range = 50
    point_range = 100

    a = random.randint(-a_range, a_range)
    b = random.randint(-b_range, b_range)

    print('a = ' + str(a))
    print('b = ' + str(b))

    num_iter = 5000
    train_size = 1000
    test_size = 500

    train_input, train_correct = create_set(train_size, a, b, point_range)
    test_input, test_correct = create_set(test_size, a, b, point_range)

    p = Perceptron(num_iter)

    errors = p.train(train_input, train_correct)

    print('Weights = ' + str(p.weights))
    print('Bias = ' + str(p.bias))

    test_results = p.test(test_input)

    print('Accuracy = ' + str(accuracy(test_results, test_correct)))

    display(test_input, test_results, a, b, point_range)
    plt.figure()

    print('Training Errors: ' + str(errors))
    # error_display(errors)
