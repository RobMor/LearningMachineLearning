import numpy as np
from .network import Network

if __name__ == "__main__":
    net = Network(np.array([3, 3, 1])) # 3 Inputs, 4 in hidden layer, 1 output
    print(net.layers)

    print(net.test([1, 1, 1]))
