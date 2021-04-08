import numpy as np
from random import shuffle

class Network(object):
    def __init__(self):
        self.num = 784
        self.weight = np.zeros((784,784))
    def set_weights(self,W):
        self.weight = W
    def sup(self,input):
        vector = input.copy()
        i = 0
        while True:
            same = True
            for k in range(0,784):
                product = np.dot(self.weight[k],vector)
                value = 0
                if product > 0:
                    value= 1
                else:
                    value = -1
                    if value != vector[k]:
                        vector[k] = value
                        same = False
                i += 1
                if i == 50 or same:
                    return vector

    def run(self, input_data):
        vector = input_data.copy()
        i = 0
        while True:
            same = True
            for k in range(0,784):
                product = .0
                product = np.dot(self.weight[k],vector)
                value = 0
                if product > 0:
                    value= 1
                else:
                    value = -1
                if value != vector[k]:
                    vector[k] = value
                    same = False
                if same:
                    return vector
                i += 1
                if i == 60:
                    return vector
def train(network, input_data):
    my_weights = np.zeros((784,784))
    n = len(input_data)
    for x in range(0,784):
        for y in range(0,784):
            w = 0.0
            for k in range(0,len(input_data)):
                w += input_data[k][x]*input_data[k][y]
            my_weights[x][y] = (1.0 / float(n)) * w
    for x in range(0,784):
        my_weights[x][x] = 0
    network.set_weights(my_weights)
