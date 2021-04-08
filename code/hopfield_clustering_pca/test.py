import struct
import numpy as np
from matplotlib import pyplot as plt
import random
import math
import time
import sklearn
import sklearn.cluster as sk
from sklearn.utils import shuffle
from sklearn.datasets import fetch_openml


def myfunction(vector,weight,index):
    w = .0
    for x in range(0,4):
        w += vector[x]*weight[index][x]
    print(w,"---VS---",np.dot(vector,weight[index]))
    if w > 0:
        return 1
    else:
        return -1
x = [1,-1,-1,1]
y = [1,1,-1,1]
z = [-1,1,1,-1]
list = [x,y,z]
first = np.zeross()
